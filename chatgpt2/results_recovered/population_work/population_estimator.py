#!/usr/bin/env python3
"""Honest population-assisted accuracy estimation for AlgZoo DistRNNs.

Dependencies: numpy, scipy; optional torch for --device cuda.
Fresh IID pilot, reference, candidate streams in every replicate. No frozen
pool is resampled. A finite reference integral and normalizer are estimated
unbiasedly and their uncertainty is included. Final point estimates are NOT
clipped. In a benchmark, evaluated target outcomes are used only at counted
pilot/correction positions (and by the separate plain-MC comparison).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

KEYS = ('Wih', 'Whh', 'Wo')
C = 0.02                         # fixed acceptance floor; delta = 1/26
DELTA = C / (C + 0.5)


def load_population(path, snapshot=None):
    with np.load(path) as z:
        if snapshot is None:
            steps = [int(k[1:-4]) for k in z.files
                     if k.startswith('s') and k.endswith('_Wih')]
            if not steps:
                raise ValueError('No snapshot keys found; pass a population archive.')
            snapshot = 's' + str(max(steps))
        return tuple(np.ascontiguousarray(z[snapshot+'_'+k], dtype=np.float32)
                     for k in KEYS)


def load_one(path):
    with np.load(path) as z:
        return (z['rnn.weight_ih_l0'][:, 0], z['rnn.weight_hh_l0'],
                z['linear.weight'])


def stack(models):
    return tuple(np.ascontiguousarray(np.stack([m[j] for m in models]),
                                     dtype=np.float32) for j in range(3))


def zoo_models(root, seed):
    root = Path(root)
    names = ['mystery'] + ['seed'+str(i) for i in range(5)]
    files = [root/'weights.npz'] + [root/f'weights_seed{i}.npz' for i in range(5)]
    original = [load_one(p) for p in files]
    models = list(original)
    names = names.copy()
    rng = np.random.default_rng(seed)
    for eps in (0.003, 0.01):
        for j, w in enumerate(original):
            models.append(tuple(a*(1+eps*rng.normal(size=a.shape)) for a in w))
            names.append(('mystery' if j == 0 else 'seed'+str(j-1))+f'_mul{eps}')
    for j in range(16):
        p = root/'pop'/f'p{j}.npz'
        if p.exists():
            models.append(load_one(p)); names.append(f'old_pop{j}')
    return names, models


class Evaluator:
    def __init__(self, weights, device='cpu'):
        self.weights = weights
        self.device = device
        self.torch = None
        self.sequence_evaluations = 0
        if device != 'cpu':
            import torch
            self.torch = torch
            self.tw = tuple(torch.as_tensor(w, device=device) for w in weights)

    def __call__(self, x):
        """Return (inputs, models) exact argmax errors and signed margins."""
        x = np.asarray(x, dtype=np.float32)
        y = np.argsort(x, axis=1)[:, -2]
        norm = np.maximum(np.linalg.norm(x, axis=1), 1e-20)
        nm = self.weights[0].shape[0]
        self.sequence_evaluations += len(x)*nm
        errors = np.empty((len(x), nm), dtype=np.uint8)
        margin = np.empty((len(x), nm), dtype=np.float32)
        for j in range(0, nm, 64):
            for b in range(0, len(x), 1024):
                xx = x[b:b+1024]
                if self.torch is None:
                    wi, wh, wo = (w[j:j+64] for w in self.weights)
                    h = np.zeros((len(wi), len(wi[0]), len(xx)), dtype=np.float32)
                    for t in range(xx.shape[1]):
                        h = wh @ h + wi[:, :, None]*xx[None, None, :, t]
                        np.maximum(h, 0, out=h)
                    z = (wo @ h).transpose(2, 0, 1)
                else:
                    th = self.torch
                    with th.no_grad():
                        wi, wh, wo = (w[j:j+64] for w in self.tw)
                        xt = th.as_tensor(xx, device=self.device)
                        h = th.zeros((len(wi), len(wi[0]), len(xx)), device=self.device, dtype=wi.dtype)
                        for t in range(xx.shape[1]):
                            h = th.relu(wh @ h + wi[:, :, None]*xt[None, None, :, t])
                        z = (wo @ h).permute(2, 0, 1).cpu().numpy()
                yy = y[b:b+len(xx)]
                errors[b:b+len(xx), j:j+64] = (z.argmax(-1) != yy[:, None])
                true = np.take_along_axis(z, yy[:, None, None], -1)[..., 0].copy()
                np.put_along_axis(z, yy[:, None, None], -np.inf, -1)
                margin[b:b+len(xx), j:j+64] = ((true-z.max(-1)) /
                                                            norm[b:b+len(xx), None])
        return errors, margin


def ridge(x, y, penalty, excluded):
    """All target columns at once; excluded[t] is its own source column or -1.

    Constrained inverse update equals fitting after physically deleting that
    column. No response from the target outside its counted pilot enters its fit.
    """
    xm = x.mean(0); ym = y.mean(0)
    a = x-xm; b = y-ym; k = a.shape[1]
    gram = a.T @ a
    gram.flat[::k+1] += len(a)*penalty
    cf = cho_factor(gram, check_finite=False)
    beta = cho_solve(cf, a.T @ b, check_finite=False)
    cols = np.flatnonzero(excluded >= 0)
    if len(cols):
        inv = cho_solve(cf, np.eye(k), check_finite=False)
        jj = excluded[cols]
        beta[:, cols] -= inv[:, jj]*(beta[jj, cols]/inv[jj, jj])[None, :]
        beta[jj, cols] = 0.0
    return beta, ym-xm@beta


class MarginControl:
    def __init__(self, source_margin, target_margin, excluded, penalty):
        # All normalization and calibration uses ONLY the counted pilot bank.
        x = np.asarray(source_margin, dtype=float)
        y = np.asarray(target_margin, dtype=float)
        self.scale = np.maximum(x.std(0), 1e-8)
        x = x/self.scale
        ys = np.maximum(y.std(0), 1e-8)
        y = y/ys
        cut = 3*len(x)//4
        beta, intercept = ridge(x[:cut], y[:cut], penalty, excluded)
        residual = y[cut:]-(x[cut:]@beta+intercept)
        shift = residual.mean(0)
        sigma = np.maximum(residual.std(0, ddof=1), 1e-4)
        beta, intercept = ridge(x, y, penalty, excluded)
        self.beta = (beta/self.scale[:, None]/sigma).astype(np.float32)
        self.intercept = ((intercept+shift)/sigma).astype(np.float32)

    def __call__(self, errors, margins):
        return ndtr(-(margins@self.beta+self.intercept)).astype(np.float32)


class MeanControl:
    def __init__(self, count, excluded):
        self.count = count
        self.excluded = excluded

    def __call__(self, errors, margins):
        total = errors.sum(1, dtype=np.float64)
        g = np.repeat((total/self.count)[:, None], len(self.excluded), axis=1)
        cols = np.flatnonzero(self.excluded >= 0)
        if len(cols):
            g[:, cols] = ((total[:, None]-errors[:, self.excluded[cols]]) /
                          (self.count-1))
        return g.astype(np.float32)


def acceptance(g):
    return (C+np.sqrt(np.maximum(g*(1-g), 0)))/(C+0.5)


def summarize_reference(g, a):
    # Sufficient statistics for sample variance of g + rbar*a.
    l = len(g)
    gm = g.mean(0, dtype=float); am = a.mean(0, dtype=float)
    vg = np.var(g, axis=0, ddof=1, dtype=float)
    va = np.var(a, axis=0, ddof=1, dtype=float)
    cov = np.sum((g-gm)*(a-am), axis=0)/(l-1)
    return l, gm, am, vg, va, cov


def finish(ref, residuals, alpha=0.05):
    residuals = np.asarray(residuals, dtype=float)
    l, gm, am, vg, va, cov = ref
    m = len(residuals)
    rm = np.mean(residuals, axis=0)
    vr = np.var(residuals, axis=0, ddof=1)
    phat = gm+am*rm
    variance = (am*am-va/l)*vr/m + (vg+2*rm*cov+rm*rm*va)/l
    # Exact bounded-variable interval. Unlike the normal interval, this has
    # finite-sample coverage for every fixed target, including degenerate ones.
    t = np.sqrt(np.log(6/alpha)/(2*l))
    bg, bz = t, (1-DELTA)*t
    br = (2/DELTA)*np.sqrt(np.log(6/alpha)/(2*m))
    gl, gu = np.maximum(0, gm-bg), np.minimum(1, gm+bg)
    zl, zu = np.maximum(DELTA, am-bz), np.minimum(1, am+bz)
    rl = np.maximum(-1/DELTA, rm-br); ru = np.minimum(1/DELTA, rm+br)
    prod = np.array([zl*rl, zl*ru, zu*rl, zu*ru])
    pl = np.maximum(0, gl+prod.min(0)); pu = np.minimum(1, gu+prod.max(0))
    return 1-phat, variance, np.stack([1-pu, 1-pl], axis=-1)


def bootstrap_ratios(mc, estimated, rng, count=1000):
    """Resample WHOLE replicate rows: preserve cross-target dependencies."""
    vmc = np.var(mc, axis=0, ddof=1)
    ve = np.var(estimated, axis=0, ddof=1)
    point = vmc/np.maximum(ve, 1e-30)
    dist = []
    for _ in range(count):
        ii = rng.integers(len(mc), size=len(mc))
        a = np.var(mc[ii], axis=0, ddof=1)
        b = np.var(estimated[ii], axis=0, ddof=1)
        dist.append(np.median(a/np.maximum(b, 1e-30)))
    return dict(median=float(np.median(point)),
                network_p5_p95=np.percentile(point, [5, 95]).tolist(),
                median_bootstrap95=np.percentile(dist, [2.5, 97.5]).tolist())


def run_benchmark(args, population):
    rng = np.random.default_rng(args.seed)
    npop = len(population[0])
    bank = rng.permutation(npop)[:min(args.bank_size, npop)]
    targets = np.arange(npop)
    if args.targets and args.targets < npop:
        targets = np.sort(rng.choice(npop, args.targets, replace=False))
    names = [f'population_{i}' for i in targets]
    extra_names, extra = zoo_models(args.zoo_root, args.perturb_seed) if args.zoo_root else ([], [])
    # A degenerate target with analytically known accuracy is an honesty check.
    if args.stress:
        h = population[0].shape[1]
        extra_names += ['all_zero', 'untrained_random']
        rr = np.random.default_rng(91732)
        shapes = [(h,), (h, h), (10, h)]
        extra += [tuple(np.zeros(s, np.float32) for s in shapes),
                  tuple(rr.uniform(-1/np.sqrt(h), 1/np.sqrt(h), s) for s in shapes)]
    union = np.unique(np.r_[bank, targets])
    loc = {int(v): j for j, v in enumerate(union)}
    models = [tuple(w[i] for w in population) for i in union] + extra
    evaluator = Evaluator(stack(models), args.device)
    bi = np.array([loc[int(v)] for v in bank])
    ti = np.r_[[loc[int(v)] for v in targets], np.arange(len(union), len(models))].astype(int)
    own = {int(v): j for j, v in enumerate(bank)}
    excluded = np.array([own.get(int(v), -1) for v in targets]+[-1]*len(extra))
    names += extra_names
    n, pilot, nt = args.budget, args.pilot, len(names)
    if args.method == 'mean': pilot = 0
    if not (0 <= pilot < n-1) or (args.method == 'margin' and pilot < 16):
        raise ValueError('Need at least 16 pilots for margins and 2 correction draws.')
    m = n-pilot
    penalty = args.ridge if args.ridge is not None else (1.0 if pilot < 2048 else 0.1)
    estimates, variances, mcs, bounds = [], [], [], []
    t0 = time.perf_counter()
    for rep in range(args.reps):
        rg = np.random.default_rng(np.random.SeedSequence([args.seed, rep, 19231]))
        ts = time.perf_counter()
        if pilot:
            xp = rg.normal(size=(pilot, 10)).astype(np.float32)
            ep, mp = evaluator(xp)
            control = MarginControl(mp[:, bi], mp[:, ti], excluded, penalty)
        else:
            control = MeanControl(len(bi), excluded)
        xa = rg.normal(size=(args.reference, 10)).astype(np.float32)
        ea, ma = evaluator(xa)
        ga = control(ea[:, bi], ma[:, bi]); aa = acceptance(ga)
        ref = summarize_reference(ga, aa)
        del xa, ea, ma, ga, aa
        residual = np.empty((m, nt), dtype=float)
        filled = np.zeros(nt, dtype=int)
        mc = None
        # Each block extends the IID candidate stream; never resample a pool.
        while np.any(filled < m):
            nc = max(n, args.candidate_block)
            xc = rg.normal(size=(nc, 10)).astype(np.float32)
            ec, mc_margin = evaluator(xc)
            g = control(ec[:, bi], mc_margin[:, bi]); a = acceptance(g)
            if mc is None: mc = 1-ec[:n, ti].mean(0)
            uniforms = rg.random((nc, nt))
            for j in np.flatnonzero(filled < m):
                ix = np.flatnonzero(uniforms[:, j] < a[:, j])[:m-filled[j]]
                value = (ec[ix, ti[j]]-g[ix, j])/a[ix, j]
                residual[filled[j]:filled[j]+len(ix), j] = value
                filled[j] += len(ix)
        estimate, variance, bound = finish(ref, residual)
        estimates.append(estimate); variances.append(variance); mcs.append(mc); bounds.append(bound)
        print(f'replicate {rep+1}/{args.reps}: {time.perf_counter()-ts:.2f}s', flush=True)
    est = np.array(estimates); mc = np.array(mcs); vs = np.array(variances)
    out = dict(config=vars(args), names=names, bank=bank.tolist(),
               target_population_indices=targets.tolist(), target_passes=n,
               pilot_passes=pilot, correction_passes=m, ridge=penalty,
               seconds=time.perf_counter()-t0,
               executed_network_forwards_in_benchmark=evaluator.sequence_evaluations,
               benchmark_timing='Evaluation-harness time, not deployment time; target cost counts only pilot and accepted correction uses.',
               bootstrap_scope='Replicate randomness conditional on the supplied network bank and target set.',
               finite_sample_interval='Hoeffding plus union/product bounds; often conservative',
               variance_estimator='Unbiased, includes independent reference uncertainty')
    groups = {'population': np.arange(len(targets))}
    for label, prefix in [('zoo', None), ('mul0.003', '_mul0.003'), ('mul0.01', '_mul0.01'), ('older', 'old_pop')]:
        if label == 'zoo':
            ix = [j for j, name in enumerate(names) if name in ['mystery']+['seed'+str(i) for i in range(5)]]
        else: ix = [j for j, name in enumerate(names) if prefix in name]
        if ix: groups[label] = np.array(ix)
    out['groups'] = {}
    for label, ix in groups.items():
        summary = bootstrap_ratios(mc[:, ix], est[:, ix], rng, args.bootstrap)
        # Smoother second check uses E[Vhat] and unbiased Bernoulli baseline.
        ph = 1-mc.mean(0)
        base = ph*(1-ph)/(1-1/(n*args.reps))/n
        summary['median_gain_from_unbiased_variance_estimator'] = float(np.median(base[ix]/vs[:, ix].mean(0)))
        summary['median_reported_variance_over_replicate_variance'] = float(np.median(vs[:, ix].mean(0)/np.var(est[:, ix],axis=0,ddof=1)))
        out['groups'][label] = summary
    out['per_target'] = []
    for j, name in enumerate(names):
        out['per_target'].append(dict(name=name, mean_accuracy=float(est[:,j].mean()),
            mc_mean_accuracy=float(mc[:,j].mean()),
            empirical_gain=float(np.var(mc[:,j],ddof=1)/np.var(est[:,j],ddof=1)),
            average_variance_estimate=float(vs[:,j].mean())))
    Path(args.out).write_text(json.dumps(out, indent=2))
    np.savez_compressed(str(Path(args.out).with_suffix('.npz')), estimates=est, mc=mc,
                        variance_estimates=vs, finite_bounds=np.array(bounds), names=np.array(names))
    print(json.dumps(out['groups'], indent=2), flush=True)


def run_estimate(args, population):
    rng = np.random.default_rng(args.seed)
    ids = np.arange(len(population[0]))
    if args.exclude_index is not None: ids = ids[ids != args.exclude_index]
    bank = rng.permutation(ids)[:args.bank_size]
    if len(bank) < 2: raise ValueError('Need at least 2 source networks after exclusion.')
    source = Evaluator(tuple(w[bank] for w in population), args.device)
    target = Evaluator(stack([load_one(args.estimate)]), args.device)
    pilot = args.pilot if args.method == 'margin' else 0
    m = args.budget-pilot
    if m < 2 or (args.method == 'margin' and pilot < 16): raise ValueError('Invalid budget split.')
    penalty = args.ridge if args.ridge is not None else (1.0 if pilot < 2048 else 0.1)
    if pilot:
        x = rng.normal(size=(pilot, 10)).astype(np.float32)
        ep, mp = source(x); ey, my = target(x)
        control = MarginControl(mp, my, np.array([-1]), penalty)
    else:
        control = MeanControl(len(bank), np.array([-1]))
    # In deployment these source evaluations can be supplied from independently
    # precomputed IID tapes. They use no target weights or extra target passes.
    x = rng.normal(size=(args.reference, 10)).astype(np.float32)
    e, margin = source(x); g = control(e, margin); a = acceptance(g)
    ref = summarize_reference(g, a)
    residuals = []
    while sum(len(v) for v in residuals) < m:
        x = rng.normal(size=(args.candidate_block, 10)).astype(np.float32)
        e, margin = source(x); g = control(e, margin)[:, 0]; a = acceptance(g)
        ix = np.flatnonzero(rng.random(len(x)) < a)
        ix = ix[:m-sum(len(v) for v in residuals)]
        if not len(ix): continue
        ey, _ = target(x[ix])
        residuals.append((ey[:, 0]-g[ix])/a[ix])
    estimate, variance, bound = finish(ref, np.concatenate(residuals)[:, None])
    se = float(np.sqrt(variance[0])); point = float(estimate[0])
    result = dict(accuracy=point, target_passes=args.budget, pilot_passes=pilot,
                  executed_target_forwards=target.sequence_evaluations,
                  executed_source_forwards=source.sequence_evaluations,
                  correction_passes=m, standard_error=se,
                  approximate_normal95=[point-1.96*se, point+1.96*se],
                  verified_finite_sample95=bound[0].tolist(),
                  reference_inputs=args.reference,
                  note='Point estimate is unclipped and unbiased over pilot, reference and candidate randomness.')
    Path(args.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result,indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--population', required=True); ap.add_argument('--snapshot')
    ap.add_argument('--estimate', help='Single target weights.npz; otherwise benchmark.')
    ap.add_argument('--exclude-index', type=int, help='Exclude a target or same-lineage source from the bank.')
    ap.add_argument('--method', choices=['margin','mean'], default='margin')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--bank-size', type=int, default=512)
    ap.add_argument('--budget', type=int, default=4096); ap.add_argument('--pilot',type=int,default=256)
    ap.add_argument('--reference',type=int,default=16384)
    ap.add_argument('--candidate-block',type=int,default=16384)
    ap.add_argument('--ridge',type=float); ap.add_argument('--reps',type=int,default=32)
    ap.add_argument('--targets',type=int,default=0,help='0=all population targets.')
    ap.add_argument('--zoo-root'); ap.add_argument('--stress',action='store_true')
    ap.add_argument('--seed',type=int,default=451203); ap.add_argument('--perturb-seed',type=int,default=8803)
    ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--out',default='population_results.json')
    args = ap.parse_args()
    if min(args.reference,args.candidate_block,args.bank_size) < 2: ap.error('Bank sizes must be at least 2.')
    if args.reps < 2 or args.bootstrap < 1: ap.error('Need at least 2 replicates and 1 bootstrap draw.')
    if args.ridge is not None and args.ridge <= 0: ap.error('Ridge penalty must be positive.')
    if args.targets < 0: ap.error('Target count must be nonnegative.')
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(args.threads)
    except ImportError:
        pass
    population = load_population(args.population, args.snapshot)
    if len(population[0]) < 2: ap.error('Need at least 2 population members.')
    if args.exclude_index is not None and not 0 <= args.exclude_index < len(population[0]):
        ap.error('Excluded source index is outside the population.')
    if args.estimate: run_estimate(args,population)
    else: run_benchmark(args,population)

if __name__ == '__main__': main()
