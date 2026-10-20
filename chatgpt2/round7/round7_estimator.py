"""Round 7 deterministic accuracy estimator; NumPy, SciPy, threadpoolctl only.

Run from the extracted project (or pass --root PROJECT):
  python round7_estimator.py --out mf4_k2
  python round7_estimator.py --K 4 --out mf4_k4
  python round7_estimator.py --K 1 --kind both --subset ends --groups none \
      --supports 128 --refit outer --out compact
  python round7_estimator.py --kind both --no-pulse --groups none --out G1 --K 1
  python round7_estimator.py --K 16 --groups fixed --supports 128 \
      --refit outer --generate --out mf4_k16
  python round7_estimator.py --predict mf4_k2.model.npz --weights weights.npz

Fit writes a portable model, cross-fit/external predictions, and JSON metrics.
Only the first 1024 labels enter fitting. New deterministic cell caches can be
made with --generate, or supplied with --cells. No input sampling is used.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import betaincinv, ndtr, ndtri
from threadpoolctl import threadpool_limits

GAM = [.003, .01, .03, .1, .3, 1.]
ALP = np.logspace(-5, 3, 9)
ARR = np.array(list(itertools.permutations(range(10), 3)))
FREE = [[i for i in range(10) if i not in a[:2]] for a in ARR]
RANK = np.array([f.index(a[2]) for f, a in zip(FREE, ARR)])
PAIR = {0: 7, 7: 0, 1: 3, 3: 1, 2: 5, 5: 2, 4: 6, 6: 4}
QREP = np.array([f[PAIR[r]] for f, r in zip(FREE, RANK)])
FOLD = np.random.default_rng(0).permutation(1024) % 8


def read_weights(path):
    with np.load(path, allow_pickle=False) as z:
        if 'rnn.weight_ih_l0' in z:
            return [z['rnn.weight_ih_l0'][:, 0].astype(float),
                    z['rnn.weight_hh_l0'].astype(float), z['linear.weight'].astype(float)]
        if 'Wih' in z:
            return [z[k].astype(float) for k in ('Wih', 'Whh', 'Wo')]
        s = max(int(k[1:-4]) for k in z.files if k.startswith('s') and k.endswith('_Wih'))
        return [z[f's{s}_{k}'].astype(float) for k in ('Wih', 'Whh', 'Wo')]


def population(root, recipe):
    if recipe == 'unif':
        paths = ['b128_n1024_seed4', 'b128_seed0', 'b64_seed5']
        groups = [read_weights(root/'pop_unif'/f'{p}.npz') for p in paths]
        groups += [[w[None] for w in read_weights(root/'weights.npz')]]
        y = np.concatenate([np.load(root/'pop_unif'/f'{p}.acc22.npy') for p in paths]
                           + [np.array([.95304775])])
    else:
        groups = [read_weights(root/'pop1k/seed0.npz')]
        groups += [[w[None] for w in read_weights(root/p)]
                   for p in ['weights.npz']+[f'weights_seed{i}.npz' for i in range(5)]]
        y = np.load(root/'pop1k/reference.npz')['acc']
    return [np.concatenate([w[i] for w in groups]) for i in range(3)], y.astype(float)


def weight_stats(wi, wh, wo):
    f = np.stack([wi, np.diagonal(wh, axis1=1, axis2=2),
                  np.log(np.linalg.norm(wh, axis=2)+1e-9),
                  np.log(np.linalg.norm(wh, axis=1)+1e-9),
                  np.log(np.linalg.norm(wo, axis=1)+1e-9)], 2)
    return np.concatenate([f.mean(1), f.std(1), f.min(1), f.max(1)], 1)


def pulse_features(wi, wh, wo):
    """Two signed impulse responses, all ages; ten features, budget six FE."""
    h = np.maximum(np.stack([wi, -wi], 2), 0)
    logits = []
    for age in range(10):
        logits.append(wo@h)
        if age < 9:
            h = np.maximum(wh@h, 0)
    lg = np.stack(logits, 1)
    centered = lg-lg.mean(2, keepdims=True)
    amp = np.maximum(np.sqrt((centered*centered).mean(2)), 1e-12)
    age, target = np.arange(10), 9-np.arange(10)
    correct = lg.argmax(2) == target[None, :, None]
    chosen = lg[:, age, target, :].copy()
    lg[:, age, target, :] = -np.inf
    margin = (chosen-lg.max(2))/amp
    growth = np.diff(np.log(amp), axis=1)
    return np.concatenate([margin.mean(1), correct.mean(1), margin[:, 5:].mean(1),
                           growth.mean(1), growth.std(1)], 1)


def arrangement_ids(spec):
    return (np.arange(720) if spec['subset'] == 'all'
            else np.flatnonzero(np.isin(RANK, [0, 7])))


def temporal_masks(ar):
    M, S, R = ARR[ar].T
    return [M == 9, S == 9, M == S+1, M > S, R > S, abs(M-S) >= 5]


def make_plan(spec, ar):
    """All geometry, input values, and trie bookkeeping are prepared offline."""
    k, kind = spec['K'], spec['kind']
    fq = betaincinv(7, 4, .5)
    fr = fq+(1-fq)*betaincinv(1, 3, .5)
    fm = fr+(1-fr)*betaincinv(2, 1, .5)
    q, r, m = ndtri([fq, fr, fm])
    ss = ndtri(fr+(fm-fr)*(np.arange(k)+.5)/k)
    mu = lambda v: -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
    types = [4, 3] if kind == 'both' else [int(kind)]
    rows, target = [], []
    for a in ar:
        for s in ss:
            for typ in types:
                x = np.full(10, mu(q if typ == 4 else r))
                x[ARR[a]] = [m, s, r]
                if typ == 4:
                    x[QREP[a]] = q
                rows.append(x)
                target.append(ARR[a, 1])
    x = np.array(rows, dtype=np.float32)
    vals, tok = np.unique(x, return_inverse=True)
    tok = tok.reshape(x.shape)
    par = np.zeros(len(x), int)
    plan = {'values': vals, 'target': np.array(target), 'ar': ar}
    for t in range(10):
        key, par = np.unique(par*len(vals)+tok[:, t], return_inverse=True)
        plan[f'parent{t}'], plan[f'token{t}'] = key//len(vals), key % len(vals)
    plan['leaves'] = par
    plan.update({f'group{i}': mask for i, mask in enumerate(temporal_masks(ar))})
    return plan


def prefix_cells(w, plan, dtype=np.float64):
    """Batched targets; input products cached once per distinct scalar."""
    wi, wh, wo = [np.asarray(a, dtype=dtype) for a in w]
    inp = wi[:, None, :]*plan['values'].astype(dtype)[None, :, None]
    h = None
    for t in range(10):
        par, tok = plan[f'parent{t}'], plan[f'token{t}']
        h = (np.maximum(inp[:, tok], 0) if h is None
             else np.maximum((h@wh.transpose(0, 2, 1))[:, par]+inp[:, tok], 0))
    lg = (h@wo.transpose(0, 2, 1))[:, plan['leaves']]
    target = plan['target']
    bits = lg.argmax(2) == target[None]
    chosen = lg[:, np.arange(len(target)), target].copy()
    lg[:, np.arange(len(target)), target] = -np.inf
    return bits, chosen-lg.max(2)


def summarize(bits, margins, stats, spec, ar, pulse=None, masks=None):
    """One shared normalization/softsign pass; H requires no extra network rows."""
    n, k = len(bits), spec['K']
    nr = 2 if spec['kind'] == 'both' else 1
    b, m = bits.reshape(n, -1, nr), margins.reshape(n, -1, nr).astype(float)
    s = np.maximum(np.sqrt((m[:, :, 0]**2).mean(1, keepdims=True)), 1e-15)
    a = m/s[:, :, None]
    aa = abs(a)
    soft = a/(.1+aa)
    f = np.concatenate([b, a, aa, soft, a/(.03+aa)], 2)
    base = np.c_[stats, s, np.log(s), m.mean(1), f.mean(1),
                 f.reshape(n, len(ar), k, -1).mean(1).reshape(n, -1)]
    if pulse is not None:
        base = np.c_[base, pulse]
    if spec['groups'] == 'none':
        return base, base.shape[1]
    masks = temporal_masks(ar) if masks is None else masks
    v = np.concatenate([b, soft], 2).reshape(n, len(ar), k, 2*nr)
    groups = np.concatenate([v[:, mask].mean(1).reshape(n, -1)
                             for mask in masks], 1)
    return np.c_[base, groups], base.shape[1]


def cached_features(root, spec, cells_path=None, generate=False, cache_out=None):
    w, y = population(root, spec['recipe'])
    ar, k = arrangement_ids(spec), spec['K']
    default = (root/f'pop_unif/cells_k{k}.npz' if spec['recipe'] == 'unif'
               else root/f'pop1k/seed0_z46_cells_b1_k{k}.npz')
    path = Path(cells_path) if cells_path else default
    types = [4, 3] if spec['kind'] == 'both' else [int(spec['kind'])]
    if path.exists():
        z = dict(np.load(path, allow_pickle=False))
    else:
        if not generate:
            raise FileNotFoundError(f'{path}; use --cells PATH or --generate')
        plan = make_plan(spec, np.arange(720))
        rows = [prefix_cells([a[i:i+24] for a in w], plan, np.float32)
                for i in range(0, len(y), 24)]
        b, m = [np.concatenate([r[j] for r in rows]) for j in range(2)]
        z = {f'{prefix}{typ}': value[:, j::len(types)]
             for j, typ in enumerate(types)
             for prefix, value in [('I', b), ('m', m)]}
        np.savez_compressed(cache_out, **z)
    for typ in types:
        for prefix in ('I', 'm'):
            if z[f'{prefix}{typ}'].shape != (len(y), 720*k):
                shape = z[f'{prefix}{typ}'].shape
                raise ValueError(
                    f'Wrong recipe/K or incomplete cache: {prefix}{typ} has shape {shape}')
    ci = (ar[:, None]*k+np.arange(k)).ravel()
    bits = np.stack([z[f'I{t}'][:, ci] for t in types], 2).reshape(len(y), -1)
    margins = np.stack([z[f'm{t}'][:, ci] for t in types], 2).reshape(len(y), -1)
    p = pulse_features(*w) if spec['pulse'] else None
    x, db = summarize(bits, margins, weight_stats(*w), spec, ar, p)
    return x, y, db, make_plan(spec, ar)


def candidates(spec):
    out = [('linear', 0., 0., float(a)) for a in ALP]
    weights = [0., .1, 1.] if spec['groups'] == 'tune' else [0.]
    return out + [('rbf', w, g, float(a))
                  for w in weights for g in GAM for a in ALP]


def metric(a, b, db, mode, weight):
    if mode != 'tune':
        return cdist(a, b, 'sqeuclidean')/a.shape[1]
    return (cdist(a[:, :db], b[:, :db], 'sqeuclidean')/db
            + weight*cdist(a[:, db:], b[:, db:], 'sqeuclidean')/(a.shape[1]-db))


def kernel_fit(a, y, db, spec, weight, gamma):
    """Offline eigensystems; Nystrom uses only a fixed subset of training rows."""
    ids = (np.random.default_rng(1701).permutation(len(a))
           [:min(spec['supports'], len(a))]
           if spec['supports'] else np.arange(len(a)))
    centers = a[ids]
    km = np.exp(-gamma*metric(centers, centers, db, spec['groups'], weight))
    e, v = eigh(km)
    if not spec['supports']:
        return centers, v@((v.T@y)[:, None]/(np.maximum(e, 0)[:, None]+ALP))
    keep = e > max(1e-12, e[-1]*1e-12)
    whitening = v[:, keep]/np.sqrt(e[keep])
    phi = np.exp(-gamma*metric(a, centers, db, spec['groups'], weight))@whitening
    e, v = eigh(phi.T@phi)
    coef = whitening@v@(
        (v.T@(phi.T@y))[:, None]/(np.maximum(e, 0)[:, None]+ALP))
    return centers, coef


def candidate_predictions(x, y, tr, te, db, spec):
    mean, sd = x[tr].mean(0), np.maximum(x[tr].std(0), 1e-6)
    a, b = (x[tr]-mean)/sd, (x[te]-mean)/sd
    ym = y[tr].mean()
    yc = y[tr]-ym
    dl = db if spec['groups'] == 'tune' else x.shape[1]
    e, v = eigh(a[:, :dl].T@a[:, :dl])
    coef = v@((v.T@(a[:, :dl].T@yc))[:, None]
              /(np.maximum(e, 0)[:, None]+len(tr)*ALP))
    out = [b[:, :dl]@coef]
    for weight in ([0., .1, 1.] if spec['groups'] == 'tune' else [0.]):
        for gamma in GAM:
            centers, coef = kernel_fit(a, yc, db, spec, weight, gamma)
            out.append(np.exp(
                -gamma*metric(b, centers, db, spec['groups'], weight))@coef)
    return ym+np.concatenate(out, 1)


def crossfit(x, y, db, spec, refit):
    pop = np.arange(1024)
    pred = np.empty(len(y))
    picks = []
    outer_loss = 0
    for f in range(8):
        tr, te = pop[FOLD != f], pop[FOLD == f]
        inner = (np.arange(len(tr))*4//len(tr))[
            np.argsort(np.argsort(tr*7919 % 1024))]
        loss = sum(
            ((candidate_predictions(
                x, y, tr[inner != j], tr[inner == j], db, spec)
              - y[tr[inner == j], None])**2).sum(0)
            for j in range(4))
        pr = candidate_predictions(x, y, tr, te, db, spec)
        pick = int(loss.argmin())
        pred[te] = pr[:, pick]
        picks.append(pick)
        outer_loss = outer_loss+((pr-y[te, None])**2).sum(0)
    if refit == 'index':
        outer_loss = sum(
            ((candidate_predictions(
                x, y, pop[pop % 8 != j], pop[pop % 8 == j], db, spec)
              - y[pop[pop % 8 == j], None])**2).sum(0)
            for j in range(8))
    choice = int(outer_loss.argmin())
    pred[1024:] = candidate_predictions(
        x, y, pop, np.arange(1024, len(y)), db, spec)[:, choice]
    return pred, picks, choice


def compile_model(x, y, db, spec, choice):
    """One selected readout is compiled; query-time prediction never fits."""
    family, weight, gamma, alpha = candidates(spec)[choice]
    mean, sd = x.mean(0), np.maximum(x.std(0), 1e-6)
    a, yc, ym = (x-mean)/sd, y-y.mean(), y.mean()
    if family == 'linear':
        dl = db if spec['groups'] == 'tune' else x.shape[1]
        e, v = eigh(a[:, :dl].T@a[:, :dl])
        beta = v@((v.T@(a[:, :dl].T@yc))/(np.maximum(e, 0)+len(y)*alpha))
        coef = np.r_[beta/sd[:dl], np.zeros(x.shape[1]-dl)]
        return {'family': np.array(family), 'coef': coef,
                'intercept': ym-mean@coef}
    centers, coef = kernel_fit(a, yc, db, spec, weight, gamma)
    coord = np.full(x.shape[1], 1/np.sqrt(x.shape[1]))
    if spec['groups'] == 'tune':
        coord = np.r_[
            np.full(db, 1/np.sqrt(db)),
            np.full(x.shape[1]-db, np.sqrt(weight/(x.shape[1]-db)))]
    return {'family': np.array(family), 'mean': mean, 'scale': coord/sd,
            'centers': centers*coord,
            'coef': coef[:, int(np.argmin(abs(ALP-alpha)))],
            'gamma': gamma, 'intercept': ym}


def predict_features(model, x):
    if str(model['family']) == 'linear':
        return np.asarray(x)@model['coef']+model['intercept']
    a = (np.atleast_2d(x)-model['mean'])*model['scale']
    return model['intercept']+np.exp(
        -model['gamma']*cdist(a, model['centers'], 'sqeuclidean'))@model['coef']


def predict_weights(model, weights):
    """Actual deterministic target query: cached prefix inputs + one readout."""
    spec = json.loads(str(model['spec']))
    shapes = [(16,), (16, 16), (10, 16)]
    if len(weights) != 3 or any(
            np.asarray(w).shape != shape for w, shape in zip(weights, shapes)):
        raise ValueError('One target requires weight shapes (16,), (16,16), (10,16)')
    w = [np.asarray(v)[None] for v in weights]
    b, m = prefix_cells(w, model)
    p = pulse_features(*w) if spec['pulse'] else None
    masks = [model[f'group{i}'] for i in range(6)]
    x, _ = summarize(b, m, weight_stats(*w), spec, model['ar'], p, masks)
    return float(predict_features(model, x)[0])


def costs(spec, plan, d, db, model):
    n = np.array([len(plan[f'parent{t}']) for t in range(10)])
    net = (256*n[:-1].sum()+16*len(plan['values'])+160*n[-1]
           +16*n[1:].sum()+16*n.sum()+20*n[-1])/2880
    k, ar = spec['K'], plan['ar']
    cells = k*len(ar)
    ops = (40 if spec['kind'] == 'both' else 22)*cells+5000
    if spec['groups'] != 'none':
        masks = [plan[f'group{i}'] for i in range(6)]
        channels = 4 if spec['kind'] == 'both' else 2
        ops += channels*k*sum(int(m.sum()) for m in masks)+channels*6*k
    feature = ops/2880
    pulse = 6. if spec['pulse'] else 0.
    ns = spec['supports'] or 1024
    budget = (2*ns*d+34*ns+2*d+3)/2880
    actual = (d+1)/2880 if str(model['family']) == 'linear' else budget
    return dict(
        anchor_rows=len(plan['target']),
        pulse_rows=2*int(spec['pulse']),
        prefix_fe=float(net), feature_fe=feature, pulse_fe=pulse,
        readout_budget_fe=budget, compiled_readout_fe=actual,
        total_fe=net+feature+pulse+budget,
        compiled_total_fe=net+feature+pulse+actual)


def metrics(pred, y, cost, recipe):
    groups = {'crossfit': slice(0, 1024)}
    groups.update(
        {'b128': slice(1024, 1088), 'b64': slice(1088, 1152),
         'mystery': slice(1152, 1153)}
        if recipe == 'unif'
        else {'mystery': slice(1024, 1025), 'seeds': slice(1025, 1030)})
    out = {}
    for name, sl in groups.items():
        e = pred[sl]-y[sl]
        mse = np.mean(e*e)
        out[name] = dict(
            rms_points=float(100*np.sqrt(mse)),
            mean_points=float(100*e.mean()),
            mse_ratio=float(mse*cost/np.mean(y[sl]*(1-y[sl]))))
        if name in ('mystery', 'seeds'):
            out[name]['errors_points'] = (100*e).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, default=Path('round7_result'))
    ap.add_argument('--recipe', choices=['unif', 'zoo'], default='unif')
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--kind', choices=['both', '4', '3'], default='4')
    ap.add_argument('--subset', choices=['all', 'ends'], default='all')
    ap.add_argument('--pulse', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--groups', choices=['none', 'fixed', 'tune'], default='tune')
    ap.add_argument('--supports', type=int, choices=[0, 128], default=0)
    ap.add_argument('--refit', choices=['outer', 'index'], default='index')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--cells', type=Path)
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--predict', type=Path)
    ap.add_argument('--weights', type=Path, default=Path('weights.npz'))
    args = ap.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.predict:
            with np.load(args.predict, allow_pickle=False) as model:
                print(json.dumps({
                    'estimated_accuracy':
                    predict_weights(model, read_weights(args.weights))}))
            return
        if args.K < 1:
            ap.error('--K must be positive')
        spec = {k: getattr(args, k) for k in [
            'recipe', 'K', 'kind', 'subset', 'pulse', 'groups', 'supports']}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        x, y, db, plan = cached_features(
            args.root, spec, args.cells, args.generate,
            args.out.with_suffix('.cells.npz'))
        pred, picks, choice = crossfit(x, y, db, spec, args.refit)
        model = compile_model(x[:1024], y[:1024], db, spec, choice)
        # Compare compilation with the selected candidate's external predictions.
        discrepancy = float(np.max(abs(
            predict_features(model, x[1024:])-pred[1024:])))
        if discrepancy > 1e-8:
            raise ArithmeticError(f'Compiled readout mismatch: {discrepancy}')
        ledger = costs(spec, plan, x.shape[1], db, model)
        result = dict(
            spec=spec, refit=args.refit,
            dimensions=x.shape[1], base_dimensions=db,
            costs=ledger,
            metrics=metrics(pred, y, ledger['total_fe'], args.recipe),
            outer_choices=[candidates(spec)[i] for i in picks],
            refit_choice=candidates(spec)[choice],
            compile_max_error=discrepancy,
            offline_seconds=time.monotonic()-started)
        np.savez_compressed(
            args.out.with_suffix('.model.npz'),
            **model, **plan, spec=np.array(json.dumps(spec)))
        np.savez_compressed(
            args.out.with_suffix('.predictions.npz'), pred=pred, y=y, X=x)
        args.out.with_suffix('.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
