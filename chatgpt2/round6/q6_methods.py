"""Q6 deterministic accuracy estimators: NumPy/SciPy, one standalone file.

Training reads the ORIGINAL archive's NPZ files; deployment needs only a
weight NPZ and fitted model NPZ. Target inputs are deterministic quadrature
or population-learned fixed probes. No target samples or quasi-random points.

Examples (set OPENBLAS_NUM_THREADS=1 before running):
 python q6_methods.py weight REPO weight_model.npz --cv
 python q6_methods.py summary REPO global_k1.npz --k 1 --cv
 python q6_methods.py subset REPO subset16.npz --count 16 --cv
 python q6_methods.py subset REPO subset16_rbf.npz --count 16 --kernel rbf --cv
 python q6_methods.py cache REPO screen.npz --k 1 --screen all
 python q6_methods.py screens REPO screens32.npz --cache screen.npz --count 32 --cv
 python q6_methods.py cache REPO large.npz --boxes 5 --k 8 --summary-only
 python q6_methods.py summary REPO large_model.npz --cache large.npz --cv
 python q6_methods.py compiled REPO compiled_k1.npz --k 1
 python q6_methods.py last REPO last_q1.npz --q 1
 python q6_methods.py summary REPO combined.npz --k 1 --last last_q1.npz --cv
 python q6_methods.py predict weights.npz global_k1.npz

The last command also loads models produced by the individual Q6 scripts;
provide --k/--boxes/--screen/--context when an older model lacks metadata.
Cost unit: 2,880 multiply-adds. Prefix arithmetic is counted explicitly.
Other helpers use stated conservative scalar-operation allowances. Exp is
charged 32 operations and Gaussian CDF 100; these are accounting assumptions.
CV for global models nests family/regularization selection. Subset/screen
CV relearns the design in every outer fold with fixed stated hyperparameters.
"""
import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import betaincinv, ndtr, ndtri, roots_jacobi

ALPHAS = np.logspace(-3, 3, 9)
GAMMAS = [.03, .1, .3, 1., 3.]
FEATURE_COST = dict(raw20=2600, readoutgauge=7000, impulse=44000,
                    raw_readout=9600, raw_impulse=46600)


# ---------- Deterministic quadrature design and exact shared prefixes ----------

def design(k=1, boxes=1, screen=False):
    """Return z46's exact float32 input rows, targets, and cell count."""
    arr = np.array(list(itertools.permutations(range(10), 3)))
    qof = {0: 7, 7: 0, 1: 3, 3: 1, 2: 5, 5: 2, 4: 6, 6: 4}
    free = [[p for p in range(10) if p not in a[:2]] for a in arr]
    qrep = np.array([f[qof[f.index(a[2])]] for f, a in zip(free, arr)])
    low6 = np.array([[p for p in range(10) if p not in a and p != q]
                     for a, q in zip(arr, qrep)])
    un = (np.array([-.9602898564975363, .9602898564975363])+1)/2
    rows = 14 if screen else 2
    ai, ki = np.arange(720)[:, None], np.arange(k)[None, :]
    beta = [(7, 4), (1, 3), (2, 1)]
    all_x = []
    for b in range(boxes**3):
        half = [(b // boxes**i) % boxes for i in range(3)]
        u = [betaincinv(*beta[i], (half[i] + .5) / boxes) for i in range(3)]
        fq = u[0]
        fr = fq + (1 - fq) * u[1]
        fm = fr + (1 - fr) * u[2]
        q, r, m = ndtri([fq, fr, fm])
        s = ndtri(fr + (fm - fr) * (np.arange(k) + .5) / k)
        mu = lambda v: -np.exp(-v*v/2) / np.sqrt(2*np.pi) / ndtr(v)
        x = np.empty((720, k, rows, 10))
        for row, lo in [(0, q), (1, r)]:
            x[:, :, row] = mu(lo)
            x[ai, ki, row, arr[:, 0, None]] = m
            x[ai, ki, row, arr[:, 1, None]] = s
            x[ai, ki, row, arr[:, 2, None]] = r
        x[ai, ki, 0, qrep[:, None]] = q
        if screen:
            for j in range(6):
                for h in range(2):
                    row = 2+2*j+h
                    x[:, :, row] = x[:, :, 0]
                    x[ai, ki, row, low6[:, j, None]] = ndtri(fq*un[h])
        all_x.append(x.astype(np.float32).reshape(-1, 10))
    x = np.concatenate(all_x)
    target = np.tile(np.repeat(arr[:, 1], rows*k), boxes**3)
    return x, target, len(x) // rows


class PrefixEvaluator:
    def __init__(self, x):
        self.values, token = np.unique(x, return_inverse=True)
        token = token.reshape(x.shape)
        parent = np.zeros(len(x), dtype=np.int64)
        self.levels = []
        for t in range(x.shape[1]):
            nodes, parent = np.unique(
                np.column_stack([parent, token[:, t]]), axis=0,
                return_inverse=True)
            self.levels.append((nodes[:, 0], nodes[:, 1]))
        self.leaf = parent
        self.counts = [len(p) for p, _ in self.levels]

    def __call__(self, wih, whh, wo, dtype=np.float64):
        wih, whh, wo = [np.asarray(w, dtype=dtype) for w in (wih, whh, wo)]
        # Reuse W_ih*x across occurrences of the same design scalar.
        inp = self.values.astype(dtype)[:, None] * wih
        p, v = self.levels[0]
        h = np.maximum(inp[v], 0)  # h_0 = 0.
        for p, v in self.levels[1:]:
            # Reuse W_hh*h among every child of the same parent.
            recurrent = h @ whh.T
            h = np.maximum(recurrent[p] + inp[v], 0)
        return (h @ wo.T)[self.leaf]

    def cost(self):
        n = np.array(self.counts)
        recurrent_mac = 256 * int(n[:-1].sum())
        input_mult = 16 * len(self.values)
        output_mac = 160 * int(n[-1])
        state_add = 16 * int(n[1:].sum())
        relu_compare = 16 * int(n.sum())
        output_scalar = 20 * int(n[-1])
        mac = recurrent_mac + input_mult + output_mac
        other = state_add + relu_compare + output_scalar
        return dict(logical_rows=len(self.leaf), prefix_counts=self.counts,
                    recurrent_mac=recurrent_mac, input_mult=input_mult,
                    output_mac=output_mac, mac_forward_equiv=mac/2880,
                    scalar_ops=other, scalar_forward_equiv=other/2880,
                    total_forward_equiv=(mac+other)/2880,
                    saving=len(self.leaf)*2880/(mac+other))


def load_models(repo):
    d = np.load(repo / 'pop1k/seed0.npz')
    p = [d['s1048576_' + key] for key in ('Wih', 'Whh', 'Wo')]
    zoo = []
    for fname in ['weights.npz'] + [f'weights_seed{i}.npz' for i in range(5)]:
        w = np.load(repo / fname)
        zoo.append([w['rnn.weight_ih_l0'][:, 0], w['rnn.weight_hh_l0'],
                    w['linear.weight']])
    return [np.concatenate([p[i], np.stack([w[i] for w in zoo])])
            for i in range(3)]


def load_weights(path):
    d = np.load(path)
    return [d['rnn.weight_ih_l0'][:, 0],
            d['rnn.weight_hh_l0'], d['linear.weight']]


def bits_margins(logits, target):
    bits = logits.argmax(-1) == target
    margin = logits[np.arange(len(target)), target].copy()
    logits[np.arange(len(target)), target] = -np.inf
    return bits, margin-logits.max(-1)


# ---------- Weight summaries and two gauge constructions ----------

def raw20(wi, A, O):
    wi, A, O = [np.asarray(w, dtype=float) for w in (wi, A, O)]
    if wi.ndim == 1:
        wi, A, O = wi[None], A[None], O[None]
    F = np.stack([
        wi, np.diagonal(A, axis1=1, axis2=2),
        np.log(np.linalg.norm(A, axis=2)+1e-9),
        np.log(np.linalg.norm(A, axis=1)+1e-9),
        np.log(np.linalg.norm(O, axis=1)+1e-9)
    ], axis=2)
    return np.concatenate(
        [F.mean(1), F.std(1), F.min(1), F.max(1)], axis=1)


def pool(x):
    return np.concatenate(
        [x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)


def readout_gauge(wi, A, O):
    eps = 1e-30
    Z = O-O.mean(1, keepdims=True)
    rn = np.linalg.norm(Z, axis=1)
    rn = np.where(rn > 0, rn, 1.)
    R = A*rn[:, :, None]/rn[:, None, :]
    U = Z/rn[:, None, :]
    b = wi*rn
    bn = np.linalg.norm(b, axis=1)
    b /= np.where(bn > 0, bn, 1.)[:, None]
    nodes = [
        b, np.diagonal(R, axis1=1, axis2=2),
        np.log(np.linalg.norm(R, axis=2)+eps),
        np.log(np.linalg.norm(R, axis=1)+eps)
    ]
    nodes.extend(U[:, j] for j in range(10))
    return pool(np.stack(nodes, axis=2))


def impulse_gauge(wi, A, O):
    eps = 1e-30
    v = np.stack([np.maximum(wi, 0), np.maximum(-wi, 0)], axis=2)
    g2 = np.sum(v*v, axis=2)
    for _ in range(9):
        v = A@v
        g2 += np.sum(v*v, axis=2)
    g = np.sqrt(np.maximum(g2, eps))
    C = A*g[:, None, :]/g[:, :, None]
    B = (O-O.mean(1, keepdims=True))*g[:, None, :]
    B /= np.maximum(np.linalg.norm(B, axis=(1, 2)), eps)[:, None, None]
    V = B.copy()
    f = []
    for age in range(10):
        target = 9-age
        rival = np.max(V[:, np.arange(10) != target], axis=1)
        scale = np.maximum(np.linalg.norm(V, axis=1), eps)
        f.extend([(V[:, target]-rival)/scale,
                  (V.argmax(1) == target).astype(float)])
        if age < 9:
            V = V@C
    q = np.stack(f, axis=2)
    q = np.concatenate([
        q, q[:, :, 0::2].mean(2, keepdims=True),
        q[:, :, 0::2].min(2, keepdims=True),
        q[:, :, 1::2].mean(2, keepdims=True)
    ], axis=2)
    return pool(q)


def feature_set(wi, A, O, name='raw_readout'):
    wi, A, O = [np.asarray(w, dtype=float) for w in (wi, A, O)]
    if wi.ndim == 1:
        wi, A, O = wi[None], A[None], O[None]
    if name == 'raw20':
        return raw20(wi, A, O)
    if name == 'readoutgauge':
        return readout_gauge(wi, A, O)
    if name == 'impulse':
        return impulse_gauge(wi, A, O)
    q = raw20(wi, A, O)
    if name == 'raw_readout':
        return np.concatenate([q, readout_gauge(wi, A, O)], axis=1)
    if name == 'raw_impulse':
        return np.concatenate([q, impulse_gauge(wi, A, O)], axis=1)
    raise ValueError(name)


# Gauge guarantees assume the normalizing energies/norms are nonzero.
# Fallbacks define finite features for degenerate networks.
# Mixed raw+gauge features retain raw20's rescaling dependence.

def weight_fit(wi, A, O, accuracy, alpha=.01, gamma=.01):
    X = raw20(wi, A, O)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    Z = (X-mu)/sd
    n = len(Z)
    zz = np.sum(Z*Z, axis=1)
    D = np.maximum(zz[:, None]+zz[None]-2*Z@Z.T, 0)
    K = np.exp(-gamma*D)
    K.flat[::n+1] += alpha
    y = np.asarray(accuracy, dtype=float)
    offset = y.mean()
    return dict(
        mean=mu, scale=sd, train=Z, train_norm=zz,
        coefficient=np.linalg.solve(K, y-offset),
        offset=np.array(offset), gamma=np.array(gamma),
        alpha=np.array(alpha))


def weight_predict(model, wi, A, O):
    Z = (raw20(wi, A, O)-model['mean'])/model['scale']
    D = np.maximum(
        np.sum(Z*Z, axis=1)[:, None]+model['train_norm'][None]
        -2*(Z@model['train'].T), 0)
    return np.exp(-model['gamma']*D)@model['coefficient']+model['offset']


# ---------- Global summaries and train-only standardized regression ----------

def summarize(d, screen='none', outer=True):
    """Works on one-network dictionaries for streaming large designs."""
    k, nb = int(d['K']), int(d['NB'])
    m4, m3 = np.asarray(d['m4'], float), np.asarray(d['m3'], float)
    n, c = m4.shape
    scale = np.sqrt((m4*m4).mean(1, keepdims=True))
    scale = np.where(scale > 0, scale, 1.)
    a, b = m4/scale, m3/scale
    i4, i3 = np.asarray(d['I4'], float), np.asarray(d['I3'], float)
    f = np.stack([
        i4, i3, a, b, abs(a), abs(b),
        a/(.1+abs(a)), b/(.1+abs(b)),
        a/(.03+abs(a)), b/(.03+abs(b))
    ], -1)
    ids = np.arange(c)
    box, sl = ids//(720*k), ids % k
    groups = [np.zeros(c, int), sl]
    if outer and nb > 1:
        groups += [(box//nb**j) % nb for j in range(3)]
    out = [scale, np.log(scale),
           m4.mean(1, keepdims=True), m3.mean(1, keepdims=True)]
    for g in groups:
        out += [f[:, g == j].mean(1) for j in np.unique(g)]
    selected = np.zeros(c, bool)
    if screen != 'none':
        arr = np.array(list(itertools.permutations(range(10), 3)))
        arr = arr[(ids//k) % 720]
        if screen == 'all':
            selected[:] = True
        elif screen == 'late':
            selected = arr[:, 0] == 9
        elif screen.startswith('stride'):
            selected = ids % int(screen[6:]) == 0
        else:
            raise ValueError(screen)
        fl = np.asarray(d['fl'][:, selected], float)
        fm = np.asarray(d['fm'][:, selected], float)
        anchor = i4[:, selected, None]
        sm = fm/scale[:, :, None]
        g = np.stack(
            [(anchor*(1-fl[:, :, h::2])).mean(2) for h in range(2)] +
            [((1-anchor)*fl[:, :, h::2]).mean(2) for h in range(2)] +
            [(sm[:, :, h::2]/(.1+abs(sm[:, :, h::2]))).mean(2)
             for h in range(2)] +
            [(sm[:, :, h::2]-a[:, selected, None]).mean(2)
             for h in range(2)], -1)
        out.append(g.mean(1))
        for j in range(k):
            take = sl[selected] == j
            if take.any():
                out.append(g[:, take].mean(1))
    return np.concatenate(out, 1), selected


def ridge_grid(x, y, tr, va, alphas):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a, b = (x[tr]-mean)/std, (x[va]-mean)/std
    ym = y[tr].mean()
    if a.shape[1] <= len(tr):
        ev, v = eigh(a.T@a, check_finite=False)
        q = v.T@(a.T@(y[tr]-ym))
        left = b@v
    else:
        ev, v = eigh(a@a.T, check_finite=False)
        q = v.T@(y[tr]-ym)
        left = (b@a.T)@v
    return ym+(
        (left*q[None, :])[:, :, None] /
        (np.maximum(ev, 0)[:, None] +
         np.asarray(alphas)[None, :])[None]
    ).sum(1)


def fit_ridge(x, y, tr, alpha):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a = (x[tr]-mean)/std
    ym = y[tr].mean()
    coef = np.linalg.solve(
        a.T@a+alpha*np.eye(x.shape[1]), a.T@(y[tr]-ym))/std
    return coef, float(ym-mean@coef)


def rbf_grid(x, y, tr, va, alphas, gamma):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a, b = (x[tr]-mean)/std, (x[va]-mean)/std
    g = gamma/x.shape[1]
    kt = np.exp(-g*cdist(a, a, 'sqeuclidean'))
    kv = np.exp(-g*cdist(b, a, 'sqeuclidean'))
    ev, v = eigh(kt, check_finite=False)
    q = v.T@(y[tr]-y[tr].mean())
    return y[tr].mean()+((kv@v)*q) @ (
        1/(np.maximum(ev, 0)[:, None]+np.asarray(alphas)))


def fit_rbf(x, y, tr, alpha, gamma):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a = (x[tr]-mean)/std
    g = gamma/x.shape[1]
    kernel = np.exp(-g*cdist(a, a, 'sqeuclidean'))
    coef = np.linalg.solve(
        kernel+alpha*np.eye(len(tr)), y[tr]-y[tr].mean())
    return dict(mean=mean, std=std, centres=a, coef=coef,
                gamma=g, intercept=y[tr].mean())


def predict_rbf(model, x):
    z = (x-model['mean'])/model['std']
    return (
        np.exp(-model['gamma'] *
               cdist(z, model['centres'], 'sqeuclidean'))
        @ model['coef'] + model['intercept'])


def search(x, y, tr, va):
    options = [('ridge', 0.)]+[('rbf', g) for g in GAMMAS]
    values = []
    for family, gamma in options:
        values.append(
            ridge_grid(x, y, tr, va, ALPHAS) if family == 'ridge'
            else rbf_grid(x, y, tr, va, ALPHAS, gamma))
    return np.concatenate(values, 1), [
        (family, g, float(a))
        for family, g in options for a in ALPHAS]


def one(x, y, tr, va, setting):
    family, gamma, alpha = setting
    return (
        ridge_grid(x, y, tr, va, [alpha]) if family == 'ridge'
        else rbf_grid(x, y, tr, va, [alpha], gamma))[:, 0]


def nested(x, y):
    pop = np.arange(1024)
    pred = np.zeros(1030)
    settings = []
    for fold in range(8):
        tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
        loss = None
        for f in range(4):
            it, iv = tr[(tr//8) % 4 != f], tr[(tr//8) % 4 == f]
            p, opt = search(x, y, it, iv)
            e = ((p-y[iv, None])**2).sum(0)
            loss = e if loss is None else loss+e
        setting = opt[int(loss.argmin())]
        settings.append(setting)
        pred[va] = one(x, y, tr, va, setting)
    loss = None
    for f in range(8):
        tr, va = pop[pop % 8 != f], pop[pop % 8 == f]
        p, opt = search(x, y, tr, va)
        e = ((p-y[va, None])**2).sum(0)
        loss = e if loss is None else loss+e
    final = opt[int(loss.argmin())]
    pred[1024:] = one(x, y, pop, np.arange(1024, 1030), final)
    if final[0] == 'ridge':
        coef, intercept = fit_ridge(x, y, pop, final[2])
        model = dict(coef=coef, intercept=intercept)
    else:
        model = fit_rbf(x, y, pop, final[2], final[1])
    return pred, dict(outer=settings, final=final), model


def fit_readout(x, y, train, kernel='ridge', alpha=10., gamma=.1):
    if kernel == 'rbf':
        return fit_rbf(x, y, train, alpha, gamma)
    coef, intercept = fit_ridge(x, y, train, alpha)
    return dict(coef=coef, intercept=intercept)


def fast_rbf(model, x):
    z = (np.atleast_2d(x)-model['mean'])/model['std']
    a = model['centres']
    if 'centre_norm2' not in model:
        model['centre_norm2'] = np.sum(a*a, axis=1)
    norm = model['centre_norm2']
    distance = np.maximum(
        np.sum(z*z, axis=1)[:, None]+norm[None]-2*z@a.T, 0)
    return model['intercept']+np.exp(
        -model['gamma']*distance)@model['coef']


def readout(model, x, fast=False):
    if 'centres' in model:
        return (fast_rbf(model, x) if fast
                else predict_rbf(model, np.atleast_2d(x)))
    return np.atleast_2d(x)@model['coef']+model['intercept']


# ---------- Strict nested group selection and offline linear compilation ----------

def group_labels(arr, k, family='global'):
    arr = np.repeat(arr, k, axis=0)
    c = len(arr)
    groups = [np.zeros(c, int), np.tile(np.arange(k), 720)]
    if family in ('positions', 'pairs', 'cells'):
        order = np.argsort(arr, axis=1)
        groups += [arr[:, j] for j in range(3)]
        groups += [order[:, 0]*9+order[:, 1],
                   arr[:, 0]-arr[:, 1]+9]
    if family == 'pairs':
        groups += [arr[:, 0]*10+arr[:, 1],
                   arr[:, 0]*10+arr[:, 2],
                   arr[:, 1]*10+arr[:, 2]]
    if family == 'cells':
        groups += [np.arange(c)]
    return groups


def grouped_parts(d, w):
    m4, m3 = np.asarray(d['m4'], float), np.asarray(d['m3'], float)
    scale = np.maximum(
        np.sqrt((m4*m4).mean(1, keepdims=True)), 1e-12)
    a, b = m4/scale, m3/scale
    f = np.stack([
        d['I4'], d['I3'], a, b, abs(a), abs(b),
        a/(.1+abs(a)), b/(.1+abs(b)),
        a/(.03+abs(a)), b/(.03+abs(b))
    ], -1)
    return np.c_[w, scale, np.log(scale), m4.mean(1), m3.mean(1)], f


def grouped_features(d, w, family='global'):
    context, f = grouped_parts(d, w)
    arr = d.get('arr', np.array(
        list(itertools.permutations(range(10), 3))))
    out = [context]
    for group in group_labels(arr, int(d['K']), family):
        out.append(np.concatenate(
            [f[:, group == j].mean(1) for j in np.unique(group)], 1))
    return np.concatenate(out, 1)


def compile_groups(coef, arr, k, family='global'):
    """Fold all group means into one fixed cell coefficient table."""
    table = np.zeros((len(arr)*k, 10))
    pos = 24
    for group in group_labels(arr, k, family):
        for label in np.unique(group):
            ix = group == label
            table[ix] += coef[pos:pos+10]/ix.sum()
            pos += 10
    assert pos == len(coef)
    return table, np.asarray(coef[:24])


def compiled_predict(model, d, w):
    context, f = grouped_parts(d, w)
    return (
        model['intercept']+context@model['context_coef']+
        f.reshape(len(w), -1)@model['table'].ravel())


def nested_compiled(d, w, y):
    """Eight outer folds; four inner folds select all families and alpha."""
    pop = np.arange(1024)
    families = ['global', 'positions', 'pairs', 'cells']
    alphas = np.logspace(-1, 5, 13)
    losses = np.empty((8, 4, 13))
    grids = np.empty((4, 1024, 13))
    dims = {}
    for fi, family in enumerate(families):
        x = grouped_features(d, w, family)
        dims[family] = x.shape[1]
        for fold in range(8):
            tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
            loss = np.zeros(13)
            for inner in range(4):
                it = tr[(tr//8) % 4 != inner]
                iv = tr[(tr//8) % 4 == inner]
                pp = ridge_grid(x, y, it, iv, alphas)
                loss += np.sum((pp-y[iv, None])**2, axis=0)
            losses[fold, fi] = loss
            grids[fi, va] = ridge_grid(x, y, tr, va, alphas)
        print(f'Finished nested linear family {family}', flush=True)

    pred = np.empty(len(y))
    choices = []
    for fold in range(8):
        fi, ai = np.unravel_index(np.argmin(losses[fold]), (4, 13))
        va = pop[pop % 8 == fold]
        pred[va] = grids[fi, va, ai]
        choices.append(dict(
            fold=fold, family=families[fi], alpha=float(alphas[ai])))

    errors = ((grids-y[None, :1024, None])**2).mean(1)
    fi, ai = np.unravel_index(np.argmin(errors), (4, 13))
    family, alpha = families[fi], float(alphas[ai])
    x = grouped_features(d, w, family)
    pred[1024:] = ridge_grid(
        x, y, pop, np.arange(1024, len(y)), [alpha])[:, 0]
    if x.shape[1] <= 1024:
        coef, intercept = fit_ridge(x, y, pop, alpha)
    else:
        # Dual ridge avoids a huge feature Gram matrix.
        mean = x[pop].mean(0)
        sd = np.maximum(x[pop].std(0), .03)
        a = (x[pop]-mean)/sd
        ym = y[pop].mean()
        coef = (
            a.T@np.linalg.solve(
                a@a.T+alpha*np.eye(1024), y[pop]-ym))/sd
        intercept = float(ym-mean@coef)

    arr = d.get('arr', np.array(
        list(itertools.permutations(range(10), 3))))
    table, context = compile_groups(coef, arr, int(d['K']), family)
    model = dict(
        kind='compiled', k=int(d['K']), family=family, alpha=alpha,
        table=table, context_coef=context, intercept=intercept)
    return pred, dict(
        outer=choices, final_family=family, final_alpha=alpha,
        feature_dimensions=dims), model


# ---------- Fixed population-learned probe subsets and screening rows ----------

def six_features(I4, I3, m4, m3, scale):
    a, b = m4/scale, m3/scale
    return np.stack([
        I4, I3, a, b, a/(.1+abs(a)), b/(.1+abs(b))], -1)


def select_groups(f, base, y, train, n=8):
    A, G = f.shape[1:]
    v = f[train].reshape(len(train), -1)
    v = (v-v.mean(0))/np.maximum(v.std(0), .03)
    wx = base[train]-base[train].mean(0)
    wx /= np.maximum(wx.std(0), .03)
    u, s, _ = np.linalg.svd(wx, full_matrices=False)
    q = u[:, s > s.max()*1e-10]
    r = y[train]-y[train].mean()
    r -= q@(q.T@r)
    v -= q@(q.T@v)
    gram = np.einsum(
        'ncj,nck->cjk', v.reshape(-1, A, G),
        v.reshape(-1, A, G), optimize=True)
    corr = (v.T@r).reshape(A, G)
    cells = []
    for step in range(n):
        score = (
            corr*np.linalg.solve(
                gram+np.eye(G), corr[..., None])[..., 0]).sum(1)
        score[cells] = -np.inf
        j = int(score.argmax())
        cells.append(j)
        z = v[:, G*j:G*j+G]
        ev, vec = np.linalg.eigh(z.T@z)
        keep = ev > max(ev[-1]*1e-10, 1e-9)
        if not keep.any():
            break
        q = z@vec[:, keep]/np.sqrt(ev[keep])
        b, u = q.T@v, q.T@r
        r -= q@u
        v -= q@b
        bg = b.reshape(len(u), A, G)
        gram -= np.einsum('gcj,gck->cjk', bg, bg, optimize=True)
        corr -= (b.T@u).reshape(A, G)
    return np.array(cells)


def legacy_global(d, w):
    """Exact K1 global feature order for the learned-screen experiment."""
    m4, m3 = np.asarray(d['m4'], float), np.asarray(d['m3'], float)
    scale = np.maximum(
        np.sqrt((m4*m4).mean(1, keepdims=True)), 1e-12)
    a, b = m4/scale, m3/scale
    f = np.stack([
        d['I4'], d['I3'], a, b, abs(a), abs(b),
        a/(.1+abs(a)), b/(.1+abs(b)),
        a/(.03+abs(a)), b/(.03+abs(b))
    ], -1)
    return np.c_[
        w, scale, np.log(scale), m4.mean(1), m3.mean(1),
        f.mean(1), f.mean(1)]


def learn_selected(d, w, y, train, count=16, screens=False,
                   kernel='ridge', alpha=10., gamma=.1):
    if screens:
        base = legacy_global(d, w)
        scale = np.maximum(
            np.sqrt(np.mean(np.asarray(d['m4'], float)**2, 1)),
            1e-12)
        fm = np.asarray(d['fm'], float).reshape(len(w), -1)/scale[:, None]
        f = np.stack([
            d['fl'].reshape(len(w), -1),
            fm, fm/(.1+abs(fm))], -1)
    else:
        base = w
        scale = np.maximum(
            np.sqrt(np.mean(np.asarray(d['m4'][train], float)**2, 0)),
            .05)
        f = six_features(
            d['I4'], d['I3'], np.asarray(d['m4'], float),
            np.asarray(d['m3'], float), scale)
    chosen = select_groups(f, base, y, train, count)
    x = np.c_[base, f[:, chosen].reshape(len(w), -1)]
    model = fit_readout(x, y, train, kernel, alpha, gamma)
    if screens:
        model.update(selected=chosen)
    else:
        model.update(cells=chosen, scale=scale[chosen])
    model['kind'] = 'screens' if screens else 'subset'
    return model, x


def selected_cv(d, w, y, count=16, screens=False,
                kernel='ridge', alpha=10., gamma=.1):
    pop = np.arange(1024)
    pred, costs = np.empty(1030), np.empty(1030)
    for fold in range(8):
        tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
        model, x = learn_selected(
            d, w, y, tr, count, screens, kernel, alpha, gamma)
        pred[va] = np.clip(readout(model, x[va], fast=True), 0, 1)
        costs[va] = selected_cost(model)['total_with_readout']
    model, x = learn_selected(
        d, w, y, pop, count, screens, kernel, alpha, gamma)
    pred[1024:] = np.clip(readout(model, x[1024:], fast=True), 0, 1)
    costs[1024:] = selected_cost(model)['total_with_readout']
    return pred, costs, model


# ---------- Exact integration of the final Gaussian input ----------

def final_integral(h, wih, whh, wo, second, maximum, ps, pm):
    """P(correct | first nine inputs), using at most 18 breakpoints.

    h is the exact state after those inputs. ps/pm give their second/max
    positions, and second/maximum their values. Persistent ties use the
    lowest index. The final Gaussian input is integrated without samples.
    """
    h, wih, whh, wo = [
        np.asarray(v, np.float64) for v in (h, wih, whh, wo)]
    n = len(h)
    b = h@whh.T
    gate = np.divide(
        -b, wih, out=np.full_like(b, np.inf), where=wih != 0)
    events = np.column_stack([gate, second, maximum])
    order = events.argsort(-1, kind='stable')
    edges = np.column_stack([
        np.full(n, -np.inf),
        np.take_along_axis(events, order, 1),
        np.full(n, np.inf)])
    active = (
        (wih < 0)[None, :] |
        ((wih == 0)[None, :] & (b > 0)))
    slope = np.broadcast_to(
        np.minimum(wih, 0)@wo.T, (n, 10)).copy()
    bias = (active*b)@wo.T
    delta = (wo*np.abs(wih)).T
    ids = np.arange(n)
    target = np.asarray(ps).copy()
    start = np.full(n, np.nan)
    end = np.full(n, np.nan)
    answer = np.zeros(n)
    segments = np.zeros(n, int)
    for step in range(19):
        a = slope[ids, target, None]-slope
        c = bias[ids, target, None]-bias
        root = np.divide(-c, a, out=np.zeros_like(c), where=a != 0)
        lo = np.maximum(
            edges[:, step], np.where(a > 0, root, -np.inf).max(1))
        hi = np.minimum(
            edges[:, step+1], np.where(a < 0, root, np.inf).min(1))
        tied_loss = (
            (c == 0) &
            (np.arange(10)[None, :] < target[:, None]))
        impossible = (
            (a == 0) & ((c < 0) | tied_loss)).any(1)
        valid = (lo < hi) & ~impossible
        old = ~np.isnan(start)
        disjoint = valid & old & (lo > end)
        answer[disjoint] += (
            ndtr(end[disjoint])-ndtr(start[disjoint]))
        segments[disjoint] += 1
        reset = valid & (~old | disjoint)
        start[reset], end[reset] = lo[reset], hi[reset]
        join = valid & old & ~disjoint
        end[join] = np.maximum(end[join], hi[join])
        if step == 18:
            break
        event = order[:, step]
        target[event == 16] = 9
        target[event == 17] = np.asarray(pm)[event == 17]
        use = event < 16
        ix, g = ids[use], event[use]
        sign = np.sign(wih[g])
        slope[ix] += delta[g]
        bias[ix] += (sign*b[ix, g])[:, None]*wo[:, g].T

    use = ~np.isnan(start)
    answer[use] += ndtr(end[use])-ndtr(start[use])
    segments[use] += 1
    return answer, segments


def beta_rule(q, a, b):
    x, w = roots_jacobi(q, b-1, a-1)
    return (x+1)/2, w/w.sum()


def prefix_rule(q=1):
    """First-nine MF3 anchor with tensor Beta quadrature."""
    arr = np.array(list(itertools.permutations(range(9), 3)))
    rules = [
        beta_rule(q, 7, 3),
        beta_rule(q, 2, 1),
        beta_rule(q, 1, 1)]
    rows, weights = [], []
    for i, j, k in itertools.product(range(q), repeat=3):
        ur, vm, vs = rules[0][0][i], rules[1][0][j], rules[2][0][k]
        um = ur+(1-ur)*vm
        r, m, s = ndtri([ur, um, ur+(um-ur)*vs])
        mu = -np.exp(-r*r/2)/np.sqrt(2*np.pi)/ur
        x = np.full((504, 9), mu)
        x[np.arange(504)[:, None], arr] = np.array([m, s, r])
        rows.append(x)
        weights.extend([
            rules[0][1][i]*rules[1][1][j]*rules[2][1][k]/504
        ]*504)
    return np.concatenate(rows), np.array(weights)


def last_estimate(x, weight, wih, whh, wo, trie=None):
    if trie is None:
        trie = PrefixEvaluator(x)
    inp = trie.values[:, None]*wih
    _, v = trie.levels[0]
    h = np.maximum(inp[v], 0)
    for p, v in trie.levels[1:]:
        h = np.maximum((h@whh.T)[p]+inp[v], 0)
    h = h[trie.leaf]
    ranks = x.argsort(-1)
    ps, pm = ranks[:, -2], ranks[:, -1]
    second = x[np.arange(len(x)), ps]
    maximum = x[np.arange(len(x)), pm]
    probability, segments = final_integral(
        h, wih, whh, wo, second, maximum, ps, pm)
    n = len(x)
    prefix_mac = 256*sum(trie.counts[:-1])+16*len(trie.values)
    prefix_scalar = 16*(2*sum(trie.counts)-trie.counts[0])
    # Explicit allowance: normal CDF costs 100 scalar/MAC units each.
    sweep_ops = 6000*n+203*int(segments.sum())+336
    cost = (prefix_mac+prefix_scalar+sweep_ops+n)/2880
    return float(weight@probability), cost, float(segments.mean())


@lru_cache(maxsize=8)
def last_plan(q):
    x, weights = prefix_rule(q)
    return x, weights, PrefixEvaluator(x)


def last_features(w, q=1):
    x, weights, trie = last_plan(q)
    return last_estimate(
        x, weights, *[np.asarray(v, float) for v in w], trie)


# ---------- Target deployment and explicit operation allowances ----------

@lru_cache(maxsize=32)
def cell_plan(k=1, boxes=1, screen='none'):
    x, target, c = design(k, boxes, screen != 'none')
    r = 2 if screen == 'none' else 14
    mask = np.zeros((c, r), bool)
    mask[:, :2] = True
    if screen != 'none':
        ids = np.arange(c)
        arr = np.array(list(itertools.permutations(range(10), 3)))
        arr = arr[(ids//k) % 720]
        take = (
            arr[:, 0] == 9 if screen == 'late'
            else ids % int(screen[6:]) == 0
            if screen.startswith('stride')
            else np.ones(c, bool))
        mask[take, 2:] = True
    return (
        PrefixEvaluator(x.reshape(c, r, 10)[mask]),
        target.reshape(c, r)[mask], mask)


def target_cells(w, k=1, boxes=1, screen='none'):
    tree, target, mask = cell_plan(k, boxes, screen)
    ok, mg = bits_margins(tree(*w, dtype=np.float32), target)
    bits = np.zeros(mask.shape, np.uint8)
    margins = np.zeros(mask.shape, np.float32)
    bits[mask], margins[mask] = ok, mg
    d = dict(
        NB=boxes, K=k,
        I4=bits[None, :, 0], I3=bits[None, :, 1],
        m4=margins[None, :, 0], m3=margins[None, :, 1])
    if screen != 'none':
        d.update(fl=bits[None, :, 2:], fm=margins[None, :, 2:])
    return d, tree.cost()


@lru_cache(maxsize=32)
def selected_plan(indices, screens=False):
    x, target, _ = design(1, 1, screens)
    ids = np.asarray(indices)
    if screens:
        rows = np.r_[
            (14*np.arange(720)[:, None]+np.arange(2)).ravel(),
            14*(ids//12)+2+ids % 12]
    else:
        rows = (2*ids[:, None]+np.arange(2)).ravel()
    return PrefixEvaluator(x[rows]), target[rows]


def selected_online(model, w):
    screens = 'selected' in model
    ids = model['selected'] if screens else model['cells']
    tree, target = selected_plan(tuple(map(int, ids)), screens)
    ok, mg = bits_margins(tree(*w, dtype=np.float32), target)
    weights = raw20(*w)
    if screens:
        bits = ok[:1440].reshape(720, 2)
        margins = mg[:1440].reshape(720, 2)
        d = dict(
            I4=bits[None, :, 0], I3=bits[None, :, 1],
            m4=margins[None, :, 0], m3=margins[None, :, 1])
        base = legacy_global(d, weights)
        scale = max(
            np.sqrt(np.mean(margins[:, 0].astype(float)**2)),
            1e-12)
        sm = mg[1440:].astype(float)/scale
        f = np.stack(
            [ok[1440:], sm, sm/(.1+abs(sm))], -1)
    else:
        bits, margins = ok.reshape(-1, 2), mg.reshape(-1, 2)
        base = weights
        f = six_features(
            bits[:, 0], bits[:, 1],
            margins[:, 0], margins[:, 1], model['scale'])
    return np.c_[base, f.reshape(1, -1)]


def selected_cost(model):
    screens = 'selected' in model
    ids = model['selected'] if screens else model['cells']
    tree, _ = selected_plan(tuple(map(int, ids)), screens)
    cost = tree.cost()
    ops = (
        40*720+5000+12*len(ids) if screens
        else 20*len(ids)+5000)
    if 'centres' in model:
        n, dim = model['centres'].shape
        ops += n*(dim+38)+4*dim+1
    cost['other_readout_equiv'] = ops/2880
    cost['total_with_readout'] = cost['total_forward_equiv']+ops/2880
    return cost


def predict_network(w, model, k=1, boxes=1, screen='none',
                    context='raw20', last_q=0):
    kind = str(model.get(
        'kind',
        'subset' if 'cells' in model else
        'screens' if 'selected' in model else
        'weight' if 'train' in model else
        'compiled' if 'table' in model else 'summary'))

    if kind in ('subset', 'screens'):
        value = readout(model, selected_online(model, w), fast=True)[0]
        return float(np.clip(value, 0, 1)), selected_cost(model)

    if kind == 'weight':
        value = float(weight_predict(model, *w)[0])
        return value, dict(
            logical_rows=0,
            total_with_readout=(2681+58*len(model['train']))/2880)

    if kind == 'compiled':
        d, cost = target_cells(w, int(model.get('k', k)))
        extra = (40*len(d['m4'][0])+5000)/2880
        cost.update(
            other_readout_equiv=extra,
            total_with_readout=cost['total_forward_equiv']+extra)
        return float(compiled_predict(model, d, raw20(*w))[0]), cost

    k = int(model.get('k', k))
    boxes = int(model.get('boxes', boxes))
    screen = str(model.get('screen', screen))
    context = str(model.get('context', context))
    d, cost = target_cells(w, k, boxes, screen)
    f, selected = summarize(d, screen)
    name = 'readoutgauge' if context == 'function_invariant' else context
    feat = np.c_[
        feature_set(*w, name),
        f[:, 4:] if context == 'function_invariant' else f]
    last_q = int(model.get('last_q', last_q))
    if last_q:
        probability, c, segments = last_features(w, last_q)
        feat = np.c_[feat, probability, segments]
        cost['last_input_forward_equiv'] = c

    extra = (
        (40+30*(boxes > 1))*len(d['m4'][0])+
        240*int(selected.sum())+5000+FEATURE_COST[name]-2600)
    if 'centres' in model:
        n, dim = model['centres'].shape
        ops = 2*n*dim+34*n+2*dim+1
    else:
        ops = feat.shape[1]+1
    cost.update(
        feature_forward_equiv=extra/2880,
        learned_readout_forward_equiv=(ops+2)/2880)
    cost['total_with_readout'] = (
        cost['total_forward_equiv']+(extra+ops+2)/2880+
        cost.get('last_input_forward_equiv', 0))
    return float(np.clip(readout(model, feat)[0], 0, 1)), cost


# ---------- Command-line fitting, regeneration, deployment, and scoring ----------

def save_model(path, model):
    np.savez(path, **model)


def score(pred, truth, indices, costs=None):
    e = pred[indices]-truth[indices]
    result = dict(
        rms_pp=float(100*np.sqrt(np.mean(e*e))),
        mean_pp=float(100*e.mean()))
    if costs is not None:
        c = np.broadcast_to(costs, truth.shape)[indices]
        sampling = np.mean(truth[indices]*(1-truth[indices])/c)
        result.update(
            mean_cost=float(c.mean()),
            sampling_rms_pp=float(100*np.sqrt(sampling)),
            mse_ratio=float(np.mean(e*e)/sampling))
    return result


def generate_cache(w, k, boxes, screen, summary_only=False):
    rows = []
    for n in range(len(w[0])):
        d, cost = target_cells([v[n] for v in w], k, boxes, screen)
        rows.append(summarize(d, screen)[0] if summary_only else d)
        if n % 128 == 127:
            print(f'Computed {n+1}/{len(w[0])} networks', flush=True)
    if summary_only:
        return dict(
            NB=boxes, K=k, summary=np.concatenate(rows)), cost
    keys = [key for key in rows[0] if key not in ('NB', 'K')]
    return dict(
        NB=boxes, K=k,
        **{key: np.concatenate([d[key] for d in rows])
           for key in keys}), cost


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=[
        'predict', 'cache', 'summary', 'compiled',
        'subset', 'screens', 'weight', 'last'])
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--k', type=int, default=1)
    p.add_argument('--boxes', type=int, default=1)
    p.add_argument('--screen', default='none')
    p.add_argument('--context', default='raw20')
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--kernel', choices=['ridge', 'rbf'], default='ridge')
    p.add_argument('--alpha', type=float)
    p.add_argument('--gamma', type=float)
    p.add_argument('--cv', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    p.add_argument('--q', type=int, default=1)
    p.add_argument('--last', type=Path)
    p.add_argument('--last-q', type=int, default=0)
    a = p.parse_args()

    if a.command == 'predict':
        value, cost = predict_network(
            load_weights(a.source), dict(np.load(a.output)),
            a.k, a.boxes, a.screen, a.context, a.last_q)
        print(json.dumps(
            dict(estimated_accuracy=value, cost=cost), indent=2))
        return

    a.output.parent.mkdir(parents=True, exist_ok=True)
    w = load_models(a.source)
    y = np.load(a.source/'pop1k/reference.npz')['acc']
    pop, zoo = np.arange(1024), np.arange(1024, 1030)

    if a.command == 'cache':
        data, cost = generate_cache(
            w, a.k, a.boxes, a.screen, a.summary_only)
        np.savez(a.output, **data)
        print(json.dumps(cost, indent=2))
        return

    if a.command == 'last':
        values = np.array([
            last_features([v[n] for v in w], a.q)
            for n in range(1030)])
        np.savez(
            a.output, q=a.q, pred=values[:, 0],
            cost=values[:, 1], segments=values[:, 2])
        return

    costs = None
    if a.command == 'weight':
        alpha, gamma = a.alpha or .01, a.gamma or .01
        pred = np.empty(1030)
        if a.cv:
            for fold in range(8):
                tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
                m = weight_fit(
                    *(v[tr] for v in w), y[tr], alpha, gamma)
                pred[va] = weight_predict(
                    m, *(v[va] for v in w))
        model = weight_fit(
            *(v[pop] for v in w), y[pop], alpha, gamma)
        pred[zoo] = weight_predict(
            model, *(v[zoo] for v in w))
        if not a.cv:
            pred[pop] = weight_predict(
                model, *(v[pop] for v in w))
        model['kind'] = 'weight'
        costs = np.r_[
            np.full(
                1024, (2681+58*(896 if a.cv else 1024))/2880),
            np.full(6, (2681+58*1024)/2880)]
    else:
        cache = a.cache or (
            a.source /
            f'pop1k/seed0_z46_cells_b{a.boxes}_k{a.k}.npz')
        d = dict(np.load(cache))
        if a.command == 'compiled':
            if int(d['NB']) != 1:
                raise ValueError(
                    'The four-family compiled design requires one outer box')
            pred, settings, model = nested_compiled(d, raw20(*w), y)
            costs = predict_network(
                [v[0] for v in w], model)[1]['total_with_readout']
            a.cv = True
        elif a.command in ('subset', 'screens'):
            if int(d['K']) != 1 or int(d['NB']) != 1:
                raise ValueError(
                    'Learned subsets/screens require the K1 one-box design')
            base = raw20(*w)
            alpha = a.alpha or (
                10. if a.kernel == 'ridge' else .01)
            gamma = a.gamma or .1
            if a.cv:
                pred, costs, model = selected_cv(
                    d, base, y, a.count, a.command == 'screens',
                    a.kernel, alpha, gamma)
            else:
                model, x = learn_selected(
                    d, base, y, pop, a.count, a.command == 'screens',
                    a.kernel, alpha, gamma)
                pred = np.clip(readout(model, x, fast=True), 0, 1)
                costs = selected_cost(model)['total_with_readout']
        else:
            f = (
                d['summary'] if 'summary' in d
                else summarize(d, a.screen)[0])
            name = (
                'readoutgauge'
                if a.context == 'function_invariant' else a.context)
            x = np.c_[
                feature_set(*w, name),
                f[:, 4:] if a.context == 'function_invariant' else f]
            if a.last:
                last = np.load(a.last)
                x = np.c_[x, last['pred'], last['segments']]
            if a.cv:
                pred, settings, model = nested(x, y)
            else:
                tr = pop[pop % 8 != 0]
                dv = pop[pop % 16 == 0]
                grid, choices = search(x, y, tr, dv)
                setting = choices[int(
                    np.mean((grid-y[dv, None])**2, 0).argmin())]
                model = fit_readout(
                    x, y, tr, setting[0], setting[2], setting[1])
                pred, settings = readout(model, x), setting
            model.update(
                kind='summary', k=int(d['K']), boxes=int(d['NB']),
                screen=a.screen, context=a.context)
            if a.last:
                model['last_q'] = (
                    int(last['q']) if 'q' in last else a.q)

    save_model(a.output, model)
    np.save(
        a.output.with_name(a.output.stem+'_pred.npy'), pred)
    report = dict(
        population=score(pred, y, pop, costs),
        zoo=score(pred, y, zoo, costs),
        zoo_errors_pp=(100*(pred[zoo]-y[zoo])).tolist(),
        cross_fitted=a.cv)
    if a.command in ('summary', 'compiled'):
        report['selection'] = settings
    if costs is not None:
        np.save(
            a.output.with_name(a.output.stem+'_cost.npy'),
            np.broadcast_to(costs, y.shape))
    a.output.with_suffix('.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
