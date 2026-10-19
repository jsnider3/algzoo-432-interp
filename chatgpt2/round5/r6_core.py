"""CPU deterministic MF3 quadrature with shared prefixes and linear-logit roots.
Run from this repository. All rules are fixed; reference values only score them.
"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, roots_jacobi
from threadpoolctl import threadpool_limits

ARR = np.array(list(itertools.permutations(range(10), 3)))

def load(ids):
    z = np.load('pop1k/seed0.npz')
    keys = ['Wih', 'Whh', 'Wo']
    pop = [z['s1048576_' + k] for k in keys]
    zoo = []
    for f in ['weights.npz'] + [f'weights_seed{i}.npz' for i in range(5)]:
        w = np.load(f)
        zoo.append([w['rnn.weight_ih_l0'][:, 0],
                    w['rnn.weight_hh_l0'], w['linear.weight']])
    return [np.ascontiguousarray(np.stack([
        pop[j][i] if i < 1024 else zoo[i-1024][j] for i in ids
    ]), dtype=np.float64) for j in range(3)]

def rule(n, a=1, b=1):
    x, w = roots_jacobi(n, b-1, a-1)
    return (x+1)/2, w/w.sum()

def forward(W, X):
    wi, wh, wo = W
    h = np.zeros((len(wi), len(X), 16))
    for t in range(10):
        h = np.maximum(h @ wh.transpose(0, 2, 1)
                       + wi[:, None, :] * X[None, :, t, None], 0)
    return h @ wo.transpose(0, 2, 1)

def prefix_forward(W, tokens, values):
    """Evaluate exactly the same rows, deduplicating identical input prefixes.
    Returns logits and actual dense multiply/add work, in 2880-MAC units.
    Also charges scalar adds and ReLU comparisons, unlike the base MAC count.
    """
    wi, wh, wo = W
    drive = wi[:, None, :] * values[None, :, None]
    h = np.zeros((len(wi), 1, 16))
    inv = np.zeros(len(tokens), dtype=int)
    work = 16*len(values)
    for t in range(10):
        edges, inv = np.unique(np.column_stack([inv, tokens[:, t]]),
                               axis=0, return_inverse=True)
        if t == 0:
            h = np.maximum(drive[:, edges[:, 1]], 0)
            work += len(edges)*16
        else:
            h = np.maximum(h[:, edges[:, 0]] @ wh.transpose(0, 2, 1)
                           + drive[:, edges[:, 1]], 0)
            work += len(edges)*(256+32)
    L = h @ wo.transpose(0, 2, 1)
    work += len(h[0])*160
    return L[:, inv], work/2880

def line_probability(L, s, y):
    """Gaussian measure of correctness for logits linearly interpolated in x_S.
    The integral of this interpolant is exact; the network need not be affine.
    """
    target = np.take_along_axis(L, y[None, :, None, None], axis=-1)
    d = target-L
    d0, dd = d[:, :, :-1], np.diff(d, axis=2)
    nz = dd != 0
    r = np.divide(-d0, dd, out=np.zeros_like(dd), where=nz)
    low = np.maximum(0, np.max(np.where(dd > 0, r, -np.inf), axis=-1))
    high = np.minimum(1, np.min(np.where(dd < 0, r, np.inf), axis=-1))
    bad = np.any((~nz) & ((d0 < 0) | ((d0 == 0) &
        (np.arange(10)[None, None, None, :] < y[None, :, None, None]))), axis=-1)
    low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
    a = s[:-1]+np.diff(s)*low
    b = s[:-1]+np.diff(s)*high
    mass = np.where((high > low) & ~bad, ndtr(b)-ndtr(a), 0)
    return mass.sum(axis=-1)/(ndtr(s[-1])-ndtr(s[0]))

def mf3(W, n, K):
    """n x n Beta outer rule; K affine interpolation intervals along S."""
    est = np.zeros(len(W[0])); cost = 0.0
    ur, wr = rule(n, 8, 3); ug, wg = rule(n, 2, 1)
    for a, wa in zip(ur, wr):
        for g, w in zip(ug, wg):
            b = a+(1-a)*g
            R, M = ndtri(a), ndtri(b)
            s = ndtri(a+(b-a)*np.arange(K+1)/K)
            mu = -np.exp(-R*R/2)/np.sqrt(2*np.pi)/a
            # symbols 0=M, 1=R, 2=mu; interior S nodes have distinct symbols.
            vals = np.r_[M, R, mu, s[1:-1]]
            st = np.r_[1, np.arange(3, K+2), 0]
            tok = np.full((720, K+1, 10), 2, dtype=int)
            for j in range(720):
                tok[j, :, ARR[j, 0]] = 0
                tok[j, :, ARR[j, 1]] = st
                tok[j, :, ARR[j, 2]] = 1
            L, c = prefix_forward(W, tok.reshape(-1, 10), vals)
            p = line_probability(L.reshape(len(W[0]), 720, K+1, 10),
                                 s, ARR[:, 1])
            est += wa*w*p.mean(axis=1)
            # 200 scalar-equivalent operations/interval: margins, roots,
            # reductions, masks and both normal CDFs. Report separately.
            cost += c+720*K*200/2880
    return est, cost

def score(est, ids, cost):
    ref = np.load('pop1k/reference.npz')
    p = ref['acc'][ids]; err = (est-p)*100
    ans = {'cost': float(cost), 'ids': list(map(int, ids)),
           'estimates': est.tolist(), 'errors_pp': err.tolist()}
    for name, mask in [('dev', ids < 1024),
                       ('held', (ids < 1024) & (ids % 16 == 8))]:
        if name == 'dev': mask &= ids % 16 == 0
        if mask.any():
            e = err[mask]; samp = 100*np.sqrt(np.mean(p[mask]*(1-p[mask]))/cost)
            ans[name] = dict(rms=float(np.sqrt(np.mean(e*e))),
                             mean=float(e.mean()), sampling=float(samp),
                             ratio=float(np.mean(e*e)/samp**2))
    ans['zoo_errors'] = err[ids >= 1024].tolist()
    return ans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['dev', 'held', 'all'], default='dev')
    ap.add_argument('--settings', default='1:1,1:2,2:4,3:8,4:16,6:32')
    ap.add_argument('--out', default='r6_results/anchors_dev.json')
    a = ap.parse_args()
    ids = np.r_[np.arange(0, 1024, 16) if a.split == 'dev' else
                np.arange(8, 1024, 16) if a.split == 'held' else
                np.sort(np.r_[np.arange(0, 1024, 16), np.arange(8, 1024, 16)]),
                np.arange(1024, 1030)]
    W = load(ids); results = []
    for n, K in [tuple(map(int, s.split(':'))) for s in a.settings.split(',')]:
        start = time.time(); est, cost = mf3(W, n, K)
        row = score(est, ids, cost); row.update(n=n, K=K)
        results.append(row)
        Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(n, K, 'seconds', round(time.time()-start, 1),
              {k: v for k, v in row.items() if k not in
               ('ids', 'estimates', 'errors_pp')}, flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
