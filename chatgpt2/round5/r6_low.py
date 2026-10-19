"""Small MF2 product-Beta rules, evaluated with actual shared-prefix work."""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits
from r6_core import load, rule, prefix_forward, score

def mf2(W, n, q):
    arr = np.array(list(itertools.permutations(range(10), 2)))
    um, wm = rule(n, 10, 1); us, ws = rule(q, 9, 1)
    est = np.zeros(len(W[0])); cost = 0.0
    for a, weight in zip(um, wm):
        M, S = ndtri(a), ndtri(a*us)
        mu = -np.exp(-S*S/2)/np.sqrt(2*np.pi)/(a*us)
        vals = np.r_[M, S, mu]
        tok = np.tile((1+q+np.arange(q))[None, :, None], (90, 1, 10))
        for j, (mp, sp) in enumerate(arr):
            tok[j, :, mp] = 0; tok[j, :, sp] = 1+np.arange(q)
        L, c = prefix_forward(W, tok.reshape(-1, 10), vals)
        I = L.reshape(len(W[0]), 90, q, 10).argmax(-1) == arr[None, :, None, 1]
        est += weight*(I*ws).sum(-1).mean(1)
        cost += c+12*90*q/2880
    return est, cost

if __name__ == '__main__':
    # The same six rules are fixed before evaluating either split.
    with threadpool_limits(limits=2):
        for split, offset in [('dev',0),('held',8)]:
            ids = np.r_[np.arange(offset,1024,16),np.arange(1024,1030)]
            W = load(ids); out = []
            for n,q in [(1,1),(1,2),(2,2),(2,4),(3,4),(4,8)]:
                est,cost = mf2(W,n,q); r = score(est,ids,cost)
                r.update(n=n,q=q); out.append(r)
                print(split,n,q,{k:v for k,v in r.items() if k not in ('ids','estimates','errors_pp')},flush=True)
            Path(f'r6_results/low_{split}.json').write_text(json.dumps(out,indent=2))
