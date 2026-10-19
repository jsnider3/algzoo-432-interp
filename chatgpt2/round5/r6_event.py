"""Deterministic event-branch quadrature, optional age/value-bin mean merging.
The ninth-step mixture is integrated exactly over the final Gaussian input.
This is a point/mean closure, not an exact affine-component distribution law.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits
from r6_core import load, rule, score

def phi(x):
    return np.exp(-x*x/2)/np.sqrt(2*np.pi)

def last_input(W, h, M, S, ip, jp):
    """Exact gate sweep with lowest-index output ties. No random numbers."""
    probs = np.zeros(h.shape[:2]); B = len(M); rows = np.arange(B)
    for n, (wi, wh, wo) in enumerate(zip(*W)):
        a = h[n] @ wh.T
        gates = np.divide(-a, wi, out=np.full_like(a, np.inf), where=wi != 0)
        bounds = np.column_stack([gates, S, M])
        order = np.argsort(bounds, axis=1)
        bounds = np.take_along_axis(bounds, order, axis=1)
        on = (wi < 0) | ((wi == 0) & (a > 0))
        c = (a*on) @ wo.T
        d = (wi*on) @ wo.T
        low = np.full(B, -np.inf); total = np.zeros(B)
        for k in range(19):
            high = bounds[:, k] if k < 18 else np.full(B, np.inf)
            tgt = np.where(low >= M, ip, np.where(low >= S, 9, jp))
            dc = c[rows, tgt, None]-c
            dd = d[rows, tgt, None]-d
            roots = np.divide(-dc, dd, out=np.zeros_like(dd), where=dd != 0)
            l = np.maximum(low, np.max(np.where(dd > 0, roots, -np.inf), axis=1))
            u = np.minimum(high, np.min(np.where(dd < 0, roots, np.inf), axis=1))
            bad = np.any((dd == 0) & ((dc < 0) | ((dc == 0) &
                          (np.arange(10)[None, :] < tgt[:, None]))), axis=1)
            total += np.where((u > l) & ~bad, ndtr(u)-ndtr(l), 0)
            if k < 18:
                gate = order[:, k]; j = np.minimum(gate, 15)
                sign = np.where(gate < 16, np.sign(wi[j]), 0)
                c += (sign*a[rows, j])[:, None]*wo[:, j].T
                d += (sign*wi[j])[:, None]*wo[:, j].T
                flip = (gate < 16) & (wi[j] != 0)
                on[rows[flip], j[flip]] ^= True
                # Subtracting gate contributions can leave roundoff when the
                # exact hidden state is zero. Restore the exact all-zero tie.
                empty = ~on.any(axis=1)
                c[empty] = 0; d[empty] = 0
            low = high
        probs[n] = total
    # Includes final recurrent map, both initial readouts, gate sorting/updates,
    # and 200 scalar-equivalents per root/CDF interval; conservatively charged.
    cost = B*(256+320+16+128+18*(40+20)+19*200)/2880
    return probs, cost

def merge(h, M, S, ip, jp, weights, bins):
    if bins == 0:
        return h, M, S, ip, jp, weights, 0.0
    keys = np.column_stack([ip, jp, np.minimum(bins-1, (bins*ndtr(M)).astype(int)),
                             np.minimum(bins-1, (bins*ndtr(S)).astype(int))])
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    ix = np.argsort(inv, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(inv[ix]))+1]
    total = np.add.reduceat(weights[ix], starts)
    merged = np.add.reduceat(h[:, ix]*weights[None, ix, None], starts, axis=1)
    merged /= total[None, :, None]
    avg = lambda x: np.add.reduceat(x[ix]*weights[ix], starts)/total
    first = ix[starts]
    cost = (32*len(weights)+16*len(total))/2880
    return merged, avg(M), avg(S), ip[first], jp[first], total, cost

def event(W, q=1, bins=0):
    """q=1 uses exact truncated means; q>1 Gauss-Legendre in conditional CDF.
    bins=0 retains every component. Positive bins merge joint hidden vectors
    by the two positions and CDF bins of M,S, keeping the entire vector mean.
    """
    nodes, weights = rule(q)
    u = np.r_[nodes/2, .5+nodes/2]
    x = np.r_[-np.sqrt(2/np.pi), np.sqrt(2/np.pi)] if q == 1 else ndtri(u)
    mass = np.tile(weights/2, 2)
    wi, wh, wo = W
    h = np.maximum(wi[:, None, :]*x[None, :, None], 0)
    M = x.copy(); S = np.full(len(x), -np.inf)
    ip = np.zeros(len(x), dtype=int); jp = np.full(len(x), -1, dtype=int)
    cost = len(x)*32/2880; counts = [len(x)]
    for t in range(1, 9):
        bounds = np.column_stack([np.zeros(len(M)), ndtr(S), ndtr(M), np.ones(len(M))])
        widths = np.diff(bounds, axis=1)
        parent, branch = np.where(widths > 1e-15)
        left = bounds[parent, branch]; width = widths[parent, branch]
        if q == 1:
            z = (phi(ndtri(left))-phi(ndtri(left+width)))/width
            w = mass[parent]*width
        else:
            z = ndtri(left[:, None]+width[:, None]*nodes).ravel()
            w = (mass[parent, None]*width[:, None]*weights).ravel()
            parent = np.repeat(parent, q); branch = np.repeat(branch, q)
        h = np.maximum(h[:, parent] @ wh.transpose(0, 2, 1)
                       + wi[:, None, :]*z[None, :, None], 0)
        cost += len(z)*(272+32)/2880
        newmax = branch == 2; newsecond = branch == 1
        S = np.where(newmax, M[parent], np.where(newsecond, z, S[parent]))
        M = np.maximum(M[parent], z)
        jp = np.where(newmax, ip[parent], np.where(newsecond, t, jp[parent]))
        ip = np.where(newmax, t, ip[parent]); mass = w
        h, M, S, ip, jp, mass, c = merge(h, M, S, ip, jp, mass, bins)
        cost += c; counts.append(len(M))
    p, c = last_input(W, h, M, S, ip, jp)
    assert abs(mass.sum()-1) < 1e-10
    return p@mass, cost+c, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['dev', 'held'], default='dev')
    ap.add_argument('--settings', default='1:1,1:2,1:4,1:8,1:0,2:2,2:4,2:8')
    ap.add_argument('--out', default='r6_results/event_dev.json')
    a = ap.parse_args()
    ids = np.r_[np.arange(0 if a.split == 'dev' else 8, 1024, 16), np.arange(1024, 1030)]
    W = load(ids); out = []
    for q, bins in [tuple(map(int, s.split(':'))) for s in a.settings.split(',')]:
        start = time.time(); est, cost, counts = event(W, q, bins)
        r = score(est, ids, cost); r.update(q=q, bins=bins, counts=counts)
        out.append(r); Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(q, bins, 'seconds', round(time.time()-start, 1),
              {k:v for k,v in r.items() if k not in ('ids','estimates','errors_pp')}, flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
