# Reimplementation of ChatGPT Pro's two estimators (its code was lost), from its description:
#  (1) "ordered Sobol": exact order-statistic transport of N(0,1)^10 driven by scrambled Sobol points
#  (2) "cmc": condition on x1..x9, integrate x10 exactly (logits are piecewise linear in x10)
# Both are unbiased per scramble; error bars come from independent scrambles.
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri, ndtr
from itertools import permutations
from seeds_clocks import load

def fwd(x, Wih, Whh):
    h = np.zeros((len(x), Whh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(x[:, t:t+1]*Wih.reshape(-1) + h@Whh.T, 0)
    return h

def transport(u, T, k=3):
    """u: (n, k+1+(T-k)) uniforms -> x (n,T), exact N(0,1)^T in distribution.
    columns: k spacing coords (top first), then arrangement, then T-k lower coords."""
    n = len(u)
    # log U_(1) = log(v0)/T ; log U_(j+1) = log U_(j) + log(v_j)/(T-j)
    logU = np.zeros((n, k)); acc = np.zeros(n)
    for j in range(k):
        acc = acc + np.log(u[:, j])/(T-j); logU[:, j] = acc
    top = -ndtri(-np.expm1(logU))            # Phi^{-1}(U) computed via upper tail for precision
    arr = np.array(list(permutations(range(T), k)))
    a = arr[np.minimum((u[:, k]*len(arr)).astype(int), len(arr)-1)]
    low = ndtri(np.exp(logU[:, -1:]) * u[:, k+1:])
    x = np.empty((n, T))
    rows = np.arange(n)[:, None]
    mask = np.ones((n, T), bool); mask[rows, a] = False
    x[rows, a] = top
    x[mask] = low.ravel()
    return x

def design(m, T, reps, seed, k=3):
    d = k+1+T-k
    return [qmc.Sobol(d, scramble=True, seed=seed+r).random_base2(m) for r in range(reps)]

def acc_plain(x, W):
    Wih, Whh, Wo = W
    return np.mean((fwd(x, Wih, Whh)@Wo.T).argmax(1) == np.argsort(x, 1)[:, -2])

def cond_prob(y, W):
    """P(correct | x1..x9 = y), exact: sweep the <=H+2 breakpoints in x10."""
    Wih, Whh, Wo = W
    n = len(y); H = Whh.shape[0]
    a = Wih.reshape(-1); b = fwd(y, Wih, Whh)@Whh.T          # pre-activation at step 10: a z + b
    with np.errstate(divide='ignore'):
        hinge = np.where(a != 0, -b/a, np.inf)
    o = np.argsort(y, 1); i1, i2 = o[:, -1], o[:, -2]
    m1 = y[np.arange(n), i1]; m2 = y[np.arange(n), i2]
    bp = np.sort(np.concatenate([hinge, m1[:, None], m2[:, None]], 1), 1)
    lo = np.concatenate([np.full((n, 1), -np.inf), bp], 1)
    hi = np.concatenate([bp, np.full((n, 1), np.inf)], 1)
    mid = np.where(np.isinf(lo), hi-1, np.where(np.isinf(hi), lo+1, (lo+hi)/2))
    tgt = np.where(mid < m2[:, None], i2[:, None], np.where(mid < m1[:, None], 9, i1[:, None]))
    act = (mid[:, :, None]*a + b[:, None, :]) > 0          # (n,C,H)
    s = (act*a) @ Wo.T; c = (act*b[:, None, :]) @ Wo.T     # logits = s z + c, (n,C,10)
    st = np.take_along_axis(s, tgt[..., None], 2); ct = np.take_along_axis(c, tgt[..., None], 2)
    ds = st - s; dc = ct - c                               # need ds z + dc > 0 for j != tgt
    with np.errstate(divide='ignore', invalid='ignore'):
        r = -dc/ds
    L = np.where(ds > 0, r, -np.inf).max(2); U = np.where(ds < 0, r, np.inf).min(2)
    bad = ((ds == 0) & (dc <= 0)); bad[np.arange(n)[:, None], np.arange(lo.shape[1]), tgt] = False
    L = np.maximum(L, lo); U = np.minimum(U, hi)
    p = np.where((U > L) & ~bad.any(2), ndtr(U) - ndtr(L), 0.0)
    return p.sum(1)

def perturb(W, sc, seed):   # i8.py convention
    rng = np.random.default_rng(seed); Wih, Whh, Wo = W
    return (Wih*(1+sc*rng.standard_normal(Wih.shape)), Whh+sc*np.abs(Whh).mean()*rng.standard_normal(Whh.shape),
            Wo*(1+sc*rng.standard_normal(Wo.shape)))

def batched(f, x, W, bs=1 << 16):
    return np.mean(np.concatenate([np.atleast_1d(f(x[i:i+bs], W)) for i in range(0, len(x), bs)]))

def cp_vec(y, W):
    return cond_prob(y, W)

def pl_vec(x, W):
    Wih, Whh, Wo = W
    return ((fwd(x, Wih, Whh)@Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float)
