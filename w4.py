# Explanation-based accuracy estimation across independently trained seeds.
# "Explanation" = error rates per mechanism-defined stratum (blind spot S<0; grid of normalized gaps
# a=(M-S)/|M|, b=(S-R)/|M|). Prework (network-independent, plus other seeds): stratum masses, input pools,
# population error rates. Per new network: n evaluations. Leave-one-seed-out; MSE vs true accuracy.
import numpy as np, sys, glob
from w1 import load, pl_vec
AB = [0, .05, .15, .4, np.inf]; BB = [0, .03, .1, .3, np.inf]
K = 1 + (len(AB)-1)*(len(BB)-1)
def strata(x):
    s = -np.sort(-x, 1); M, S, R = s[:, 0], s[:, 1], s[:, 2]
    a = (M-S)/np.abs(M); b = (S-R)/np.abs(M)
    h = 1 + (np.digitize(a, AB)-1)*(len(BB)-1) + (np.digitize(b, BB)-1)
    return np.where(S < 0, 0, h)
POOL = 1 << 15
rng = np.random.default_rng(11)
cnt = np.zeros(K); pool = [[] for _ in range(K)]; have = np.zeros(K, int); tot = 0
while have.min() < POOL:
    x = rng.standard_normal((1 << 22, 10)); h = strata(x); tot += len(x); cnt += np.bincount(h, minlength=K)
    for k in range(K):
        if have[k] < POOL:
            xs = x[h == k][:POOL-have[k]]; pool[k].append(xs); have[k] += len(xs)
w = cnt/tot; pool = [np.concatenate(p) for p in pool]
print("strata masses", w.round(4), "draws", tot, flush=True)
files = sys.argv[1:]
F = np.array([[pl_vec(pool[k], load(f)) for k in range(K)] for f in files])   # (nets, K, POOL) correctness
r = 1-F.mean(2)                       # true per-stratum error rates (pool)
truth = 1-r@w
print("accuracies", truth.round(4), flush=True)
N = len(files); REPS = 400
for n in [256, 1024, 4096, 16384]:
    res = {k: [] for k in ["iid", "strat", "strat+EB", "prior", "EB-only-small"]}
    for i in range(N):
        o = np.delete(np.arange(N), i); m = r[o].mean(0)
        v = np.maximum(r[o].var(0, ddof=1), 1e-12)
        alpha = np.clip(m*(1-m)/v-1, 0, 1e7)                       # beta-binomial prior strength
        sig = np.sqrt(np.maximum(m*(1-m), 1e-6)); nh = np.maximum(2, np.round(n*w*sig/(w*sig).sum())).astype(int)
        e_iid, e_st, e_eb = [], [], []
        for _ in range(REPS):
            ni = rng.multinomial(n, w)
            ki = np.array([F[i, k, rng.integers(0, POOL, ni[k])].sum() for k in range(K)])
            e_iid.append(ki.sum()/n)
            ks = np.array([n_ - F[i, k, rng.integers(0, POOL, n_)].sum() for k, n_ in enumerate(nh)])   # error counts
            e_st.append(1-w@(ks/nh))
            e_eb.append(1-w@((ks+alpha*m)/(nh+alpha)))
        for key, e in [("iid", e_iid), ("strat", e_st), ("strat+EB", e_eb)]:
            res[key].append(np.mean((np.array(e)-truth[i])**2))
        res["prior"].append((1-w@m-truth[i])**2)
    print(f"n={n:6d} RMSE (pp) over {N} held-out seeds: " + " | ".join(f"{k} {100*np.sqrt(np.mean(v)):.3f}" for k, v in res.items() if v)
          + f" | MSE gain strat {np.mean(res['iid'])/np.mean(res['strat']):.2f}x, strat+EB {np.mean(res['iid'])/np.mean(res['strat+EB']):.2f}x", flush=True)
