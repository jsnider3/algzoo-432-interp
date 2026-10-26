# Sequential, most-informative-first accuracy estimation across independently trained seeds.
# Strata = mechanism failure modes (sign of S; normalized gaps a=(M-S)/|M|, b=(S-R)/|M|). Network-independent prework:
# stratum masses + input pools. Per held-out network: batches allocated by estimated Neyman weights
# w_h*sqrt(e_h(1-e_h)) (error rates shrunk toward other seeds), stop when estimated SE < target.
# Compare runs used vs brute force runs needed for the same SE, and check realized error.
import numpy as np, sys
from w1 import load, pl_vec
AB = [0, .02, .05, .1, .2, .4, np.inf]; BB = [0, .01, .03, .06, .1, .2, .4, np.inf]; SB = [-np.inf, -.5, 0]
NA, NB = len(AB)-1, len(BB)-1; K = (len(SB)-1) + NA*NB
def strata(x):
    s = -np.sort(-x, 1); M, S, R = s[:, 0], s[:, 1], s[:, 2]
    a = (M-S)/np.abs(M); b = (S-R)/np.abs(M)
    h = (len(SB)-1) + (np.digitize(a, AB)-1)*NB + (np.digitize(b, BB)-1)
    return np.where(S < 0, np.digitize(S, SB)-1, h)
POOL = 1 << 14
rng = np.random.default_rng(11)
cnt = np.zeros(K); pool = [[] for _ in range(K)]; have = np.zeros(K, int); tot = 0
while have.min() < POOL and tot < 2**28:
    x = rng.standard_normal((1 << 22, 10)); h = strata(x); tot += len(x); cnt += np.bincount(h, minlength=K)
    for k in np.flatnonzero(have < POOL):
        xs = x[h == k][:POOL-have[k]]; pool[k].append(xs); have[k] += len(xs)
w = cnt/tot; pool = [np.concatenate(p) for p in pool]; ps = np.array([len(p) for p in pool])
print(f"{K} strata, masses {w.min():.1e}..{w.max():.2f}, smallest pool {ps.min()}, draws {tot}", flush=True)
files = sys.argv[1:]; N = len(files)
F = [[pl_vec(pool[k], load(f)) for k in range(K)] for f in files]
r = np.array([[1-F[i][k].mean() for k in range(K)] for i in range(N)])
truth = 1-r@w
print("accuracies", np.sort(truth).round(3), flush=True)
B = 256; REPS = 100
for tgt in [0.005, 0.002, 0.001, 0.0005]:
    used, err, iidn, cov = [], [], [], []
    for i in range(N):
        o = np.delete(np.arange(N), i); m = np.clip(r[o].mean(0), 1e-4, 1-1e-4)
        v = np.maximum(r[o].var(0, ddof=1), 1e-10); alpha = np.clip(m*(1-m)/v-1, 1, 1e4)
        p = truth[i]; iidn.append(p*(1-p)/tgt**2)
        for _ in range(REPS):
            n = np.zeros(K); k_ = np.zeros(K)
            while True:
                e = (k_+alpha*m)/(n+alpha)                       # posterior-mean error rates
                sig = np.sqrt(e*(1-e)); want = w*sig/(w*sig).sum()*(n.sum()+B)
                add = np.maximum(want-n, 0); add = np.floor(add/add.sum()*B + rng.random(K)).astype(int) if add.sum() > 0 else rng.multinomial(B, w)
                for h in np.flatnonzero(add):
                    k_[h] += add[h]-F[i][h][rng.integers(0, ps[h], add[h])].sum(); n[h] += add[h]
                nn = np.maximum(n, 1); eh = k_/nn
                se = np.sqrt(np.sum(w**2*np.maximum(eh*(1-eh), e*(1-e))/nn))   # floor variance by prior to avoid 0/0 stops
                if se < tgt and n.sum() >= 4*B: break
            est = 1-w@np.where(n > 0, k_/nn, m)
            used.append(n.sum()); err.append(est-p); cov.append(abs(est-p) < 1.96*se)
    used = np.array(used); err = np.array(err)
    print(f"target SE {100*tgt:.2f}pp: runs used median {np.median(used):.0f} (vs brute force {np.median(iidn):.0f}) "
          f"-> {np.median(iidn)/np.median(used):.1f}x fewer | realized RMSE {100*np.sqrt(np.mean(err**2)):.3f}pp, "
          f"bias {100*err.mean():+.3f}pp, 95% coverage {np.mean(cov):.2f}", flush=True)
