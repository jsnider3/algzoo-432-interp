# Diagnose x2's blur formula: compare with the true conditional P(correct | M,S,R,positions) (MC over the lower inputs).
import numpy as np
from scipy.special import ndtr, ndtri
from x2 import cond_correct, perms
from w1 import load, fwd
W = load("weights.npz"); Wih, Whh, Wo = W
rng = np.random.default_rng(4); n = 3000; K = 300
x = rng.standard_normal((n, 10)); s = -np.sort(-x, 1); M, S, R = s[:, 0], s[:, 1], s[:, 2]
pidx = rng.integers(0, 720, n); P = perms[pidx]
pc_all, ind_all = cond_correct(M, S, R, W); pc = pc_all[np.arange(n), pidx]; ind = ind_all[np.arange(n), pidx]
# true conditional: redraw lower inputs K times from N(0,1) truncated above R
mc = np.zeros(n)
for j in range(n):
    u = rng.random((K, 7))*ndtr(R[j]); low = ndtri(u)
    X = np.empty((K, 10)); mask = np.ones(10, bool); mask[P[j]] = False
    X[:, P[j]] = [M[j], S[j], R[j]]; X[:, mask] = low
    mc[j] = np.mean((fwd(X, Wih, Whh)@Wo.T).argmax(1) == P[j, 1])
print("means: true cond", mc.mean().round(4), "| blur formula", pc.mean().round(4), "| MF step", ind.mean().round(4))
print("mean |blur - true|", np.abs(pc-mc).mean().round(4), "| mean |step - true|", np.abs(ind-mc).mean().round(4))
first_low = np.array([np.min(np.setdiff1d(np.arange(10), p)) for p in P])
a = (M-S)/np.abs(M); b = (S-R)/np.abs(M)
for lab, m in [("S<0", S < 0), ("clean a,b>.1,S>0", (a > .1) & (b > .1) & (S > 0)), ("near tie b<.05", (b < .05) & (S > 0)), ("near tie a<.05", (a < .05) & (S > 0))]:
    print(f"{lab:18s} n {m.sum():5d} | true {mc[m].mean():.3f} blur {pc[m].mean():.3f} step {ind[m].mean():.3f}")
for t in range(4):
    m = first_low == t; print(f"earliest lower input at t={t}: n {m.sum()} true {mc[m].mean():.3f} blur {pc[m].mean():.3f}")
bad = np.argsort(mc-pc)[-5:]
for j in bad: print("worst:", "pos", P[j], "M S R", np.round([M[j], S[j], R[j]], 2), "true", mc[j].round(2), "blur", pc[j].round(3), "step", ind[j])
