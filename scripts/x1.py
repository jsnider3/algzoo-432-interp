# Predict the network's decision boundary from the explanation: an online 2nd-max tracker whose event thresholds
# (new max / new 2nd max / neither) are the network's, possibly age-dependent. Fit the thresholds to the network's
# outputs, then measure how often tracker and network disagree. Disagreement d bounds the control-variate gain:
#   gain = p(1-p) / (P(F != G) - (p - E G)^2)   with F = network correct, G = tracker correct.
import numpy as np, sys
from scipy.optimize import minimize
from w1 import load, fwd
T = 10
def tracker(x, th):
    """th = (t1, t1_ageM, t2, t2_ageM, t2_ageS, t2_gap). Returns predicted answer position."""
    n = len(x); r = np.arange(n)
    iM = np.zeros(n, int); M = x[:, 0].copy()
    big = x[:, 1] > M
    iS = np.where(big, 0, 1); S = np.where(big, M, x[:, 1]); iM = np.where(big, 1, 0); M = np.maximum(M, x[:, 1])
    for t in range(2, T):
        v = x[:, t]; A = np.abs(M); aM = t-iM; aS = t-iS; gap = (M-S)/A
        newM = (v-M)/A > th[0] + th[1]*aM
        newS = ~newM & ((v-S)/A > th[2] + th[3]*aM + th[4]*aS + th[5]*gap)
        S = np.where(newM, M, np.where(newS, v, S)); iS = np.where(newM, iM, np.where(newS, t, iS))
        M = np.where(newM, v, M); iM = np.where(newM, t, iM)
    return iS
def fit(x, out, sweeps=4):
    """coordinate grid search, zooming in each sweep (objective is piecewise constant)."""
    th = np.zeros(6); half = np.array([.6, .1, .6, .1, .1, 1.])
    for sw in range(sweeps):
        for j in range(6):
            vals = th[j] + np.linspace(-half[j], half[j], 41); sc = []
            for v in vals:
                t2 = th.copy(); t2[j] = v; sc.append(np.mean(tracker(x, t2) == out))
            th[j] = vals[int(np.argmax(sc))]
        half = half/3
    return th
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    xf = rng.standard_normal((100000, T)); xt = rng.standard_normal((400000, T))
    tf, tt = np.argsort(xf, 1)[:, -2], np.argsort(xt, 1)[:, -2]
    for f in sys.argv[1:]:
        Wih, Whh, Wo = load(f)
        of = (fwd(xf, Wih, Whh)@Wo.T).argmax(1); ot = (fwd(xt, Wih, Whh)@Wo.T).argmax(1)
        th = fit(xf, of)
        g = tracker(xt, th); F = ot == tt; G = g == tt
        p = F.mean(); d = np.mean(F != G); gain = p*(1-p)/(d-(p-G.mean())**2)
        print(f"{f:20s} acc {p:.4f} | tracker acc {G.mean():.4f} | output agreement {np.mean(g == ot):.4f} | "
              f"correctness disagreement {d:.4f} | ceiling gain {gain:.2f}x | th {np.round(th, 3)}", flush=True)
