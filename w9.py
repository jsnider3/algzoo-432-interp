# Upper bound on "most informative first": Neyman-allocation gain p(1-p)/(sum_h w_h sigma_h)^2 with TRUE stratum error
# rates, for several stratifications. No stratification can do better than its bound (any sequential rule included).
import numpy as np, sys
from w1 import load, pl_vec
AB = [0, .02, .05, .1, .2, .4, np.inf]; BB = [0, .01, .03, .06, .1, .2, .4, np.inf]; SB = [-np.inf, -.5, 0]
def s44(x):   # same strata as w8.py
    s = -np.sort(-x, 1); M, S, R = s[:, 0], s[:, 1], s[:, 2]
    a = (M-S)/np.abs(M); b = (S-R)/np.abs(M)
    h = (len(SB)-1) + (np.digitize(a, AB)-1)*(len(BB)-1) + (np.digitize(b, BB)-1)
    return np.where(S < 0, np.digitize(S, SB)-1, h)
def feats(x):
    o = np.argsort(-x, 1); s = np.take_along_axis(x, o, 1); M, S, R = s[:, 0], s[:, 1], s[:, 2]
    return o[:, 0], o[:, 1], (M-S)/np.abs(M), (S-R)/np.abs(M), S
def bound(F, lab):
    K = lab.max()+1; n = np.bincount(lab, minlength=K); e = np.bincount(lab, 1-F, minlength=K)/np.maximum(n, 1)
    w = n/len(lab); p = F.mean()
    return p*(1-p)/np.sum(w*np.sqrt(e*(1-e)))**2
rng = np.random.default_rng(2)
x = rng.standard_normal((1 << 23, 10)); iM, iS, a, b, S = feats(x)
L44 = s44(x)
ab = np.digitize(a, [.01, .02, .05, .1, .2, .4])*8 + np.digitize(b, [.005, .01, .03, .06, .1, .2, .4])
Lfine = np.where(S < 0, np.digitize(S, [-1, -.5, -.2]), 4+ab)
Lpos = Lfine*100 + iM*10 + iS                      # + ages of max and 2nd max
for f in sys.argv[1:]:
    F = pl_vec(x, load(f))
    print(f"{f:22s} acc {F.mean():.3f} | Neyman bound: 44 strata {bound(F, L44):5.2f}x | 60 finer gap strata {bound(F, Lfine):5.2f}x | "
          f"+ positions of max & 2nd (6000) {bound(F, np.unique(Lpos, return_inverse=True)[1]):5.2f}x", flush=True)
