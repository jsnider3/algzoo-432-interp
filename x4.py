# Near-tie error structure: symmetric blur vs gap, and which features predict near-tie errors (gradient boosting).
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from w1 import load, fwd
rng = np.random.default_rng(0); x = rng.standard_normal((4000000, 10))
o = np.argsort(-x, 1); s = np.take_along_axis(x, o, 1); M, S, R = s[:, 0], s[:, 1], s[:, 2]
b = (S-R)/np.abs(M); a = (M-S)/np.abs(M); nt = (b < 0.03) & (S > 0); na = (a < 0.03) & (S > 0)
feats = {"gap b only": np.c_[b], "+ a, M": np.c_[b, a, M], "+ positions of M,S,R": np.c_[b, a, M, o[:, :3]],
         "+ lower inputs & positions": np.c_[b, a, M, o[:, :3], s[:, 3:], o[:, 3:]], "everything in time order": np.c_[b, a, M, o[:, :3], x]}
for f in ["weights.npz", "weights_seed0.npz"]:
    Wih, Whh, Wo = load(f); out = (fwd(x, Wih, Whh)@Wo.T).argmax(1); err = out != o[:, 1]
    print(f, "err", err.mean().round(4))
    print("   near-tie b: S before R", err[nt & (o[:, 1] < o[:, 2])].mean().round(3), "| S after R", err[nt & (o[:, 1] > o[:, 2])].mean().round(3))
    print("   near-tie a: M before S", err[na & (o[:, 0] < o[:, 1])].mean().round(3), "| M after S", err[na & (o[:, 0] > o[:, 1])].mean().round(3))
    for lo, hi in [(0, .005), (.005, .01), (.01, .02), (.02, .03), (.03, .06), (.06, .1)]:
        m = (S > 0) & (b >= lo) & (b < hi); print(f"   b in [{lo},{hi}): err {err[m].mean():.3f}")
    e = err[nt]; n = len(e); tr = np.arange(n) < 0.7*n; base = e.mean()
    for k, X in feats.items():
        X = X[nt]; m = HGB(max_iter=400, learning_rate=0.1, max_leaf_nodes=63).fit(X[tr], e[tr])
        print(f"   {k:30s} residual var {np.mean((e[~tr]-m.predict_proba(X[~tr])[:, 1])**2):.4f} (vs {base*(1-base):.4f})", flush=True)
