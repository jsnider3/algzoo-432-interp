# How much variance is left after integrating x10 exactly, and how much more would integrating x9 remove?
# Var(F) [plain] vs Var(g10) [x10 exact] vs Var(g9,10) [x9 by fine grid, x10 exact]. Raw per-node gains, ignoring cost.
import numpy as np
from w1 import load
from w5 import condprob
rng = np.random.default_rng(5)
z = np.linspace(-6, 6, 1201); wz = np.exp(-z**2/2); wz /= wz.sum()
for f in ["weights.npz", "weights_seed2.npz", "weights_seed0.npz"]:
    Wih, Whh, Wo = load(f); wi = Wih.reshape(-1).astype(float)
    g10 = condprob(rng.standard_normal((200000, 9)), wi, Whh, Wo)
    P = rng.standard_normal((3000, 8))
    Y = np.concatenate([np.repeat(P, len(z), 0), np.tile(z, len(P))[:, None]], 1)
    g910 = (condprob(Y, wi, Whh, Wo).reshape(len(P), len(z))*wz).sum(1)
    p = g10.mean()
    print(f"{f}: p {p:.4f} | Var plain {p*(1-p):.4f}, x10 exact {g10.var():.5f} ({p*(1-p)/g10.var():.1f}x), "
          f"x9+x10 exact {g910.var():.5f} ({p*(1-p)/g910.var():.1f}x)", flush=True)
