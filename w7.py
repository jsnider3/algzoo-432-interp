# Try integrating two inputs: x10 exact + q stratified points over x9 (unbiased). Cost vs matched forward (warm, interleaved, median).
import numpy as np, time
from w1 import load
from w5 import correct, condprob, condprob2
rng = np.random.default_rng(9)
Q = [2, 4, 8]
def tm(fn, reps=7):
    return fn
for f in ["weights.npz", "weights_seed2.npz", "weights_seed0.npz"]:
    Wih, Whh, Wo = load(f); wi = Wih.reshape(-1).astype(float)
    x = rng.standard_normal((1 << 16, 10)); Us = {q: rng.random((len(x), q)) for q in Q}
    jobs = {"fwd": lambda: correct(x, wi, Whh, Wo), "x10": lambda: condprob(x[:, :9], wi, Whh, Wo)}
    for q in Q: jobs[f"q{q}"] = (lambda q=q: condprob2(x[:, :8], Us[q], wi, Whh, Wo))
    out = {k: j() for k, j in jobs.items()}                 # warm-up (JIT) + values
    T = {k: [] for k in jobs}
    for _ in range(7):
        for k in rng.permutation(list(jobs)):
            t = time.perf_counter(); jobs[k](); T[k].append(time.perf_counter()-t)
    tf = np.median(T["fwd"]); p = out["x10"].mean()
    line = f"{f}: p {p:.4f}"
    for k in ["x10"]+[f"q{q}" for q in Q]:
        c = np.median(T[k])/tf; raw = p*(1-p)/out[k].var()
        line += f" | {k}: cost {c:.2f} raw {raw:.2f}x net {raw/c:.2f}x"
    print(line, flush=True)
