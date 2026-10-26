# Benchmark w1 estimators across independently trained networks:
# variance across independent scrambles vs IID MC at equal cost.  usage: w2.py m1,m2,.. file1.npz file2.npz ...
import numpy as np, time, sys
from w1 import *
ms = [int(a) for a in sys.argv[1].split(',')]; files = sys.argv[2:]
nets = {f: load(f) for f in files}
rng = np.random.default_rng(7)
x = rng.standard_normal((1 << 17, 10)); W = nets[files[0]]
t = time.perf_counter(); [batched(pl_vec, x, W) for _ in range(5)]; tp = time.perf_counter()-t
t = time.perf_counter(); [batched(cp_vec, x[:, :9], W) for _ in range(5)]; tc = time.perf_counter()-t
c = tc/tp; print(f"cost ratio cond/forward = {c:.2f}", flush=True)
for m in ms:
    reps = max(16, 2**(18-m)) if m < 18 else 16
    X10 = [transport(u, 10) for u in design(m, 10, reps, 1000*m)]     # network-independent prework
    Y9 = [transport(u, 9) for u in design(m, 9, reps, 2000*m)]
    for f, W in nets.items():
        e0 = np.array([batched(pl_vec, rng.standard_normal((2**m, 10)), W) for _ in range(reps)])
        e1 = np.array([batched(pl_vec, x, W) for x in X10]); e2 = np.array([batched(cp_vec, y, W) for y in Y9])
        p = e2.mean(); mc = p*(1-p)/2**m
        print(f"n=2^{m} reps {reps} {f:20s} p {p:.5f} | iid {mc/np.var(e0,ddof=1):5.2f}x | sobol {mc/np.var(e1,ddof=1):5.2f}x | "
              f"cmc+sobol {mc/np.var(e2,ddof=1)/c:5.2f}x/cost (raw {mc/np.var(e2,ddof=1):5.1f}x)", flush=True)
