import numpy as np
from seeds_clocks import load, run, T
# REPORT 7.14 barrier caveat: straight-line loss between models that are known to be in one basin.
# fork_s0.npz (saved by k8.py) is the switched fork of seed 0 from 2^22; ckpt/s0_e22 is its parent, ckpt/s0_e24 the original run's descendant.
x=np.random.default_rng(301).standard_normal((200000,T)); y=np.argsort(x,1)[:,-2]
def loss(W):
    L=run(x,W[0],W[1])[:,-1]@W[2].T; L=L-L.max(1,keepdims=True)
    return -np.mean(L[np.arange(len(x)),y]-np.log(np.exp(L).sum(1)))
f=np.load("fork_s0.npz"); fork=[f['arr_0'],f['arr_1'],f['arr_2']]; e22=load("ckpt/s0_e22.npz"); e24=load("ckpt/s0_e24.npz")
for name,A,B in [("memory 2^24 -> switched fork (k6.py)",e24,fork),("2^22 -> its own 2^24 descendant",e22,e24),("2^22 -> switched fork",e22,fork)]:
    print(f"{name:38s}", " ".join(f"{loss([(1-a)*p+a*q for p,q in zip(A,B)]):.3f}" for a in [0,.25,.5,.75,1]))
