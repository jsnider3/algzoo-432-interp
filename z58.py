"""Batch size check: G(1,K) trained on the 1,024 mystery-recipe networks at batch 128 (z55's 'unif'), scored on the 64
mystery-recipe networks trained at batch 64 (pop_unif/b64_seed5.npz, labels on 2^22 N(0,1) inputs), with the mystery
model's accuracy percentile in each population.
Usage: python z58.py
"""
import os, numpy as np, torch as th
from z2 import load
from batch_train import accuracy
import gfeat
def labels(W,path,log2=22,seed=55):          # as z55.labels
    if os.path.exists(path): return np.load(path)
    g=th.Generator(device=gfeat.dev).manual_seed(seed); x=th.randn(2**log2,10,generator=g,device=gfeat.dev); P=[w.to(gfeat.dev).float() for w in W]
    a=np.concatenate([accuracy([p[i:i+256] for p in P],x) for i in range(0,len(P[0]),256)]); np.save(path,a); return a
PU,_=load('pop_unif/b128_n1024_seed4.npz'); yu=np.load('pop_unif/b128_n1024_seed4.acc22.npy')
P6,_=load('pop_unif/b64_seed5.npz'); y6=labels(P6,'pop_unif/b64_seed5.acc22.npy'); ym=0.95304775
print(f"b64: mean {100*y6.mean():.2f}% SD {100*y6.std():.2f}; mystery percentile: b64 {100*(y6<ym).mean():.1f}, b128 (1,024) {100*(yu<ym).mean():.1f}")
W=[th.cat([a.float(),b.float()]) for a,b in zip(PU,P6)]; y=np.r_[yu,y6]; tr=np.arange(1024); te=np.arange(1024,1088)
for K in (1,2,4):
    F,net,C=gfeat.summary(W,1,K); X=np.c_[gfeat.stats(*[w.numpy().astype(float) for w in W]),F]; tot=gfeat.cost(net,C,1,X.shape[1],1024)
    e=gfeat.fit_predict(X,y,tr,te)-y6; r=np.sqrt((e**2).mean())
    print(f"G(1,{K}) {tot:7.1f} FE: b64 RMS {100*r:.3f} mean {100*e.mean():+.3f} MSE/s {r**2/((y6*(1-y6)).mean()/tot):.2f}",flush=True)
