import sys, time, numpy as np
from o3 import boundary_quad
cfg=[(8,6,12,10),(12,8,16,12),(16,12,24,12),(20,16,48,14)]
true,smf=0.9532,0.9535
for nr,ng2,ns,nb in cfg:
    t0=time.time(); v,m,nev=boundary_quad("weights.npz",nr=nr,ng2=ng2,nscan=ns,nbis=nb)
    sd=np.sqrt(true*(1-true)/nev)
    print(f"mystery boundary-aware nr={nr} ng2={ng2} scan={ns}: {nev/1e6:.2f}M evals, estimate {v:.4f}, |error vs true| {abs(v-true)*100:.3f} pts | sampling sd at same cost {sd*100:.3f} pts | ratio {abs(v-true)/sd:.0f}x [{time.time()-t0:.0f}s]",flush=True)
