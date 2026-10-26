import numpy as np
from seeds_clocks import load, run, T
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(2).standard_normal((400000,T)); L=run(x,Wih,Whh)[:,-1]@Wo.T; pm=L.argmax(1)
    L=L-L.mean(1,keepdims=True)
    o=np.argsort(-x,1); srt=-np.sort(-x,1); aM=9-o[:,0]; g=srt[:,1]-srt[:,2]
    tr=np.arange(len(x))<300000; te=~tr; P=np.zeros_like(L)
    for a in range(10):
        s=aM==a; c,*_=np.linalg.lstsq(x[s&tr],L[s&tr],rcond=None); P[s]=x[s]@c
    r2=1-((P-L)[te]**2).mean()/L[te].var()
    P2=P.copy(); P2[np.arange(len(x)),o[:,0]]=-1e9   # also force-suppress the max
    easy=te&(g>0.3)
    print(f"{name:8s} logit R2 {r2:.3f} | agreement: linear-given-maxpos {(P.argmax(1)==pm)[te].mean():.3f}, + hard max suppression {(P2.argmax(1)==pm)[te].mean():.3f}, same on easy (gap>.3) {(P2.argmax(1)==pm)[easy].mean():.3f} | model acc on easy {(pm==o[:,1])[easy].mean():.3f}")
