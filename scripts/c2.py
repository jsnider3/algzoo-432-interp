import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def cellfit(h,keys,F):
    pred=np.zeros_like(h)
    for k in np.unique(keys):
        s=keys==k; c,*_=np.linalg.lstsq(F[s],h[s],rcond=None); pred[s]=F[s]@c
    return 1-((pred-h)**2).mean(0)/(h.var(0)+1e-12)
for name,path in [("mystery","weights.npz"),("seed 0","weights_seed0.npz"),("seed 2","weights_seed2.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(1).standard_normal((200000,T)); h=run(x,Wih,Whh)[:,-1]
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]
    print(f"\n=== {name}  (final-step R2 per neuron)")
    print("raw x linear           ", cellfit(h,np.zeros(len(x),int),x).round(2))
    print("raw x + M,S linear     ", cellfit(h,np.zeros(len(x),int),np.c_[x,M,S]).round(2))
    print("cells(aM) x [x,M,S]    ", cellfit(h,aM,np.c_[x,M,S]).round(2))
    print("cells(aM,aS) x [x,M,S] ", cellfit(h,aM*10+aS,np.c_[x,M,S]).round(2))
