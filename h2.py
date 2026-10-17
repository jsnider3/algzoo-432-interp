import numpy as np
from seeds_clocks import load, run, T
Wih,Whh,Wo=load("weights_seed2.npz")
x=np.random.default_rng(82).standard_normal((300000,T)); hs=run(x,Wih,Whh)
for n in [10,15,2,3,0,6]:
    print(f"\nseed 2 neuron {n} (Wih {Wih[n]:+.1f}): firing rate at step t by where x_t falls vs prefix M,S")
    rows={'x>M':[],'S<x<M':[],'x<S':[]}
    for t in range(2,T):
        P=x[:,:t]; srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; xt=x[:,t]; f=hs[:,t,n]>0
        rows['x>M'].append(f[xt>M].mean()); rows['S<x<M'].append(f[(xt>S)&(xt<M)].mean()); rows['x<S'].append(f[xt<S].mean())
    for k,v in rows.items(): print(f"  {k:6s} " + " ".join(f"{a:.2f}" for a in v))
    # threshold fit
    t=7; P=x[:,:t]; o=np.argsort(-P,1); srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; aM=t-1-o[:,0]
    thr=-(hs[:,t-1]@Whh[n])/Wih[n]; A=np.c_[M,S,np.ones(len(x))]; c,*_=np.linalg.lstsq(A,thr,rcond=None)
    print(f"  threshold at t=7 ≈ {c[0]:.2f}·M + {c[1]:.2f}·S + {c[2]:.2f}  (R2 {1-((A@c-thr)**2).mean()/thr.var():.2f}); fires when x {'>' if Wih[n]>0 else '<'} thr")
