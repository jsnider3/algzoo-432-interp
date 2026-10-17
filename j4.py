import numpy as np
from seeds_clocks import load, run, T
x=np.random.default_rng(191).standard_normal((200000,T)); n=10
for e in [22,23,24,25,26,28,30]:
    Wih,Whh,Wo=load(f"ckpt/s2_e{e}.npz"); hs=run(x,Wih,Whh); out=[]
    for t in [4,7]:
        P=x[:,:t]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]
        thr=-(hs[:,t-1]@Whh[n])/Wih[n]; A=np.c_[M,S,np.ones(len(x))]; c,*_=np.linalg.lstsq(A,thr,rcond=None)
        r2=1-((A@c-thr)**2).mean()/thr.var(); out.append(f"t={t}: thr≈{c[0]:+.2f}M{c[1]:+.2f}S{c[2]:+.2f} (R2 {r2:.2f})")
    print(f"2^{e}: Wih[10]={Wih[n]:+.2f} | "+" | ".join(out))
