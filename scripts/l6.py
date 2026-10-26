import numpy as np
from seeds_clocks import load, run, T
x=np.random.default_rng(601).standard_normal((100000,T))
rate={0:None,1:None,2:None,3:None,4:None}
print("linear decodability (R2) of prefix M and S from the state h_t, pooled over t=2..9")
for s in range(5):
    row=[]
    for e in [0,20,21,22]:
        Wih,Whh,Wo=load(f"ckpt/s{s}_e{e}.npz"); hs=run(x,Wih,Whh); r2=[]
        for which in [0,1]:
            Hs=[];Ys=[]
            for t in range(2,T):
                srt=-np.sort(-x[:,:t+1],1); Hs.append(hs[:,t]); Ys.append(srt[:,which])
            H=np.concatenate(Hs); Y=np.concatenate(Ys); A=np.c_[H,np.ones(len(H))]; c,*_=np.linalg.lstsq(A,Y,rcond=None)
            r2.append(1-np.mean((A@c-Y)**2)/Y.var())
        row.append(f"2^{e}: M {r2[0]:.2f} S {r2[1]:.2f}" if e else f"init: M {r2[0]:.2f} S {r2[1]:.2f}")
    print(f"seed {s}: "+" | ".join(row))
