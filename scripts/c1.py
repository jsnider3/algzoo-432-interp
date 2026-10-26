import numpy as np, torch as th
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def probe(Hm,y,k,iters=300):
    Hm=th.tensor(Hm,dtype=th.float32); y=th.tensor(y); n=len(Hm)*3//4
    mu,sd=Hm[:n].mean(0),Hm[:n].std(0)+1e-6; Hm=(Hm-mu)/sd
    lin=th.nn.Linear(Hm.shape[1],k); opt=th.optim.Adam(lin.parameters(),lr=0.05)
    for _ in range(iters):
        opt.zero_grad(); th.nn.functional.cross_entropy(lin(Hm[:n]),y[:n]).backward(); opt.step()
    return (lin(Hm[n:]).argmax(1)==y[n:]).float().mean().item()
def r2cells(h,keys,M,S,x9):
    pred=np.zeros_like(h)
    for k in np.unique(keys):
        s=keys==k; A=np.c_[M[s],S[s],x9[s]]; c,*_=np.linalg.lstsq(A,h[s],rcond=None); pred[s]=A@c
    return 1-((pred-h)**2).mean(0)/(h.var(0)+1e-12)
for name,path in [("mystery","weights.npz"),("seed 0","weights_seed0.npz"),("seed 2","weights_seed2.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(1).standard_normal((200000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]; x9=x[:,9]
    print(f"\n=== {name}")
    print("Wih:", Wih.round(1))
    print("frac active final:", (h>0).mean(0).round(2))
    print("R2 cells aM   ", r2cells(h,aM,M,S,x9).round(2))
    print("R2 cells aS   ", r2cells(h,aS,M,S,x9).round(2))
    print("R2 cells aM,aS", r2cells(h,aM*10+aS,M,S,x9).round(2))
    d=aS-aM+9
    before=aS>aM
    print(f"probe aS {probe(h,aS,10):.3f} | aS-aM {probe(h,d,19):.3f} | aS given S-before-M {probe(h[before],aS[before],10):.3f} | given S-after-M {probe(h[~before],aS[~before],10):.3f}")
