import numpy as np
from seeds_clocks import load, run, T
rng=np.random.default_rng(92)
def estimate(path, nprobe=20000, umax=0.3):
    Wih,Whh,Wo=load(path); f=lambda x:(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)
    x=rng.standard_normal((nprobe,T)); srt=-np.sort(-x[:,:9],1); M,S=srt[:,0],srt[:,1]
    keep=S>0; x,M,S=x[keep],M[keep],S[keep]; n=len(x)
    u=rng.uniform(-umax,umax,n); x[:,9]=S+u*M; p=f(x); tgt=np.argsort(x,1)[:,-2]
    err=(p!=tgt)
    bins=np.linspace(0,umax,16); ib=np.clip(np.digitize(np.abs(u),bins)-1,0,14)
    curve=np.array([err[ib==b].mean() for b in range(15)])
    xs=rng.standard_normal((400000,T)); xs=xs[(-np.sort(-xs,1))[:,1]<0][:2000]; eneg=np.mean(f(xs)!=np.argsort(xs,1)[:,-2])
    z=rng.standard_normal((4_000_000,T)); s=-np.sort(-z,1); Mz,Sz,Rz=s[:,0],s[:,1],s[:,2]; pos=Sz>0
    e=lambda g: curve[np.clip(np.digitize(g,bins)-1,0,14)]
    # an error can come from either comparison; treat as independent
    pc=(1-e((Sz-Rz)/Mz))*(1-e((Mz-Sz)/Mz))
    return np.mean(np.where(pos,pc,1-eneg)), curve, n+2000
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz"),("seed 0","weights_seed0.npz"),("seed 1","weights_seed1.npz"),("seed 4","weights_seed4.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(5).standard_normal((3000000,T)); true=np.mean((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2])
    acc,curve,cost=estimate(path); sd=np.sqrt(true*(1-true)/cost)
    print(f"{name:8s} true {true:.4f} | estimate {acc:.4f} (error {acc-true:+.4f}) | {cost} evals, sampling sd {sd:.4f} | probe error curve (|u| 0→0.3): {np.round(curve[[0,1,2,4,7,14]],3)}")
