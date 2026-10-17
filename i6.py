import numpy as np
from seeds_clocks import load, run, T
GB=np.array([0,.01,.02,.035,.05,.075,.1,.15,.2,.3,10])  # normalized-gap bins
def features(x):
    o=np.argsort(-x,1); s=-np.sort(-x,1); M,S,R=s[:,0],s[:,1],s[:,2]
    g12=(M-S)/np.abs(M); g23=(S-R)/np.abs(M)
    typ=(g12<g23).astype(int)                    # 1: M~S decisive, 0: S~R3 decisive
    g=np.where(typ==1,g12,g23)
    tstar=np.where(typ==1,np.maximum(o[:,0],o[:,1]),np.maximum(o[:,1],o[:,2]))
    gb=np.clip(np.digitize(g,GB)-1,0,len(GB)-2)
    neg=S<0
    cell=np.where(neg, 0, 1+ (typ*10+tstar)*10+gb)   # cell 0 = blind spot
    return cell
NC=1+2*10*10
rng=np.random.default_rng(101)
W=np.bincount(features(rng.standard_normal((20_000_000,T))),minlength=NC)/20_000_000   # model-free cell weights
def estimate(f,budget,rng,smooth=True):
    # probe: plain samples (no oversampling) -> per-cell error; smooth across t* within (type,gap) when cells are sparse
    x=rng.standard_normal((budget,T)); c=features(x); e=(f(x)!=np.argsort(x,1)[:,-2]).astype(float)
    n=np.bincount(c,minlength=NC); k=np.bincount(c,weights=e,minlength=NC)
    est_cell=np.where(n>0,k/np.maximum(n,1),np.nan)
    if smooth:
        # structural assumption: error depends on (type, gap bin, t*) ; back off to (type, gap bin) pooled over t*
        for typ in range(2):
            for gb in range(10):
                idx=[1+(typ*10+t)*10+gb for t in range(10)]
                pooled=k[idx].sum()/max(n[idx].sum(),1)
                for i in idx:
                    lam=n[i]/(n[i]+20)                   # shrink small cells toward pooled
                    est_cell[i]=lam*(k[i]/max(n[i],1))+(1-lam)*pooled
        est_cell[0]=k[0]/max(n[0],1) if n[0]>0 else 0.77
    est_cell=np.nan_to_num(est_cell)
    return 1-np.sum(W*est_cell)
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); f=lambda x:(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)
    xt=np.random.default_rng(5).standard_normal((3000000,T)); true=np.mean(f(xt)==np.argsort(xt,1)[:,-2])
    for budget in [2000,8000]:
        naive=[np.mean(f(z)==np.argsort(z,1)[:,-2]) for z in (rng.standard_normal((budget,T)) for _ in range(200))]
        strc=[estimate(f,budget,rng) for _ in range(200)]
        mn=np.mean((np.array(naive)-true)**2); ms=np.mean((np.array(strc)-true)**2)
        print(f"{name:8s} budget {budget}: naive MSE {mn:.2e} | structured (type × gap × step, shrunk) MSE {ms:.2e}, bias {np.mean(strc)-true:+.4f} -> {mn/ms:.2f}x")
