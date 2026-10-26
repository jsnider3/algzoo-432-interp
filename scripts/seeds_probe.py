import numpy as np, torch as th
from seeds_clocks import load, run, T
def probe(Hm,y,k):
    Hm=th.tensor(Hm,dtype=th.float32); y=th.tensor(y); n=len(Hm)*3//4
    mu,sd=Hm[:n].mean(0),Hm[:n].std(0)+1e-6; Hm=(Hm-mu)/sd
    lin=th.nn.Linear(Hm.shape[1],k); opt=th.optim.Adam(lin.parameters(),lr=0.05)
    for _ in range(300):
        opt.zero_grad(); th.nn.functional.cross_entropy(lin(Hm[:n]),y[:n]).backward(); opt.step()
    return (lin(Hm[n:]).argmax(1)==y[n:]).float().mean().item()
def plane_steps(h, M, ag):
    # centroid of h/M for each age 2..9, project on top-2 PCs, report angle steps
    C=np.stack([(h/M[:,None])[ag==a].mean(0) for a in range(2,10)]); C=C-C.mean(0)
    U,s,Vt=np.linalg.svd(C,full_matrices=False); P=C@Vt[:2].T
    ang=np.degrees(np.unwrap(np.arctan2(P[:,1],P[:,0]))); return np.diff(ang), s[:2]**2/(s**2).sum()
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(1).standard_normal((200000,T)); h=run(x,Wih,Whh)[:,-1]
    o=np.argsort(-x,1); aM=9-o[:,0]; aS=9-o[:,1]; M=-np.sort(-x,1)[:,0]; keep=M>0.3
    pM=probe(h,aM,10); pS=probe(h,aS,10)
    dM,vM=plane_steps(h[keep],M[keep],aM[keep]); dS,vS=plane_steps(h[keep],M[keep],aS[keep])
    print(f"{name:8s} probe ageM {pM:.3f} ageS {pS:.3f} | ageM-centroid steps {np.round(dM,0)} (var {vM.sum():.2f}) | ageS-centroid steps {np.round(dS,0)} (var {vS.sum():.2f})")
