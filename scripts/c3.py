import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(2).standard_normal((400000,T)); L=run(x,Wih,Whh)[:,-1]@Wo.T; pm=L.argmax(1)
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]; p1=o[:,0]
    tr=np.arange(len(x))<300000; te=~tr
    def fit(keys,F):
        P=np.zeros_like(L); C={}
        for k in np.unique(keys):
            s=keys==k; c,*_=np.linalg.lstsq(F[s&tr],L[s&tr],rcond=None); P[s]=F[s]@c; C[k]=c
        return P,C
    PA,CA=fit(aM,np.c_[x,M]); PB,_=fit(aM*10+aS,np.c_[x,M,S])
    # diagonality of cells(aM) fit: for cell with max at position p, rows i!=p: |diag| / sum|offdiag| over x-columns
    dg=[]
    for a,c in CA.items():
        p=9-a; W=c[:10].T   # (logit i, input j)
        for i in range(10):
            if i==p: continue
            off=np.abs(np.delete(W[i],[i,p])).sum(); dg.append(abs(W[i,i])/(off+1e-9))
    # "suppress-the-max then argmax of x" baseline equals true answer; measure how close fitted logits are to c*x_i
    print(f"{name:8s} acc {(pm==o[:,1]).mean():.4f} | agreement of logit fit cells(aM)x[x,M]: {(PA.argmax(1)==pm)[te].mean():.4f}, cells(aM,aS)x[x,M,S]: {(PB.argmax(1)==pm)[te].mean():.4f} | median diag/offdiag ratio {np.median(dg):.2f}")
