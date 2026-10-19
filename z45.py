"""Calibration against the population: cross-validated fits of true accuracy on deterministic per-network quantities
from z41 (pop1k/seed0_z41.npz), optionally plus network features.  8 interleaved folds over the 1,024 networks
(network i in fold i mod 8).  Network features: mean/SD/min/max over the 16 neurons of z2's per-neuron features
(pop1k/seed0.roles.npz); the first 5 are weights only (free), all 24 need z2's 4,096 probe forwards.
OLS for the deterministic quantities alone; ridge on standardized features when network features are added (alpha
picked from a small grid by the same CV, slightly optimistic).  Zoo errors come from a fit on all 1,024.
Costs are forward-equivalents per network.  Usage: python z45.py
"""
import numpy as np
r=np.load('pop1k/reference.npz'); acc=r['acc']; F=dict(np.load('pop1k/seed0_z41.npz'))
c6=F['anchor6_cost']; flag=F['flag']; pop=np.arange(1024); pq=(acc[pop]*(1-acc[pop])).mean()
F['est']=F['anchor6']+F['I4c']-F['I3c']+F['brk']+F['fix']; F['est_g']=F['anchor6']+F['I4c']-F['I3c']+F['brk_g']+F['fix_g']
Rf=np.load('pop1k/seed0.roles.npz')['feats'].astype(float)
agg=lambda f: np.concatenate([f.mean(1),f.std(1),f.min(1),f.max(1)],1)
NF={'none':(np.zeros((1024,0)),0),'weights (free)':(agg(Rf[:,:,:5]),0),'roles (4,096)':(agg(Rf),4096)}
DET=[('none',[],np.zeros(1030)),
     ('n1 coarse I4, I3',['I4c1','I3c1'],np.full(1030,11520.)),
     ('n2 coarse I4, I3',['I4c','I3c'],np.full(1030,92160.)),
     ('exact MF3 anchor (order 6)',['anchor6'],c6),
     ('exact MF3 + n2 coarse I4, I3',['anchor6','I4c','I3c'],c6+92160),
     ('two-level screened: total',['est_g'],c6+46080*(14+36*flag)),
     ('two-level screened: components',['anchor6','I4c','I3c','brk_g','fix_g'],c6+46080*(14+36*flag)),
     ('two-level full: total',['est'],c6+2304000),
     ('two-level full: components',['anchor6','I4c','I3c','brk','fix'],c6+2304000)]
def fit(X,y,al):
    I=np.eye(X.shape[1])*al; I[0,0]=0; return np.linalg.solve(X.T@X+I,X.T@y)
def cv(D,Nf):
    X=np.c_[D[pop],Nf]; ridge=Nf.shape[1]>0
    if ridge: mu_,sd_=X.mean(0),np.maximum(X.std(0),1e-12); X=(X-mu_)/sd_
    X=np.c_[np.ones(1024),X]; best=None
    for al in ((1e-3,1e-2,0.1,1,10,100) if ridge else (0,)):
        pred=np.zeros(1024)
        for f in range(8):
            te=pop%8==f; pred[te]=X[te]@fit(X[~te],acc[pop][~te],al)
        e=np.sqrt(np.mean((pred-acc[pop])**2))
        if best is None or e<best[0]: best=(e,al)
    zoo=None
    if not ridge:
        b=fit(X,acc[pop],0); zoo=100*(np.c_[np.ones(6),D[1024:]]@b-acc[1024:])
    return 100*best[0],zoo
print(f"{'deterministic part':32s} {'+ network features':18s} {'forwards':>9s} {'CV RMS':>7s} {'sampling':>8s} {'MSE/samp':>8s}   zoo errors (OLS on all 1,024)")
for dn,ks,cost in DET:
    D=np.column_stack([F[k] for k in ks]) if ks else np.zeros((1030,0))
    for fn,(Nf,cf) in NF.items():
        if not ks and fn=='none': e,z=100*acc[pop].std(),None
        else: e,z=cv(D,Nf)
        C=cost[pop].mean()+cf; s=100*np.sqrt(pq/C) if C else np.inf
        print(f"{dn:32s} {fn:18s} {C:9.0f} {e:7.3f} {s:8.3f} {(e/s)**2 if C else 0:8.2f}   "+(" ".join(f"{v:+.2f}" for v in z) if z is not None else ""))
raw=lambda k: 100*np.sqrt(np.mean((F[k][pop]-acc[pop])**2))
print(f"uncalibrated RMS on all 1,024: exact MF3 anchor {raw('anchor6'):.3f}, two-level screened {raw('est_g'):.3f}, two-level full {raw('est'):.3f}")
