"""Independent check of ChatGPT round 6's network-level readout ("global summary" G(1,K)) on the z46 one-box caches,
and where its information comes from.
Per network, from the 720 K cells' two rows (MF4 and MF3 anchors): ten cell features (bits I4, I3; margins a = m4/s,
b = m3/s with s the RMS of m4; |a|, |b|; a/(.1+|a|), b/(.1+|b|), a/(.03+|a|), b/(.03+|b|)) averaged over all cells and
over each S slice; the logit-scale context (s, log s, mean m4, mean m3); 20 weight statistics (mean, SD, min, max over
hidden units of W_ih, diag W_hh, log row and column norms of W_hh, log column norms of W_o).
Readout: ridge or RBF kernel ridge on train-fold-standardized features, family and penalty chosen by 4 inner folds
inside each of 8 outer folds (a wider grid than ChatGPT's); folds by i mod 8 as ChatGPT, and by 3 random permutations.
Ablations: which feature groups carry the accuracy.  The cell features are invariant under the function-preserving
maps (positive diagonal rescaling of the hidden units, positive scaling of W_ih or of W_o); s, the raw margins and the
weight statistics are not.  "invariant weights" = the same 20 statistics after putting each network in a canonical
gauge (hidden unit j rescaled so its centred W_o column has unit norm, then W_ih scaled to unit norm).
Usage: python scripts/z49.py
"""
import numpy as np
from z2 import load
from seeds_clocks import load as zload
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wih,Whh,Wo=[np.concatenate([a.numpy().astype(float),np.stack([z[i] for z in Z]).astype(float)]) for i,a in enumerate(P)]
Wih=Wih.reshape(len(Wih),-1); acc=np.load('pop1k/reference.npz')['acc']; pq=acc*(1-acc); NT=len(acc); pop=np.arange(1024)
def stats(wi,A,O):
    F=np.stack([wi,np.diagonal(A,axis1=1,axis2=2),np.log(np.linalg.norm(A,axis=2)),np.log(np.linalg.norm(A,axis=1)),np.log(np.linalg.norm(O,axis=1))],2)
    return np.concatenate([F.mean(1),F.std(1),F.min(1),F.max(1)],1)
def canonical(wi,A,O):
    d=np.linalg.norm(O-O.mean(1,keepdims=True),axis=1)                 # (N,16) centred readout column norms
    A2=A*d[:,:,None]/d[:,None,:]; O2=O/d[:,None,:]; wi2=wi*d; wi2/=np.linalg.norm(wi2,axis=1,keepdims=True)
    return wi2,A2,O2
def cells(K):
    d=np.load(f'pop1k/seed0_z46_cells_b1_k{K}.npz'); m4,m3=d['m4'].astype(float),d['m3'].astype(float)
    s=np.sqrt((m4**2).mean(1,keepdims=True)); a,b=m4/s,m3/s
    f=np.stack([d['I4'],d['I3'],a,b,abs(a),abs(b),a/(.1+abs(a)),b/(.1+abs(b)),a/(.03+abs(a)),b/(.03+abs(b))],-1).astype(float)
    sl=np.arange(f.shape[1])%K; g=[f.mean(1)]+([f[:,sl==k].mean(1) for k in range(K)] if K>1 else [])
    return np.concatenate(g,1), np.c_[s,np.log(s),m4.mean(1),m3.mean(1)]
GAM=[.003,.01,.03,.1,.3,1.]; ALP=np.logspace(-5,3,9)
def fold_preds(x,tr,va):
    """predictions on va for every (family, gamma, alpha): ridge first, then RBF per gamma; (len(va), 9*(1+len(GAM)))"""
    m=x[tr].mean(0); sd=np.maximum(x[tr].std(0),1e-6); a,b=(x[tr]-m)/sd,(x[va]-m)/sd; y=acc[tr]-acc[tr].mean(); out=[]
    ev,V=eigh(a.T@a); q=V.T@(a.T@y); out.append(((b@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None]*len(tr))))
    D=cdist(a,a,'sqeuclidean'); Dv=cdist(b,a,'sqeuclidean')
    for g in GAM:
        ev,V=eigh(np.exp(-g/x.shape[1]*D)); q=V.T@y; out.append(((np.exp(-g/x.shape[1]*Dv)@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None])))
    return acc[tr].mean()+np.concatenate(out,1)
def nested(x,fold):
    pred=np.zeros(1024); pick=[]
    for f in range(8):
        tr,va=pop[fold!=f],pop[fold==f]; inner=(np.arange(len(tr))*4//len(tr))[np.argsort(np.argsort(tr*7919%1024))]
        loss=sum(((fold_preds(x,tr[inner!=j],tr[inner==j])-acc[tr[inner==j],None])**2).sum(0) for j in range(4))
        c=int(loss.argmin()); pick.append(c); pred[va]=fold_preds(x,tr,va)[:,c]
    return pred,pick
def name(c): fam=c//9; return ('ridge' if fam==0 else f'rbf g{GAM[fam-1]}')+f' a{ALP[c%9]:.0e}'
W=stats(Wih,Whh,Wo); Wc=stats(*canonical(Wih,Whh,Wo)); folds={'i mod 8':pop%8}
for s in range(3): folds[f'perm {s}']=np.random.default_rng(100+s).permutation(1024)%8
print(f"population SD of accuracy {100*acc[:1024].std():.3f} points; mean p(1-p) {pq[:1024].mean():.5f}")
print("RMS (points), cross-fitted on the 1,024; folds: i mod 8 | three random permutations; MSE/sampling at ChatGPT's total cost")
COST={1:618.55,2:1129.66,4:2151.90}
for K in (1,2,4):
    f,ctx=cells(K)
    sets={'cells only (invariant)':f,'cells + logit scale':np.c_[f,ctx],'cells + weights':np.c_[f,W],'cells + scale + weights (= G)':np.c_[f,ctx,W],
          'cells + invariant weights':np.c_[f,Wc],'weights only':W,'invariant weights only':Wc,'logit scale + weights':np.c_[ctx,W]}
    if K>1: sets={k:v for k,v in sets.items() if 'cells' in k}
    print(f"K = {K} (total cost {COST[K]} FE, sampling RMS {100*np.sqrt(pq[:1024].mean()/COST[K]):.3f})")
    for lab,x in sets.items():
        r=[];pk=None
        for fn,fo in folds.items():
            p,pick=nested(x,fo); r.append(100*np.sqrt(((p-acc[:1024])**2).mean())); pk=pk or pick
        ratio=(r[0]/100)**2/(pq[:1024].mean()/COST[K])
        print(f"  {lab:32s} {x.shape[1]:3d} feats  {r[0]:.4f} | {' '.join(f'{v:.4f}' for v in r[1:])}  MSE/s {ratio:.3f}  picks: {name(pk[0])}",flush=True)
# gauge check on 64 networks: random positive diagonal rescaling and W_o x 2 preserve every logit argmax
rng=np.random.default_rng(0); ix=np.arange(0,1024,16); Dg=np.exp(rng.uniform(-np.log(2),np.log(2),(64,16)))
wi2=Wih[ix]*Dg; A2=Whh[ix]*Dg[:,:,None]/Dg[:,None,:]; O2=2*Wo[ix]/Dg[:,None,:]
x=rng.standard_normal((4096,10))
def logits(wi,A,O,x):
    h=np.zeros((len(wi),len(x),16))
    for t in range(10): h=np.maximum(np.einsum('nbj,nij->nbi',h,A)+x[None,:,t,None]*wi[:,None,:],0)
    return np.einsum('nbj,nij->nbi',h,O)
L1,L2=logits(Wih[ix],Whh[ix],Wo[ix],x),logits(wi2,A2,O2,x)
print(f"gauge check: max |2 L - L'| / max |L| = {np.abs(2*L1-L2).max()/np.abs(L1).max():.1e}; argmax agreement {(L1.argmax(-1)==L2.argmax(-1)).mean():.6f}")
print(f"  raw weight statistics change by RMS {np.sqrt(((stats(wi2,A2,O2)-W[ix])**2).mean()):.3f} (population SD of a statistic, median {np.median(W[:1024].std(0)):.3f}); "
      f"canonical ones by {np.abs(stats(*canonical(wi2,A2,O2))-Wc[ix]).max():.1e}")
