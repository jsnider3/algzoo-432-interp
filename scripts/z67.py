"""Neuron roles from the weights alone: bucket each hidden unit into z3's roles with a classifier trained on the
population (roles measured offline on 4,096 probes, z63's cache), then use a network's predicted role counts, which
cost almost nothing on the target, as estimator quantities.
Per-unit weight descriptors (unit k; the rescaling-invariant ones marked *):
  w_in, self-loop A_kk*, log incoming / outgoing recurrent norms, log readout norm, w_in * readout norm, the readout
  column's Fourier shares over the 10 output positions (DC, dominant frequency, its share)*, input-normalized
  recurrent drives r_kj = A_kj w_j / w_k* (sum of positive parts, sum of negative parts, max, min over j != k) and the
  same into k from units with positive / negative input weight, and the loop products sum_j A_kj A_jk*.
Classifiers: multinomial logistic regression (about 1 FE for 16 units) and gradient boosting (HistGradientBoosting,
100 iterations, depth <= 4; charged as 4 comparisons per tree per unit).  Nested: for each outer fold the classifier is
trained on the other folds' units, with inner out-of-fold predictions for those networks' own counts.
Reports: unit-level AUC per role, network-level correlation of predicted and true counts, and the predicted counts
(8 roles + P(late pointer) + P(event pointer)) as quantities: alone with the 20 weight statistics, added to z63's
minimal estimator, and added to the compact design's 44 named quantities + weight statistics (ridge); cross-fitted on
the 1,024 and applied unchanged to the 512 new networks.
Usage: python scripts/z67.py
"""
import numpy as np, torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from z2 import load
from z3 import ROLES
from seeds_clocks import load as zload
import gfeat, z60
from z62 import named, ridge
def unit_desc(wi,A,O):
    N=len(wi); eye=np.eye(16,dtype=bool); ro=np.linalg.norm(O,axis=1); F=np.abs(np.fft.rfft(O,axis=1))**2; sh=F/np.maximum(F.sum(1,keepdims=True),1e-12)
    r=A*wi[:,None,:]/np.where(np.abs(wi)>1e-9,wi,1e-9)[:,:,None]; r=np.where(eye[None],0,r)
    loop=np.einsum('nkj,njk->nk',A,A)-np.einsum('nkk->nk',A)**2; pos=(wi>0)[:,None,:]
    cols=[wi,np.einsum('nkk->nk',A),np.log(np.linalg.norm(A,axis=2)+1e-9),np.log(np.linalg.norm(A,axis=1)+1e-9),np.log(ro+1e-9),wi*ro,
          sh[:,0],sh[:,1:].argmax(1)+1,sh[:,1:].max(1),np.maximum(r,0).sum(2),np.minimum(r,0).sum(2),r.max(2),r.min(2),
          np.where(pos,r,0).sum(2),np.where(~pos,r,0).sum(2),loop]
    return np.stack(cols,2)                                                     # (N,16,16)
def fit_predict_units(D,Rl,trn,tst,kind):
    Xa,ya=D[trn].reshape(-1,D.shape[2]),Rl[trn].ravel(); sc=StandardScaler().fit(Xa)
    m=(LogisticRegression(max_iter=2000,C=1.) if kind=='logistic' else HistGradientBoostingClassifier(max_iter=100,max_depth=4,random_state=0)).fit(sc.transform(Xa),ya)
    p=np.zeros((len(tst)*16,len(ROLES))); p[:,m.classes_]=m.predict_proba(sc.transform(D[tst].reshape(-1,D.shape[2]))); return p.reshape(len(tst),16,-1)
def net_counts(p): return np.c_[p.sum(1),1-np.prod(1-p[:,:,3],1),1-np.prod(1-p[:,:,2],1)]

if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    z=np.load('pop_unif/roles_b128_n1024_n512.npz'); R=z['roles']; cnt=np.stack([(R==k).sum(1) for k in range(len(ROLES))],1)
    D=unit_desc(*Wnp)
    _,_,_,c=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),c)); Xn,_=named(cells,z60.ENDS,P)
    m=cells['m4'].reshape(-1,720,2)[:,z60.ENDS,0].astype(float); s=np.sqrt((m**2).mean(1)); a=m/s[:,None]; MQ=np.c_[(a/(.1+abs(a))).mean(1),np.log(s),P[:,6],P[:,8],P[:,9]]
    xr=gfeat.rows(1,2).reshape(720,2,2,10)[z60.ENDS][:,0,0]; lv,_,nv=gfeat.trie(xr); n=[len(p) for p,_ in lv]
    base_fe=(256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+22*len(xr)+5000)/2880+6; cfull=z60.total_cost(2,'4',z60.ENDS,1,1,1,0)
    def score(tag,tot,cf,pr):
        r=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean())
        print(f"  {tag:58s} {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.2f}x) | new {100*rn:.3f} ({rn**2/(pqn/tot):.2f}x) | mystery {100*(pr[-1]-y[mys]):+.2f}",flush=True)
        return cf-y[tr]
    for kind,cfe in (('logistic',(16*16*8+300)/2880+1),('boosting',(100*len(ROLES)*4*16+300)/2880+1)):
        print(f"\n{kind} unit classifier ({cfe:.1f} FE for the descriptors and the classifier)",flush=True)
        oof=np.zeros((1024,16,len(ROLES))); outer={}
        for f in range(8):
            A,B=tr[fold!=f],tr[fold==f]; oof[B]=fit_predict_units(D,R,A,B,kind)
        full=fit_predict_units(D,R,tr,te,kind)
        Y=R[tr].ravel(); PP=oof.reshape(-1,len(ROLES))
        print("  unit AUC: "+", ".join(f"{ROLES[k]} {roc_auc_score(Y==k,PP[:,k]):.3f}" for k in range(len(ROLES)) if 0<(Y==k).mean()<1),flush=True)
        CO=net_counts(oof); CF=net_counts(full)
        print("  network counts, r(predicted, true): "+", ".join(f"{ROLES[k]} {np.corrcoef(CO[:,k],cnt[tr,k])[0,1]:+.2f}" for k in range(1,len(ROLES)))
              +f"; late pointer {np.corrcoef(CO[:,8],(R[tr]==3).any(1))[0,1]:+.2f}, event pointer {np.corrcoef(CO[:,9],(R[tr]==2).any(1))[0,1]:+.2f}",flush=True)
        # nested readouts: inside each outer fold, the training networks' counts come from inner out-of-fold classifiers
        CN=np.zeros((1024,CO.shape[1])); CT={}
        for f in range(8):
            A,B=tr[fold!=f],tr[fold==f]; inner=np.zeros((len(A),16,len(ROLES))); fi=np.arange(len(A))%8
            for g in range(8): inner[fi==g]=fit_predict_units(D,R,A[fi!=g],A[fi==g],kind)
            CT[f]=net_counts(inner); CN[B]=net_counts(oof[B])
        Cnew=net_counts(full)
        for tag,G,fe in (('weights only: 20 statistics',lambda idx: ST[idx],1.8),('weights only: 20 statistics + predicted roles',None,1.8+cfe),
                         ('minimal estimator',lambda idx: MQ[idx],base_fe),('minimal estimator + predicted roles',None,base_fe+cfe),
                         ('compact 44 + weight statistics',lambda idx: np.c_[Xn,ST][idx],cfull),('compact 44 + weight statistics + predicted roles',None,cfull+cfe)):
            base={'weights only: 20 statistics + predicted roles':lambda idx: ST[idx],'minimal estimator + predicted roles':lambda idx: MQ[idx],
                  'compact 44 + weight statistics + predicted roles':lambda idx: np.c_[Xn,ST][idx]}.get(tag)
            cf=np.zeros(1024)
            for f in range(8):
                A,B=tr[fold!=f],tr[fold==f]
                FA,FB=(G(A),G(B)) if G else (np.c_[base(A),CT[f]],np.c_[base(B),CN[B]])
                cf[B]=ridge(FA,y[A],FB)[0]
            FA,FB=(G(tr),G(te)) if G else (np.c_[base(tr),CO],np.c_[base(te),Cnew])
            e=score(tag,fe,cf,ridge(FA,y[tr],FB)[0])
            if 'minimal' in tag: print(f"    residual vs true max detectors {np.corrcoef(e,cnt[tr,4])[0,1]:+.2f}, late pointer {np.corrcoef(e,(R[tr]==3).any(1))[0,1]:+.2f}")
