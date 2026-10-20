"""A blind test of the 4,096-network learned-row estimators of z79.  `--fit` fits them on the 4,096 training networks
(seeds 4, 7, 8) and saves them (pop_unif/frozen_721.pkl) before the 2,048 seed-9 networks
(pop_unif/b128_n2048_seed9.npz) have finished training; `--score` applies them unchanged to those networks, labelled
on an independent bank of 2^22 inputs (seed 80).  Estimator: z78/z79's combined pool (24,140 rows) + pulses + 20
weight statistics + role counts (z67's logistic unit classifier: out-of-fold for the training networks, the classifier
fitted on all 4,096 networks' units for the test networks); OMP; ridge; k = 64, 128, 256, 384.
Usage: python z80.py --fit | --score
"""
import os, sys, pickle, numpy as np, torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from z2 import load
from z72 import labels, fit_ridge, apply_ridge
import gfeat, z60, z64, z66
from z67 import unit_desc, net_counts
from z68 import omp
FROZEN='pop_unif/frozen_721.pkl'; TEST='pop_unif/b128_n2048_seed9'; KS=(64,128,256,384)
def pool(W):
    Wnp=[w.numpy().astype(float) for w in W]
    _,_,_,c=gfeat.summary(W,2,4,cells=True); Ms=[c[2].astype(np.float32)]; Bs=[c[0].astype(np.float32)]; rows=[gfeat.rows(2,4).reshape(-1,2,10)[:,0].astype(float)]; del c
    pul=np.concatenate([np.eye(10),-np.eye(10)]); fams=dict(z66.FAMS); fams['double pulses']=(z64.X.astype(float),z64.TGT,None); fams['single pulses']=(pul,np.r_[np.arange(10),np.arange(10)],None)
    for nm,(x,tg,_) in fams.items(): b,m,_=z66.run(Wnp,x,np.asarray(tg)); Ms.append(m.astype(np.float32)); Bs.append(b.astype(np.float32)); rows.append(x)
    return np.concatenate(Ms,1),np.concatenate(Bs,1),np.concatenate(rows),np.c_[z60.pulses(*Wnp),gfeat.stats(*Wnp)],unit_desc(*Wnp)
def lfeats(M,B,sel,base): m=M[:,sel].astype(float); s=np.sqrt((m**2).mean(1)); return np.c_[base,m/s[:,None],B[:,sel],np.log(s)]
def clf_fit(D,R):
    Xa=D.reshape(-1,16); sc=StandardScaler().fit(Xa); return sc,LogisticRegression(max_iter=2000,C=1.).fit(sc.transform(Xa),R.ravel())
def clf_counts(clf,D):
    sc,lr=clf; p=np.zeros((len(D)*16,8)); p[:,lr.classes_]=lr.predict_proba(sc.transform(D.reshape(-1,16))); return net_counts(p.reshape(len(D),16,8))
def cost(ROWS,sel):
    lev,_,nv=gfeat.trie(ROWS[sel].astype(np.float32)); n=[len(p) for p,_ in lev]
    return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6+1.8
if __name__=='__main__' and sys.argv[1:]==['--fit']:
    assert not os.path.exists(TEST+'.npz'), 'the test networks already exist; this would not be a blind test'
    Ws=[load(f'pop_unif/{f}.npz')[0] for f in ('b128_n1024_seed4','b128_n1024_seed7','b128_n2048_seed8')]; W=[th.cat([w[i].float() for w in Ws]) for i in range(3)]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n1024_seed7.acc22s72.npy'),np.load('pop_unif/b128_n2048_seed8.acc22s79.npy')]
    R=np.r_[np.load('pop_unif/roles_b128_n1024_n512.npz')['roles'][:1024],np.load('pop_unif/roles_b128_n1024_seed7.npz')['roles'],np.load('pop_unif/roles_b128_n2048_seed8.npz')['roles']]
    M,B,ROWS,PS,D=pool(W); fold=np.random.default_rng(0).permutation(len(y))%8; CO=np.zeros((len(y),10))
    for f in range(8): CO[fold==f]=clf_counts(clf_fit(D[fold!=f],R[fold!=f]),D[fold==f])
    base=np.c_[PS,CO]; sel=omp(M.astype(float),y,base,max(KS)); out={'clf':clf_fit(D,R),'models':{}}
    for k in KS: out['models'][k]=dict(sel=sel[:k],cost=cost(ROWS,sel[:k]),fit=fit_ridge(lfeats(M,B,sel[:k],base),y))
    pickle.dump(out,open(FROZEN,'wb')); print(f"saved {FROZEN}: "+", ".join(f"{k} rows ({v['cost']:.1f} FE)" for k,v in out['models'].items()))
if __name__=='__main__' and sys.argv[1:]==['--score']:
    out=pickle.load(open(FROZEN,'rb')); W,_=load(TEST+'.npz'); yt=labels(W,TEST+'.acc22s80.npy',seed=80); M,B,ROWS,PS,D=pool(W)
    ytr=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n1024_seed7.acc22s72.npy'),np.load('pop_unif/b128_n2048_seed8.acc22s79.npy')]
    mu,t2=ytr.mean(),ytr.var(); pq=(yt*(1-yt)).mean(); v=((yt-mu)**2).mean(); base=np.c_[PS,clf_counts(out['clf'],D)]
    print(f"{len(yt)} blind test networks: accuracy {100*yt.mean():.2f}% SD {100*yt.std():.2f}")
    for k,mo in out['models'].items():
        e=apply_ridge(mo['fit'],lfeats(M,B,mo['sel'],base))-yt; r=np.sqrt((e**2).mean()); C=mo['cost']; w=t2/(t2+pq/C); sh=np.sqrt(w*w*pq/C+(1-w)**2*v)
        q=np.random.default_rng(80).integers(0,len(yt),(2000,len(yt))); bq=(e[q]**2).mean(1)/((yt[q]*(1-yt[q])).mean(1)/C)
        print(f"  {k:3d} learned rows {C:6.1f} FE | RMS {100*r:.3f} mean {100*e.mean():+.3f} | MSE/sampling {r*r/(pq/C):.3f} ({np.percentile(bq,2.5):.3f}-{np.percentile(bq,97.5):.3f}) | vs shrunk sampling {r*r/sh**2:.2f}",flush=True)
