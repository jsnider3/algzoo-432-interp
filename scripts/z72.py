"""A blind test of 7.19-7.20's estimators.  `--fit` fits each estimator on the 1,024 training networks
(pop_unif/b128_n1024_seed4) and saves it (pop_unif/frozen_720.pkl), before the 1,024 test networks of seed 7
(pop_unif/b128_n1024_seed7.npz) have finished training; `--score` applies the saved estimators unchanged to those
networks, labelled on an independent bank of 2^22 N(0,1) inputs (seed 72), and also re-scores 7.18's compiled models.
Estimators (costs as in z63, z67, z69, z71):
  minimal 2      S-vs-R separation + log s (180 compact lower-slice MF4 rows)
  minimal 5 + roles   the same + three pulse quantities + role counts from z67's logistic unit classifier
  compact + roles     the compact design's 44 named quantities + 20 weight statistics + role counts
  learned K8 k   z69: k rows from the K = 8 MF4 pool + pulses + weight statistics + role counts (k = 16, 32, 64)
  learned mixed k     z71: k rows from the mixed pool + pulses + weight statistics (k = 16, 32, 128)
Every readout is a ridge (alpha by leave-one-out on the 1,024); role counts for the training networks are z67's
out-of-fold predictions, and for any other network the classifier fitted on all 1,024 networks' units.
Usage: python scripts/z72.py --fit | --score
"""
import sys, json, pickle, os, numpy as np, torch as th
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from z2 import load
from seeds_clocks import load as zload
from batch_train import accuracy
import gfeat, z60, z64, z66
from z62 import named, ALP
from z67 import unit_desc, net_counts
from z68 import omp
FROZEN='pop_unif/frozen_720.pkl'; TEST='pop_unif/b128_n1024_seed7'
def fit_ridge(F,y):
    m,sd=F.mean(0),np.maximum(F.std(0),1e-9); r=RidgeCV(alphas=ALP*len(y)).fit((F-m)/sd,y); return (m,sd,r)
def apply_ridge(R,F): m,sd,r=R; return r.predict((F-m)/sd)
def mixed_pool(Wnp,W):
    _,_,_,c=gfeat.summary(W,1,8,cells=True); Ms=[c[2].astype(float)]; Bs=[c[0].astype(float)]; rows=[gfeat.rows(1,8).reshape(720,8,2,10)[:,:,0].reshape(-1,10).astype(float)]
    pul=np.concatenate([np.eye(10),-np.eye(10)]); fams=dict(z66.FAMS); fams['double pulses']=(z64.X.astype(float),z64.TGT,None); fams['single pulses']=(pul,np.r_[np.arange(10),np.arange(10)],None)
    for nm,(x,tg,_) in fams.items(): b,m,_=z66.run(Wnp,x,np.asarray(tg)); Ms.append(m); Bs.append(b.astype(float)); rows.append(x)
    return np.concatenate(Ms,1),np.concatenate(Bs,1),np.concatenate(rows),c
def lfeats(M,B,sel,base): m=M[:,sel]; s=np.sqrt((m**2).mean(1)); return np.c_[base,m/s[:,None],B[:,sel],np.log(s)]
def quantities(W):
    """everything the estimators read from a set of networks (W: torch float32 weights)"""
    Wnp=[w.numpy().astype(float) for w in W]; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp); D=unit_desc(*Wnp)
    M8,B8,ROWS,c8=mixed_pool(Wnp,W)
    _,_,_,c2=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),c2)); Xn,_=named(cells,z60.ENDS,P)
    m=cells['m4'].reshape(-1,720,2)[:,z60.ENDS,0].astype(float); s=np.sqrt((m**2).mean(1)); a=m/s[:,None]
    MQ=np.c_[(a/(.1+abs(a))).mean(1),np.log(s),P[:,6],P[:,8],P[:,9]]
    return dict(P=P,ST=ST,D=D,M=M8,B=B8,ROWS=ROWS,Xn=Xn,MQ=MQ)
def trie_fe(x,extra):
    lev,_,nv=gfeat.trie(x.astype(np.float32)); n=[len(p) for p,_ in lev]
    return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1])/2880+extra
def labels(W,path,seed=72,log2=22):
    if os.path.exists(path): return np.load(path)
    g=th.Generator(device=gfeat.dev).manual_seed(seed); x=th.randn(2**log2,10,generator=g,device=gfeat.dev); P=[w.to(gfeat.dev).float() for w in W]
    a=np.concatenate([accuracy([p[i:i+256] for p in P],x) for i in range(0,len(P[0]),256)]); np.save(path,a); return a

if __name__=='__main__' and sys.argv[1:]==['--fit']:
    W,_=load('pop_unif/b128_n1024_seed4.npz'); y=np.load('pop_unif/b128_n1024_seed4.acc22.npy'); Q=quantities(W)
    R=np.load('pop_unif/roles_b128_n1024_n512.npz')['roles'][:1024]; fold=np.random.default_rng(0).permutation(1024)%8
    def unit_clf(idx):
        Xa=Q['D'][idx].reshape(-1,16); sc=StandardScaler().fit(Xa); return sc,LogisticRegression(max_iter=2000,C=1.).fit(sc.transform(Xa),R[idx].ravel())
    def counts(clf,D):
        sc,lr=clf; p=np.zeros((len(D)*16,8)); p[:,lr.classes_]=lr.predict_proba(sc.transform(D.reshape(-1,16))); return net_counts(p.reshape(len(D),16,8))
    CO=np.zeros((1024,10))
    for f in range(8): CO[fold==f]=counts(unit_clf(np.where(fold!=f)[0]),Q['D'][fold==f])
    clf=unit_clf(np.arange(1024)); out={'clf':clf,'models':{}}
    lower=gfeat.rows(1,2).reshape(720,2,2,10)[z60.ENDS][:,0,0]; base_fe=trie_fe(lower,(22*len(lower)+5000)/2880)
    out['models']['minimal 2']=dict(cost=base_fe,fit=fit_ridge(Q['MQ'][:,:2],y))
    out['models']['minimal 5 + roles']=dict(cost=base_fe+6+1.8,fit=fit_ridge(np.c_[Q['MQ'],CO],y))
    out['models']['compact + roles']=dict(cost=z60.total_cost(2,'4',z60.ENDS,1,1,1,0)+1.8,fit=fit_ridge(np.c_[Q['Xn'],Q['ST'],CO],y))
    K8=np.arange(5760); base=np.c_[Q['P'],Q['ST'],CO]; sel=omp(Q['M'][:,K8],y,base,64)
    for k in (16,32,64):
        out['models'][f'learned K8 {k}']=dict(cost=trie_fe(Q['ROWS'][sel[:k]],(30*k+5000)/2880+6+1.8),sel=sel[:k],roles=True,fit=fit_ridge(lfeats(Q['M'],Q['B'],sel[:k],base),y))
    base=np.c_[Q['P'],Q['ST']]; sel=omp(Q['M'],y,base,128)
    for k in (16,32,128):
        out['models'][f'learned mixed {k}']=dict(cost=trie_fe(Q['ROWS'][sel[:k]],(30*k+5000)/2880+6),sel=sel[:k],roles=False,fit=fit_ridge(lfeats(Q['M'],Q['B'],sel[:k],base),y))
    pickle.dump(out,open(FROZEN,'wb')); print(f"saved {FROZEN}: "+", ".join(f"{k} ({v['cost']:.1f} FE)" for k,v in out['models'].items()))

if __name__=='__main__' and sys.argv[1:]==['--score']:
    out=pickle.load(open(FROZEN,'rb')); W,_=load(TEST+'.npz'); yt=labels(W,TEST+'.acc22s72.npy'); Q=quantities(W)
    ytr=np.load('pop_unif/b128_n1024_seed4.acc22.npy'); mu,t2=ytr.mean(),ytr.var(); pq=(yt*(1-yt)).mean(); v=((yt-mu)**2).mean()
    print(f"{len(yt)} blind test networks: accuracy {100*yt.mean():.2f}% SD {100*yt.std():.2f}; mystery percentile {100*(yt<0.95304775).mean():.1f}")
    sc,lr=out['clf']; p=np.zeros((len(yt)*16,8)); p[:,lr.classes_]=lr.predict_proba(sc.transform(Q['D'].reshape(-1,16))); C=net_counts(p.reshape(len(yt),16,8))
    def line(tag,cost,pr):
        e=pr-yt; r=np.sqrt((e**2).mean()); w=t2/(t2+pq/cost); sh=np.sqrt(w*w*pq/cost+(1-w)**2*v)
        q=np.random.default_rng(72).integers(0,len(yt),(2000,len(yt))); bq=(e[q]**2).mean(1)/((yt[q]*(1-yt[q])).mean(1)/cost)
        print(f"  {tag:24s} {cost:7.1f} FE | RMS {100*r:.3f} mean {100*e.mean():+.3f} | MSE/sampling {r*r/(pq/cost):.3f} ({np.percentile(bq,2.5):.3f}-{np.percentile(bq,97.5):.3f}) | vs shrunk sampling {r*r/sh**2:.2f}",flush=True)
    for nm,mo in out['models'].items():
        if nm=='minimal 2': F=Q['MQ'][:,:2]
        elif nm=='minimal 5 + roles': F=np.c_[Q['MQ'],C]
        elif nm=='compact + roles': F=np.c_[Q['Xn'],Q['ST'],C]
        else: F=lfeats(Q['M'],Q['B'],mo['sel'],np.c_[Q['P'],Q['ST'],C] if mo['roles'] else np.c_[Q['P'],Q['ST']])
        line(nm,mo['cost'],apply_ridge(mo['fit'],F))
    sys.path.insert(0,'chatgpt2/round7'); import round7_estimator as r7; wn=[w.numpy().astype(float) for w in W]
    for nm in ('compact','mf4_k2_small','mf4_k2','mf4_k4'):
        m=dict(np.load(f'chatgpt2/round7/runs/{nm}.model.npz',allow_pickle=False)); cost=json.load(open(f'chatgpt2/round7/runs/{nm}.json'))['costs']['total_fe']
        line(f'7.18 {nm}',cost,np.array([r7.predict_weights(m,[w[i] for w in wn]) for i in range(len(yt))]))
