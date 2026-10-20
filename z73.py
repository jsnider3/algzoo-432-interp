"""Can a nonlinear readout break the learned rows' plateau (z69: about 0.88 points from 64 rows on)?  Same features as
z69 (k = 64 rows from the K = 8 MF4 pool, chosen by OMP inside each fold, + pulses + 20 weight statistics + predicted
role counts), with readouts whose inference is cheap on the target:
  ridge     d multiply-adds (reference);
  boosting  sklearn HistGradientBoostingRegressor (learning rate 0.05, depth <= 3, iterations by inner early
            stopping, at most 600), charged 4 operations per tree;
  mlp       one hidden layer of 32 ReLU units (sklearn MLPRegressor, L2 penalty by inner 4-fold CV over 1e-3..1, 5 seeds
            averaged), charged 5 x (32 d + 64) operations;
  nystrom   RBF kernel ridge on 128 landmark networks (gamma and penalty by inner 4-fold CV), charged 2 x 128 d + 34 x 128.
Fitted on the 1,024 (8 folds, permutation 0), applied unchanged to the 512 new networks.
Usage: python z73.py
"""
import numpy as np, torch as th
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60
from z62 import ridge
from z67 import unit_desc, fit_predict_units, net_counts
from z68 import omp, feats
K=64
def boost(Fa,ya,Fb):
    m=HistGradientBoostingRegressor(learning_rate=.05,max_depth=3,max_iter=600,early_stopping=True,validation_fraction=.15,n_iter_no_change=30,random_state=0).fit(Fa,ya)
    return m.predict(Fb),m.n_iter_
def mlp(Fa,ya,Fb):
    sc=StandardScaler().fit(Fa); a,b=sc.transform(Fa),sc.transform(Fb); ym=ya.mean(); yc=(ya-ym)*100; fi=np.arange(len(a))%4; best=None
    for al in (1e-3,1e-2,1e-1,1.):
        l=sum((((MLPRegressor(hidden_layer_sizes=(32,),alpha=al,max_iter=2000,random_state=0).fit(a[fi!=j],yc[fi!=j]).predict(a[fi==j]))-yc[fi==j])**2).sum() for j in range(4))
        if best is None or l<best[0]: best=(l,al)
    return ym+np.mean([MLPRegressor(hidden_layer_sizes=(32,),alpha=best[1],max_iter=2000,random_state=s).fit(a,yc).predict(b) for s in range(5)],0)/100,best[1]
def nystrom(Fa,ya,Fb,L=128):
    sc=StandardScaler().fit(Fa); a,b=sc.transform(Fa),sc.transform(Fb); d=a.shape[1]; ym=ya.mean(); yc=ya-ym
    cen=a[np.random.default_rng(1701).permutation(len(a))[:L]]; fi=np.arange(len(a))%4
    def fitp(A,y,B,g,lam):
        Kc=np.exp(-g/d*cdist(cen,cen,'sqeuclidean')); e,V=np.linalg.eigh(Kc); keep=e>e[-1]*1e-10; Wh=V[:,keep]/np.sqrt(e[keep])
        Pa=np.exp(-g/d*cdist(A,cen,'sqeuclidean'))@Wh; Pb=np.exp(-g/d*cdist(B,cen,'sqeuclidean'))@Wh
        return Pb@np.linalg.solve(Pa.T@Pa+lam*np.eye(Pa.shape[1]),Pa.T@y)
    grid=[(g,l) for g in (.03,.1,.3,1.) for l in (1e-3,1e-2,1e-1,1.)]
    loss=[sum(((fitp(a[fi!=j],yc[fi!=j],a[fi==j],g,l)-yc[fi==j])**2).sum() for j in range(4)) for g,l in grid]; g,l=grid[int(np.argmin(loss))]
    return ym+fitp(a,yc,b,g,l),(g,l)
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    R=np.load('pop_unif/roles_b128_n1024_n512.npz')['roles']; D=unit_desc(*Wnp)
    _,_,_,c=gfeat.summary(W,1,8,cells=True); M=c[2].astype(float); B=c[0].astype(float)
    ROWS=gfeat.rows(1,8).reshape(720,8,2,10)[:,:,0].reshape(-1,10)
    def lev_cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel]); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6+1.8
    oof=np.zeros((1024,16,8))
    for f in range(8): A,Bf=tr[fold!=f],tr[fold==f]; oof[Bf]=fit_predict_units(D,R,A,Bf,'logistic')
    CO=net_counts(oof); Cnew=net_counts(fit_predict_units(D,R,tr,te,'logistic'))
    names=('ridge','boosting','mlp','nystrom'); cf={n:np.zeros(1024) for n in names}; rows_fe=[]; info={n:[] for n in names}
    for f in range(8):
        A,Bf=tr[fold!=f],tr[fold==f]; inner=np.zeros((len(A),16,8)); fi=np.arange(len(A))%8
        for g in range(8): inner[fi==g]=fit_predict_units(D,R,A[fi!=g],A[fi==g],'logistic')
        roles=np.zeros((len(y),10)); roles[A]=net_counts(inner); roles[Bf]=CO[Bf]; base=np.c_[P,ST,roles]
        sel=omp(M[A],y[A],base[A],K); F=feats(M,B,sel,base); rows_fe.append(lev_cost(sel))
        cf['ridge'][Bf]=ridge(F[A],y[A],F[Bf])[0]
        cf['boosting'][Bf],it=boost(F[A],y[A],F[Bf]); info['boosting'].append(it)
        cf['mlp'][Bf],al=mlp(F[A],y[A],F[Bf]); info['mlp'].append(al)
        cf['nystrom'][Bf],gl=nystrom(F[A],y[A],F[Bf]); info['nystrom'].append(gl)
        print(f"  fold {f} done",flush=True)
    roles=np.zeros((len(y),10)); roles[tr]=CO; roles[te]=Cnew; base=np.c_[P,ST,roles]; sel=omp(M[tr],y[tr],base[tr],K); F=feats(M,B,sel,base)
    d=F.shape[1]; rfe=max(max(rows_fe),lev_cost(sel))
    pr={'ridge':ridge(F[tr],y[tr],F[te])[0]}; pr['boosting'],it=boost(F[tr],y[tr],F[te]); pr['mlp'],_=mlp(F[tr],y[tr],F[te]); pr['nystrom'],_=nystrom(F[tr],y[tr],F[te])
    ro={'ridge':d,'boosting':4*max(info['boosting']+[it]),'mlp':5*(32*d+64),'nystrom':2*128*d+34*128}
    print(f"\n{K} learned rows + pulses + weight statistics + roles ({d} features); rows, pulses and roles {rfe:.1f} FE")
    for n in names:
        tot=rfe+ro[n]/2880; r=np.sqrt(((cf[n]-y[tr])**2).mean()); rn=np.sqrt(((pr[n][:-1]-y[new])**2).mean())
        print(f"  {n:9s} {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.3f}x) | new {100*rn:.3f} ({rn**2/(pqn/tot):.3f}x) | mystery {100*(pr[n][-1]-y[mys]):+.2f} | {info[n][:8] if info[n] else ''}",flush=True)
