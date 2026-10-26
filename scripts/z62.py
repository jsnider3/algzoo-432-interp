"""What has the readout learned?  A readable estimator from a few named, mechanistic quantities.
Each network is summarized by named quantities computed from MF4 probe rows at K = 2 S slices (slice 0: S in the
lower half of its range, near R; slice 1: the upper half, near M) and the two signed pulses of z60:
  bit[s], soft[s], |a|[s], fragile[s]   per slice: share of MF4 rows answered right, mean a/(.1+|a|), mean |a| and
                                        share with |a| < 0.1, where a = margin / s and s = RMS margin over the rows
  <group> bit[s], <group> soft[s]       the same bit and soft margin over z60's six arrangement groups: M last,
                                        S last, M right after S, M after S, R after S, |M - S| >= 5
  pulse+/- margin, wins, late margin, growth, growth SD   z60's pulse features (+1 and -1 pulses)
  log s, mean margin                    the logit scale
and, separately, the 20 raw weight statistics (gfeat.stats), which carry the training recipe's fingerprint.
Designs: 'all' = the 720 arrangements (1,440 MF4 rows), 'compact' = z60's 180 (360 rows).
Models, all fitted on the 1,024 training networks (pop_unif/b128_n1024_seed4) and cross-fitted with 8 folds (random
permutation 0); every choice, including which features enter, is made inside each training fold:
  kernel    gfeat's nested ridge / RBF readout on all named quantities (the reference);
  linear    ridge on all named quantities;
  forward   greedy forward selection of k = 1..12 quantities by inner leave-one-out ridge error, then ridge;
  additive  the forward-selected quantities, each through a hinge basis (knots at its 25/50/75% training quantiles).
The final models are refitted on all 1,024 and applied unchanged to the 512 new networks of z61 (labels from the
independent seed-61 bank) and to the mystery model.  Cost: z60.total_cost (the RBF readout over 1,024 supports for
the kernel; no supports for the linear and additive readouts, whose evaluation is a few operations per feature).
Usage: python scripts/z62.py
"""
import numpy as np, torch as th
from sklearn.linear_model import RidgeCV
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60
ALP=np.logspace(-4,3,15); GN=['M last','S last','M right after S','M after S','R after S','|M-S|>=5']
PN=[f'pulse{s} {q}' for q in ('margin','wins','late margin','growth','growth SD') for s in '+-']
def named(c,sub,P):
    i4=c['I4'].reshape(len(P),720,2)[:,sub].astype(float); m4=c['m4'].reshape(len(P),720,2)[:,sub].astype(float)
    s=np.sqrt((m4**2).mean((1,2))); a=m4/s[:,None,None]; soft=a/(.1+abs(a)); F={}; nm=['S near R','S near M']
    for k in range(2):
        F[f'bit[{nm[k]}]']=i4[:,:,k].mean(1); F[f'soft[{nm[k]}]']=soft[:,:,k].mean(1)
        F[f'|a|[{nm[k]}]']=abs(a[:,:,k]).mean(1); F[f'fragile[{nm[k]}]']=(abs(a[:,:,k])<.1).mean(1)
        for g,msk in zip(GN,z60.MASKS):
            F[f'{g} bit[{nm[k]}]']=i4[:,msk[sub],k].mean(1); F[f'{g} soft[{nm[k]}]']=soft[:,msk[sub],k].mean(1)
    for j,n in enumerate(PN): F[n]=P[:,j]
    F['log s']=np.log(s); F['mean margin']=m4.mean((1,2))
    return np.stack(list(F.values()),1),list(F)

def ridge(Xa,ya,Xb):
    m,sd=Xa.mean(0),np.maximum(Xa.std(0),1e-9); r=RidgeCV(alphas=ALP*len(ya)).fit((Xa-m)/sd,ya); return r.predict((Xb-m)/sd),r
def loo(Xa,ya):
    m,sd=Xa.mean(0),np.maximum(Xa.std(0),1e-9); return RidgeCV(alphas=ALP*len(ya)).fit((Xa-m)/sd,ya).best_score_
def forward(X,y,kmax=12):
    sel=[]
    for _ in range(kmax):
        sc=[loo(X[:,sel+[j]],y) if j not in sel else -np.inf for j in range(X.shape[1])]; sel.append(int(np.argmax(sc)))
    return sel
def hinge(Xa,Xb):
    q=np.percentile(Xa,[25,50,75],0); f=lambda Z: np.concatenate([Z]+[np.maximum(Z-q[i],0) for i in range(3)],1); return f(Xa),f(Xb)

if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    _,_,_,c=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),c)); P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    fold=np.random.default_rng(0).permutation(1024)%8
    def score(tag,tot,cf,pn,pm):
        r=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((pn-y[new])**2).mean())
        print(f"  {tag:44s} {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.2f}x) | new {100*rn:.3f} mean {100*(pn-y[new]).mean():+.3f} ({rn**2/(pqn/tot):.2f}x) | mystery {100*(pm-y[mys]):+.2f}",flush=True)
    for design,sub in (('all',np.ones(720,bool)),('compact',z60.ENDS)):
        X,names=named(cells,sub,P); d=X.shape[1]
        print(f"\n{design} design: {d} named quantities",flush=True)
        for wtag,XX in (('named',X),('named + 20 weight statistics',np.c_[X,ST])):
            tot=z60.total_cost(2,'4',sub,1,1,XX.shape[1])
            cf=gfeat.crossfit(XX[tr],y[tr],tr,fold); pr=gfeat.fit_predict(XX,y,tr,np.r_[new,mys]); score(f'kernel, {wtag}',tot,cf,pr[:-1],pr[-1])
            cf=np.zeros(1024)
            for f in range(8): a,b=tr[fold!=f],tr[fold==f]; cf[b]=ridge(XX[a],y[a],XX[b])[0]
            pr=ridge(XX[tr],y[tr],XX[np.r_[new,mys]])[0]; score(f'linear, {wtag}',z60.total_cost(2,'4',sub,1,1,XX.shape[1],0),cf,pr[:-1],pr[-1])
        cfk={k:np.zeros(1024) for k in range(1,13)}; cfa={k:np.zeros(1024) for k in range(1,13)}
        for f in range(8):
            a,b=tr[fold!=f],tr[fold==f]; sel=forward(X[a],y[a])
            for k in range(1,13):
                cfk[k][b]=ridge(X[a][:,sel[:k]],y[a],X[b][:,sel[:k]])[0]; ha,hb=hinge(X[a][:,sel[:k]],X[b][:,sel[:k]]); cfa[k][b]=ridge(ha,y[a],hb)[0]
        sel=forward(X[tr],y[tr]); print(f"  forward selection on all 1,024: "+"; ".join(f"{k+1}. {names[j]}" for k,j in enumerate(sel)),flush=True)
        for k in (1,2,3,4,6,8,12):
            tot=z60.total_cost(2,'4',sub,1,1,k,0); te=np.r_[new,mys]
            pr=ridge(X[tr][:,sel[:k]],y[tr],X[te][:,sel[:k]])[0]; score(f'forward, {k} quantities, linear',tot,cfk[k],pr[:-1],pr[-1])
            ha,hb=hinge(X[tr][:,sel[:k]],X[te][:,sel[:k]]); pr=ridge(ha,y[tr],hb)[0]; score(f'forward, {k} quantities, additive',tot,cfa[k],pr[:-1],pr[-1])
        for k in (3,6):
            p,r=ridge(X[tr][:,sel[:k]],y[tr],X[tr][:,sel[:k]])
            print(f"  {k}-quantity linear model, points of accuracy per training SD: "+", ".join(f"{names[j]} {100*b:+.2f}" for j,b in zip(sel[:k],r.coef_))
                  +f"; intercept {100*r.intercept_:.2f}%",flush=True)
