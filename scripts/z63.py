"""What the readable models of z62 measure, what they cost at their minimum, and where they fail.
  A. Invariant only: forward selection (as z62, inside each fold) without log s and mean margin, the two named
     quantities that change when W_o is rescaled (an exact symmetry), on both designs.
  B. The minimal readable estimator: soft[S near R], log s and the pulse quantities, computed from only the rows they
     need (the compact design's 180 MF4 rows at the lower S slice, s taken over those rows), with its exact trie cost.
  C. Neuron roles (z2.neuron_features on 4,096 N(0,1) probes, z3.roles; offline, for interpretation only, never an
     estimator input) of the 1,024 training and 512 new networks: accuracy by strategy and comparator count, the
     correlation of each selected quantity with role counts, how much of the readable model's cross-fitted residual
     the roles explain (8-fold ridge), and the worst-predicted networks.
  D. The comparator that may also use the population: sampling shrunk toward the population mean.
Usage: python scripts/z63.py   (needs pop_unif/b128_n512_seed6*.{npz,npy} from z61)
"""
import os, numpy as np, torch as th
from z2 import load, neuron_features
from z3 import roles, selectivity, ROLES
from seeds_clocks import load as zload
import gfeat, z60
from z62 import named, ridge, forward, hinge
Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
_,_,_,c=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),c)); P=z60.pulses(*Wnp)
fold=np.random.default_rng(0).permutation(1024)%8
def score(tag,tot,cf,pr):
    r=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean())
    print(f"  {tag:46s} {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.2f}x) | new {100*rn:.3f} mean {100*(pr[:-1]-y[new]).mean():+.3f} ({rn**2/(pqn/tot):.2f}x) | mystery {100*(pr[-1]-y[mys]):+.2f}",flush=True)
def fwd_models(X,names,cost,ks=(3,6,12),tag=''):
    cfk={k:np.zeros(1024) for k in ks}
    for f in range(8):
        a,b=tr[fold!=f],tr[fold==f]; s=forward(X[a],y[a],max(ks))
        for k in ks: cfk[k][b]=ridge(X[a][:,s[:k]],y[a],X[b][:,s[:k]])[0]
    s=forward(X[tr],y[tr],max(ks)); print(f"  selected{tag}: "+"; ".join(f"{i+1}. {names[j]}" for i,j in enumerate(s)),flush=True)
    for k in ks: score(f'{k} quantities{tag}, linear',cost,cfk[k],ridge(X[tr][:,s[:k]],y[tr],X[te][:,s[:k]])[0])
    return s,cfk

print("A. invariant quantities only (no log s, no mean margin)")
for design,sub in (('all',np.ones(720,bool)),('compact',z60.ENDS)):
    X,names=named(cells,sub,P); keep=[i for i,n in enumerate(names) if n not in ('log s','mean margin')]
    fwd_models(X[:,keep],[names[i] for i in keep],z60.total_cost(2,'4',sub,1,1,len(keep),0),tag=f', {design}')

print("\nB. minimal readable estimator: compact design, MF4 rows at the lower S slice only")
x=gfeat.rows(1,2).reshape(720,2,2,10)[z60.ENDS][:,0,0]; lev,_,nv=gfeat.trie(x); n=[len(p) for p,_ in lev]
rows_fe=(256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1])/2880
m=cells['m4'].reshape(-1,720,2)[:,z60.ENDS,0].astype(float); s=np.sqrt((m**2).mean(1)); a=m/s[:,None]
Q={'soft[S near R]':(a/(.1+abs(a))).mean(1),'log s':np.log(s),'pulse+ growth':P[:,6],'pulse+ growth SD':P[:,8],'pulse- growth SD':P[:,9]}
for keys,pfe in ((['soft[S near R]','log s'],0),(['soft[S near R]','log s','pulse+ growth'],3),(['soft[S near R]','log s','pulse+ growth','pulse+ growth SD'],3),
                 (['soft[S near R]','log s','pulse+ growth','pulse+ growth SD','pulse- growth SD'],6)):
    X=np.stack([Q[k] for k in keys],1); tot=rows_fe+pfe+(22*len(x)+5000+2*len(keys)+3)/2880
    cf=np.zeros(1024)
    for f in range(8): A,B=tr[fold!=f],tr[fold==f]; cf[B]=ridge(X[A],y[A],X[B])[0]
    p,r=ridge(X[tr],y[tr],X[te]); score(' + '.join(keys),tot,cf,p)
    ha,hb=hinge(X[tr],X[te]); cfa=np.zeros(1024)
    for f in range(8): A,B=tr[fold!=f],tr[fold==f]; h1,h2=hinge(X[A],X[B]); cfa[B]=ridge(h1,y[A],h2)[0]
    score('  the same, additive',tot,cfa,ridge(ha,y[tr],hb)[0])
    print("    points per training SD: "+", ".join(f"{k} {100*b:+.2f}" for k,b in zip(keys,r.coef_)),flush=True)
print(f"  ({len(x)} rows, {rows_fe:.1f} FE of rows)")

print("\nC. neuron roles (offline interpretation)")
cache='pop_unif/roles_b128_n1024_n512.npz'
if os.path.exists(cache): z=np.load(cache); R,SEL=z['roles'],z['sel']
else:
    nm=list(np.load('pop1k/seed0.roles.npz')['names']); XP=th.randn(4096,10,generator=th.Generator().manual_seed(11))
    Fn=np.concatenate([neuron_features([w[i:i+128] for w in W[:3]],XP) for i in range(0,1536,128)]); R=roles(Fn,nm); SEL=selectivity(Fn,nm)
    np.savez(cache,roles=R,sel=SEL)
cnt=np.stack([(R==k).sum(1) for k in range(len(ROLES))],1); strat=np.where((R==3).any(1),'late',np.where((R==2).any(1),'event','none'))
for g in ('late','event','none'):
    i=tr[strat[tr]==g]; print(f"  {g:5s} pointer: {len(i)} networks, accuracy {100*y[i].mean():.2f}% (SD {100*y[i].std():.2f})")
for k in range(0,8):
    i=tr[cnt[tr,6]==k]
    if len(i)>=10: print(f"  {k} comparators: {len(i)} networks, accuracy {100*y[i].mean():.2f}%")
print("  mean role counts: "+", ".join(f"{ROLES[k]} {cnt[tr,k].mean():.2f}" for k in range(len(ROLES))))
Xa,names=named(cells,np.ones(720,bool),P); D={n:Xa[:,i] for i,n in enumerate(names)}
R2=np.c_[cnt,SEL,strat=='late',strat=='event'].astype(float); RN=ROLES+['selectivity','late pointer','event pointer']
for q in ('soft[S near R]','log s','pulse+ growth','pulse+ growth SD','bit[S near M]','|a|[S near M]'):
    r=[np.corrcoef(D[q][tr],R2[tr,j])[0,1] for j in range(R2.shape[1])]; o=np.argsort(-np.abs(r))[:4]
    print(f"  {q:18s} r with accuracy {np.corrcoef(D[q][tr],y[tr])[0,1]:+.2f}; strongest role links: "+", ".join(f"{RN[j]} {r[j]:+.2f}" for j in o))
s6=forward(Xa[tr],y[tr],6); cf=np.zeros(1024)
for f in range(8): A,B=tr[fold!=f],tr[fold==f]; ss=forward(Xa[A],y[A],6); cf[B]=ridge(Xa[A][:,ss],y[A],Xa[B][:,ss])[0]
e=cf-y[tr]; ce=np.zeros(1024)
for f in range(8): A,B=tr[fold!=f],tr[fold==f]; ce[B]=ridge(R2[A],e[A],R2[B])[0]
print(f"  6-quantity model (all design): cross-fit RMS {100*np.sqrt((e**2).mean()):.3f}; after a cross-fitted ridge on the roles {100*np.sqrt(((e-ce)**2).mean()):.3f}")
r=[np.corrcoef(e,R2[tr,j])[0,1] for j in range(R2.shape[1])]; o=np.argsort(-np.abs(r))[:5]
print("  residual (estimate - truth) vs roles: "+", ".join(f"{RN[j]} {r[j]:+.2f}" for j in o))
for i in np.argsort(-abs(e))[:10]:
    print(f"    network {i:4d}: accuracy {100*y[i]:.2f}%, estimate {100*cf[i]:.2f}% ({100*e[i]:+.2f}); {strat[i]} pointer; "
          +", ".join(f"{ROLES[k]} {cnt[i,k]}" for k in range(1,len(ROLES)) if cnt[i,k]))

print("\nD. sampling that may also use the population: the sampled accuracy shrunk toward the 1,024's mean (normal prior,")
print("   weight w = var / (var + mean p(1-p) / C)), scored on the 512 new networks")
mu,t2=y[tr].mean(),y[tr].var(); s2=(y[new]*(1-y[new])).mean(); v=((y[new]-mu)**2).mean()
print(f"  prior mean {100*mu:.2f}%, SD {100*np.sqrt(t2):.2f} points; the prior mean alone beats plain sampling below {s2/v:.0f} FE")
for C in (84,87,132,207,651,1216):
    w=t2/(t2+s2/C); print(f"  {C:5d} FE: plain sampling RMS {100*np.sqrt(s2/C):.2f}, shrunk sampling RMS {100*np.sqrt(w*w*s2/C+(1-w)**2*v):.2f}")
