"""Is the mystery model's large logit scale explained by its strategy?  In the 1,024-network population (2^30 snapshot),
logit scale (RMS of the MF4-anchor margin over z46's 1,440 one-box K = 1 probe rows, as in z50) by strategy, with z3's
neuron roles (pop1k/seed0.buckets.npz): late pointer (a one-step-late write pointer, like the mystery model's n5),
event pointer (seed 2's n10), or no pointer (memory).  The zoo models get roles the same way (z2.neuron_features on
z2's 4,096 probes, then z3.roles).  Then: the scale expected from accuracy within each group, and where the mystery
model sits among the population networks of its own strategy and similar accuracy.
Usage: python z51.py
"""
import itertools, numpy as np, torch as th
from batch_train import forward
from z2 import load, neuron_features
from z3 import roles, ROLES
from seeds_clocks import load as zload
from scipy.special import ndtr, ndtri, betaincinv
dev='cuda' if th.cuda.is_available() else 'cpu'
P,s=load('pop1k/seed0.npz'); assert int(np.load('pop1k/seed0.roles.npz')['snapshot'])==s
Zw=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]; Wz=[th.tensor(np.stack([w[i] for w in Zw]),dtype=th.float32) for i in range(3)]
names=['mystery','seed0','seed1','seed2','seed3','seed4']; acc=np.load('pop1k/reference.npz')['acc']
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
ARR=np.array(list(itertools.permutations(range(10),3))); QOF={0:7,7:0,1:3,3:1,2:5,5:2,4:6,6:4}
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; QREP=np.array([f[QOF[f.index(r[2])]] for f,r in zip(free,ARR)])
cen=[betaincinv(*ab,0.5) for ab in [(7,4),(1,3),(2,1)]]; fq=cen[0]; fr=fq+(1-fq)*cen[1]; fm=fr+(1-fr)*cen[2]; Q,R,M=ndtri([fq,fr,fm]); S_=ndtri(fr+(fm-fr)*.5)
X=np.full((720,10),mu(Q)); i_=np.arange(720); X[i_,ARR[:,0]]=M; X[i_,ARR[:,1]]=S_; X[i_,ARR[:,2]]=R; X[i_,QREP]=Q
X=th.tensor(X,dtype=th.float32,device=dev); Y=th.tensor(ARR[:,1],device=dev)
def scale(W):
    W=[w.to(dev) for w in W]
    with th.no_grad():
        lg=forward(*W,X[None].expand(len(W[0]),-1,-1)); y=Y[None,:,None].expand(len(W[0]),-1,1)
        m=lg.gather(-1,y)[...,0]-lg.scatter(-1,y,-1e30).max(-1).values
    return m.pow(2).mean(1).sqrt().cpu().numpy()
sp,sz=scale(P),scale(Wz); ap=acc[:1024]; az=acc[1024:]
rp=np.load('pop1k/seed0.buckets.npz')['roles']
g=th.Generator().manual_seed(11); xp=th.randn(4096,10,generator=g); Fz=neuron_features(Wz,xp); nm=list(np.load('pop1k/seed0.roles.npz')['names'])
rz=roles(Fz,nm)
def strat(r): return np.where((r==3).any(1),'late pointer',np.where((r==2).any(1),'event pointer','no pointer'))
gp,gz=strat(rp),strat(rz)
print("zoo roles: "+"; ".join(f"{n}: {g_} ({', '.join(f'n{j} {ROLES[k]}' for j,k in enumerate(r) if k in (1,2,3))})" for n,g_,r in zip(names,gz,rz)))
print(f"\npopulation (2^30), logit scale and accuracy by strategy:")
print(f"  {'strategy':14s} {'n':>5s}  {'scale median (5-95%)':24s} {'accuracy median':>15s}  {'log-scale ~ log-acc slope':>25s}  scale expected at 95.30%")
fits={}
for k in ('late pointer','event pointer','no pointer'):
    m=gp==k; f=np.polyfit(np.log(ap[m]),np.log(sp[m]),1); fits[k]=f
    print(f"  {k:14s} {m.sum():5d}  {np.median(sp[m]):6.2f} ({np.percentile(sp[m],5):5.2f}-{np.percentile(sp[m],95):5.2f})      {100*np.median(ap[m]):10.2f}%  {f[0]:25.1f}  {np.exp(np.polyval(f,np.log(az[0]))):.2f}")
print("\nzoo models against their own strategy group: scale, percentile in group, scale expected from accuracy in group, ratio")
for n,g_,s_,a_ in zip(names,gz,sz,az):
    m=gp==g_; e=np.exp(np.polyval(fits[g_],np.log(a_))); print(f"  {n:8s} {g_:14s} acc {100*a_:.2f}%  scale {s_:6.2f}  {100*(sp[m]<s_).mean():5.1f}th pct  expected {e:5.2f}  ratio {s_/e:.2f}")
m=(gp=='late pointer')&(np.abs(ap-az[0])<0.01)
print(f"\nlate-pointer networks within 1 point of the mystery's accuracy: {m.sum()}; their scales median {np.median(sp[m]):.2f} (max {sp[m].max():.2f}); "
      f"mystery {sz[0]:.2f} exceeds {100*(sp[m]<sz[0]).mean():.0f}% of them")
r=np.log(sp)-np.polyval(np.polyfit(np.log(ap),np.log(sp),1),np.log(ap))
print(f"population residual SD of log scale given accuracy {r.std():.3f}; mystery's residual {np.log(sz[0])-np.polyval(np.polyfit(np.log(ap),np.log(sp),1),np.log(az[0])):.3f} "
      f"= {(np.log(sz[0])-np.polyval(np.polyfit(np.log(ap),np.log(sp),1),np.log(az[0])))/r.std():.1f} SD; networks with a larger residual: {(r>np.log(sz[0])-np.polyval(np.polyfit(np.log(ap),np.log(sp),1),np.log(az[0]))).sum()}")
