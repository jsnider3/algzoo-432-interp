"""Where in input space do the readable models of z62/z63 go blind?  (Offline diagnosis: sampling the training networks
is allowed for understanding; nothing here is an estimator.)
Regions: the age of the max M and of the 2nd max S (0 = last step; 0, 1, 2, 3, 4+, with the order of M and S kept
when both are 4+), crossed with whether M - S and S - R are below 0.05 |M| (near ties, z1's normalized gaps).
Each network's points of accuracy lost in each region, c_r = 100 P(region and wrong), are measured on 2^20 shared N(0,1)
inputs for the 1,024 training and 512 new networks.  For a readable model's quantities (z63's minimal two, and z62's
44 named quantities + 20 weight statistics on the compact design), each region's c_r is predicted by an 8-fold
cross-fitted ridge; the unexplained variance of c_r, its covariance with the accuracy residual and its link to the
max-detector count and the late pointer (z63's role cache) say which regions the probes cannot see.
Usage: python z65.py
"""
import numpy as np, torch as th
from z2 import load
from z3 import ROLES
from seeds_clocks import load as zload
from batch_train import forward, target
import gfeat, z60
from z62 import named, ridge
dev=gfeat.dev
def regions(x):
    o=np.argsort(-x,1); s=-np.sort(-x,1); aM=9-o[:,0]; aS=9-o[:,1]
    b=lambda a: np.minimum(a,4); ka=b(aM)*5+b(aS); both=(aM>=4)&(aS>=4); ka=np.where(both,25+(aS>aM),ka)
    tieMS=(s[:,0]-s[:,1])<.05*abs(s[:,0]); tieSR=(s[:,1]-s[:,2])<.05*abs(s[:,0])
    return ka*4+tieMS*2+tieSR
def describe(r):
    k,t=divmod(r,4); ag=lambda a: '4+' if a==4 else str(a)
    ages='M and S both 4+, S older' if k==26 else 'M and S both 4+, M older' if k==25 else f'M age {ag(k//5)}, S age {ag(k%5)}'
    return ages+(', M~S' if t&2 else '')+(', S~R' if t&1 else '')
Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]; tr=np.arange(1024)
g=th.Generator(device=dev).manual_seed(65); X=th.randn(2**20,10,generator=g,device=dev); lab=regions(X.cpu().numpy()); NR=27*4
w=np.bincount(lab,minlength=NR)/len(lab); L=th.tensor(lab,device=dev); C=np.zeros((len(y),NR))
with th.no_grad():
    for c0 in range(0,len(y),128):
        P=[v[c0:c0+128].to(dev) for v in W]; n=len(P[0]); E=th.zeros(n,NR,device=dev)
        for i in range(0,len(X),2**15):
            xc=X[i:i+2**15]; e=(forward(*P,xc[None].expand(n,-1,-1)).argmax(-1)!=target(xc)[None]).float()
            E+=e@th.nn.functional.one_hot(L[i:i+2**15],NR).float()
        C[c0:c0+n]=100*E.cpu().numpy()/len(X)
print(f"check: accuracy from the regions vs labels, RMS difference {np.sqrt(((100-C.sum(1))-100*y)[:1536].__pow__(2).mean()):.3f} points (sampling noise about 0.03)")
_,_,_,cl=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),cl)); Pz=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
m=cells['m4'].reshape(-1,720,2)[:,z60.ENDS,0].astype(float); s=np.sqrt((m**2).mean(1)); a=m/s[:,None]
MIN=np.c_[(a/(.1+abs(a))).mean(1),np.log(s)]; Xn,_=named(cells,z60.ENDS,Pz); FULL=np.c_[Xn,ST]
z=np.load('pop_unif/roles_b128_n1024_n512.npz'); R=z['roles']; mdet=(R==4).sum(1); late=(R==3).any(1).astype(float)
fold=np.random.default_rng(0).permutation(1024)%8; ct=C[tr]; tot=ct.var(0)
print(f"regions: {NR} defined, {int((w>0).sum())} occupied; accuracy lost per network {ct.sum(1).mean():.2f} points on average")
for tag,F in (('minimal: S-vs-R separation + log s',MIN),('compact: 44 named + 20 weight statistics',FULL)):
    pa=np.zeros(1024); pr=np.zeros((1024,NR))
    for f in range(8):
        A,B=tr[fold!=f],tr[fold==f]; pa[B]=ridge(F[A],y[A],F[B])[0]; pr[B]=np.stack([ridge(F[A],ct[A,r],F[B])[0] for r in range(NR)],1)
    res=100*(pa-y[tr]); U=ct-pr; print(f"\n{tag}: accuracy residual RMS {np.sqrt((res**2).mean()):.3f} points")
    cov=np.array([np.cov(U[:,r],res)[0,1] for r in range(NR)])   # res = -(sum of region misses) + noise, so the misses add up to the residual
    print(f"  sum over regions of cov(unexplained loss, residual) / var(residual) = {cov.sum()/res.var():+.2f} (about +1: the residual is the sum of the regions' misses)")
    o=np.argsort(-cov)[:12]
    print("  regions carrying the residual (share of residual variance; region weight; mean loss; unexplained / total SD of the loss; correlation of the loss with max detectors, late pointer):")
    for r in o:
        cm=np.corrcoef(ct[:,r],mdet[tr])[0,1]; cl=np.corrcoef(ct[:,r],late[tr])[0,1]
        print(f"    {100*cov[r]/res.var():5.1f}%  {describe(r):38s} weight {100*w[r]:5.2f}%  loss {ct[:,r].mean():.3f}  {np.sqrt(U[:,r].var()):.3f} / {np.sqrt(tot[r]):.3f}  max-det {cm:+.2f}  late {cl:+.2f}")
    fam={'M last (age 0)':lambda k: k<25 and k//5==0,'S last (age 0)':lambda k: k<25 and k%5==0,'M age 1':lambda k: k<25 and k//5==1,
         'both 4+':lambda k: k>=25}
    for nm,fn in fam.items():
        sel=[r for r in range(NR) if fn(r//4)]; print(f"  family {nm:16s}: {100*cov[sel].sum()/res.var():5.1f}% of the residual variance, weight {100*w[sel].sum():.1f}%")
    for nm,bit in (('M~S',2),('S~R',1)):
        sel=[r for r in range(NR) if r%4&bit]; print(f"  near tie {nm:4s}: {100*cov[sel].sum()/res.var():5.1f}% of the residual variance, weight {100*w[sel].sum():.1f}%")
