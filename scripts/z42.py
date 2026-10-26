"""Where the two-level correction lives, from information the estimator already computes (z41's rule: n2/K8/q8,
720 balanced arrangements, exact-S MF3 anchor from pop1k/seed0_z41.npz).  Per configuration the 48 lower-input rows
are needed only where F4 can differ from I4.
  margin gating: evaluate them only on the fraction f of configurations with the smallest |MF4 logit margin|
    (per network), F4 = I4 elsewhere; cost 2 + 48 f rows per configuration.
  extreme-node screen: first evaluate each lower input at its largest and smallest Gauss-Legendre node (12 rows),
    and the other 36 only where one of them flips I4; S1 = largest node only (6 rows).
Then splits F4 - I4 by which extreme node flips the anchor and whether the anchor is right: the correction is two
opposing flows.  Dev (0::16), held out (8::16), zoo.  Usage: python scripts/z42.py
"""
import itertools, numpy as np, torch as th
from batch_train import forward
from z2 import load
from seeds_clocks import load as zload
from scipy.special import ndtr, ndtri, roots_jacobi
dev='cuda' if th.cuda.is_available() else 'cpu'; n,K,q=2,8,8
ids=np.r_[np.arange(0,1024,16),np.arange(8,1024,16),np.arange(1024,1030)]
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b])[ids].to(dev) for a,b in zip(P,Wz)]; N=len(ids)
ref=np.load('pop1k/reference.npz'); z41=np.load('pop1k/seed0_z41.npz'); anc=z41['anchor6'][ids]; ancC=z41['anchor6_cost'][ids]
def rule(n,a,b):
    x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
PAIRS=[(0,7),(7,0),(1,3),(3,1),(2,5),(5,2),(4,6),(6,4)]
arr=np.array([r for r in itertools.permutations(range(10),4) if ([x for x in range(10) if x not in r[:2]].index(r[2]),[x for x in range(10) if x not in r[:2]].index(r[3])) in PAIRS])
A=len(arr); low=np.array([[i for i in range(10) if i not in r] for r in arr])
rules=[rule(n,7,4),rule(n,1,3),rule(n,2,1)]; uq,wq=rule(q,1,1); wq_t=th.tensor(wq,dtype=th.float32,device=dev)
rows=2+6*q; Wc,M4,D43,D44,SC1,SC0,A0=[],[],[],[],[],[],[]; bs=90
for ij in itertools.product(range(n),repeat=3):
    u=[rules[t][0][j] for t,j in enumerate(ij)]; wt=np.prod([rules[t][1][j] for t,j in enumerate(ij)])
    fq=u[0]; fr=fq+(1-fq)*u[1]; fm=fr+(1-fr)*u[2]; Q,R,M=ndtri(fq),ndtri(fr),ndtri(fm)
    s=ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K); zn=ndtri(fq*uq)
    for st in range(0,A,bs):
        aa=arr[st:st+bs]; ll=low[st:st+bs]; b=len(aa); bi=np.arange(b)[:,None]; ki=np.arange(K)[None,:]
        base=np.full((b,K,10),mu(Q))
        base[bi,ki,aa[:,0,None]]=M; base[bi,ki,aa[:,1,None]]=s; base[bi,ki,aa[:,2,None]]=R; base[bi,ki,aa[:,3,None]]=Q
        X=np.repeat(base[:,:,None,:],rows,axis=2)
        for j in range(6):
            for r in range(q): X[bi,ki,1+j*q+r,ll[:,j,None]]=zn[r]
        m3=np.full((b,K,10),mu(R)); m3[bi,ki,aa[:,0,None]]=M; m3[bi,ki,aa[:,1,None]]=s; m3[bi,ki,aa[:,2,None]]=R; X[:,:,-1]=m3
        with th.no_grad():
            Lg=forward(*W,th.tensor(X.reshape(-1,10),dtype=th.float32,device=dev)[None].expand(N,-1,-1)).view(N,b,K,rows,10)
            y=th.tensor(aa[:,1],device=dev)[None,:,None,None].expand(N,b,K,rows)
            I=(Lg.argmax(-1)==y).float(); L0=Lg[:,:,:,0]; yt=y[...,0:1]
            m4=L0.gather(-1,yt)[...,0]-L0.scatter(-1,yt,-1e30).max(-1).values
            i0=I[...,0]; top=I[...,1:1+6*q].reshape(N,b,K,6,q); F=(i0+((top*wq_t).sum(-1)-i0[...,None]).sum(-1)).clamp(0,1)
        f=lambda t: t.reshape(N,-1).cpu().numpy()
        Wc.append(np.full(b*K,wt/(A*K))); M4.append(f(m4)); D43.append(f(i0-I[...,-1])); D44.append(f(F-i0)); A0.append(f(i0))
        SC1.append(f((top[...,-1]!=i0[...,None]).any(-1))); SC0.append(f((top[...,0]!=i0[...,None]).any(-1)))
Wc=np.concatenate(Wc); M4,D43,D44,SC1,SC0,A0=(np.concatenate(v,1) for v in (M4,D43,D44,SC1,SC0,A0)); C=len(Wc)
acc=ref['acc'][ids]; pq=acc*(1-acc); base43=(D43*Wc).sum(1); dv,hd=slice(0,64),slice(64,128)
def score(est,cost):
    e=100*(est-acc); o=[]
    for sl in (dv,hd):
        samp2=1e4*np.mean(pq[sl]/cost[sl]); o+=[np.sqrt((e[sl]**2).mean()),e[sl].mean(),(e[sl]**2).mean()/samp2]
    return o,e[128:]
print(f"{C} configurations x {rows} rows; F4 differs from I4 on {100*(D44[:128]!=0).mean():.1f}% of them (points; MSE/sampling at each network's cost)")
print(f"  margin gating:  {'f':>5s} {'forwards':>9s} {'RMS dev':>8s} {'held':>7s} {'mean dev':>8s} {'MSE/s dev':>9s} {'held':>6s}  |F4-I4| mass kept")
order=np.argsort(np.abs(M4),axis=1)
for fr in (1.0,0.5,0.3,0.2,0.1,0.05,0.0):
    k=int(round(fr*C)); g=np.zeros_like(D44,dtype=bool)
    if k: np.put_along_axis(g,order[:,:k],True,axis=1)
    cost=ancC+C*(2+6*q*fr); o,_=score(anc+base43+(D44*g*Wc).sum(1),cost)
    kept=(np.abs(D44)*g*Wc).sum(1)/np.maximum((np.abs(D44)*Wc).sum(1),1e-12)
    print(f"                  {fr:5.2f} {cost[dv].mean():9.0f} {o[0]:8.4f} {o[3]:7.4f} {o[1]:+8.4f} {o[2]:9.1f} {o[5]:6.1f}  {100*np.median(kept[:128]):.1f}%")
for name,g,fixed in (('S1 (largest node)',SC1,6),('S2 (largest and smallest)',SC1|SC0,12)):
    fl=g.mean(1); cost=ancC+C*(2+fixed+(6*q-fixed)*fl); o,z=score(anc+base43+(D44*g*Wc).sum(1),cost)
    kept=(np.abs(D44)*g*Wc).sum(1)/np.maximum((np.abs(D44)*Wc).sum(1),1e-12)
    print(f"  screen {name:26s} flags {100*fl[:128].mean():.1f}%  forwards {cost[dv].mean():.0f}  RMS {o[0]:.4f} / {o[3]:.4f}  mean {o[1]:+.4f}  MSE/s {o[2]:.1f} / {o[5]:.1f}  mass kept {100*np.median(kept[:128]):.1f}%  zoo "+" ".join(f"{v:+.3f}" for v in z))
print("  F4 - I4 by which extreme node flips the MF4 anchor (mean over the 128 networks, points):")
for name,g in (('largest node only',SC1&~SC0),('smallest node only',SC0&~SC1),('both',SC1&SC0),('neither',~SC1&~SC0)):
    for an,mask in (('anchor right',A0==1),('anchor wrong',A0==0)):
        m=(D44*g*mask*Wc).sum(1)[:128]*100; fr=((g&mask)*Wc).sum(1)[:128]*100
        print(f"    {name:20s} {an:13s} configurations {fr.mean():6.2f}%   mass {m.mean():+.3f}")
