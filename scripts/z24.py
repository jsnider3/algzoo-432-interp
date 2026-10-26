"""Can the 5,040-arrangement enumeration be cut?  Per-arrangement MF4 anchor accuracies a_pi (the anchor integral within
one arrangement), by the order-5 tensor Gauss-Jacobi outer rule of z21 (K equal-mass S points), for every 64th
population network + the zoo.  Then:
  - how much of the variance of a_pi across arrangements the positions of M and S carry (90 classes), and with the
    order of R and Q relative to M and S added;
  - the error in the total if each (M, S)-position class is represented by a fixed deterministic subset of its 56
    (R, Q) placements, against the sampling line at the reduced cost.
Usage: python scripts/z24.py [K, default 128]
"""
import sys, itertools, numpy as np, torch as th
from scipy.special import roots_jacobi
from batch_train import forward
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; K=int(sys.argv[1]) if len(sys.argv)>1 else 128; n=5
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev); A=len(ARR)
nd=th.distributions.Normal(0.,1.)
def beta_rule(n,a,b):
    x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()
@th.no_grad()
def G(Q,R,M):
    fR,fM=float(th.special.ndtr(th.tensor(R))),float(th.special.ndtr(th.tensor(M)))
    S=th.special.ndtri(th.tensor(fR+(fM-fR)*(np.arange(K)+.5)/K,dtype=th.float32,device=dev))
    mu=float(mu_below(th.tensor([[Q]]))[0,0]); out=th.zeros(N,A,device=dev); ch=1260
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; a=len(arr); X=th.full((a,K,10),mu,device=dev)
        for c,v in ((0,M),(2,R),(3,Q)): X.scatter_(-1,arr[:,None,c:c+1].expand(a,K,1),th.full((a,K,1),v,device=dev))
        X.scatter_(-1,arr[:,None,1:2].expand(a,K,1),S[None,:,None].expand(a,K,1))
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,a,K,10)
        out[:,a0:a0+a]=(L.argmax(-1)==arr[None,:,None,1]).float().mean(-1)
    return out
(uq,wq),(ur,wr),(ug,wg)=beta_rule(n,7,4),beta_rule(n,1,3),beta_rule(n,2,1); acc=th.zeros(N,A,device=dev)
for a,wa in zip(uq,wq):
    Q=float(nd.icdf(th.tensor(a)))
    for b,wb in zip(ur,wr):
        pr=a+(1-a)*b; R=float(nd.icdf(th.tensor(pr)))
        for c,wc in zip(ug,wg):
            M=float(nd.icdf(th.tensor(pr+(1-pr)*c))); acc+=wa*wb*wc*G(Q,R,M)
a_=acc.cpu().numpy(); arr=ARR.cpu().numpy(); tot=a_.mean(1)
np.savez('pop1k/seed0_z24.npz',a=a_,arr=arr)
ms=arr[:,0]*10+arr[:,1]                                                              # (M, S) position class
rel=ms*16+(arr[:,2]>arr[:,1])*8+(arr[:,3]>arr[:,1])*4+(arr[:,2]>arr[:,0])*2+(arr[:,3]>arr[:,0])  # + order of R, Q vs S, M
def share(g):
    u,inv=np.unique(g,return_inverse=True); mean=np.stack([np.bincount(inv,weights=a_[i])/np.bincount(inv) for i in range(N)])
    return 1-((a_-mean[:,inv])**2).mean(1)/a_.var(1)
print(f"{N-6} population networks (every 64th) + 6 zoo; per-arrangement MF4 anchor accuracy, order-{n} outer rule, K = {K}")
print(f"  SD of a_pi across arrangements: pop median {100*np.median(a_[pop].std(1)):.2f} points")
for lab,g in (("(M, S) positions (90 classes)",ms),("+ order of R, Q relative to S and M",rel)):
    v=share(g); print(f"  variance share explained by {lab:40s} pop median {np.median(v[pop]):.3f}  zoo "+" ".join(f"{x:.3f}" for x in v[-6:]))
pq=tot*(1-tot)
print("  error of the total when each (M, S) class keeps only a fixed subset of its 56 (R, Q) placements (points):")
rng_pairs=[(r,q) for r in range(10) for q in range(10) if r!=q]
for lab,rule in (("R, Q at the two earliest free positions",lambda f:[(f[0],f[1])]),
                 ("R, Q at the two latest free positions",lambda f:[(f[-2],f[-1])]),
                 ("R, Q in the middle",lambda f:[(f[3],f[4])]),
                 ("4 placements: R, Q each early or late",lambda f:[(f[0],f[1]),(f[1],f[0]),(f[-2],f[-1]),(f[-1],f[-2])]),
                 ("8 placements: (f0,f7),(f7,f0),(f2,f5),(f5,f2),(f1,f3),(f3,f1),(f4,f6),(f6,f4)",lambda f:[(f[0],f[7]),(f[7],f[0]),(f[2],f[5]),(f[5],f[2]),(f[1],f[3]),(f[3],f[1]),(f[4],f[6]),(f[6],f[4])]),
                 ("14 placements: every 4th of the 56",lambda f:[p for i,p in enumerate([(r,q) for r in f for q in f if r!=q]) if i%4==0])):
    keep=np.zeros(A,bool); cnt=0
    for i,(pm,ps,pr,pq_) in enumerate(arr):
        f=[p for p in range(10) if p not in (pm,ps)]
        if (pr,pq_) in rule(f): keep[i]=True
    est=np.stack([np.bincount(ms[keep],weights=a_[i][keep],minlength=100)[np.unique(ms)]/np.bincount(ms[keep],minlength=100)[np.unique(ms)] for i in range(N)]).mean(1)
    e=100*(est-tot); cost=9300*125*keep.sum()/A
    print(f"    {lab:76s} {keep.sum():5d} arrangements: RMS {np.sqrt(np.mean(e[pop]**2)):.3f} (median {np.median(e[pop]):+.3f})"
          f"   order-5 cost {cost:>9,.0f}, sampling there {100*np.sqrt(np.mean(pq[pop])/cost):.3f}")
