"""How fast does a deterministic outer quadrature converge for the top-4 (MF4) integral?
The MF4 accuracy is E[I0] over the ordered top-4 values and their 5,040 arrangements, the 6 lower inputs at mu(Q).
In CDF coordinates (as in ChatGPT's round-4 integrator): Phi(Q) ~ Beta(7,4); (Phi(R)-Phi(Q))/(1-Phi(Q)) ~ Beta(1,3);
(Phi(M)-Phi(R))/(1-Phi(R)) ~ Beta(2,1); Phi(S) uniform between Phi(R) and Phi(M).  So
  E[I0] = E_outer[ G ],   G(Q, R, M) = mean over the 5,040 arrangements of the S-integral of I0
Here G is computed by brute force (all arrangements, K equal-mass S points), and the outer expectation by tensor
Gauss-Jacobi rules of order n = 1..nmax per coordinate (n^3 outer nodes).  The reference is E[I0] at 2^22 sampled
top-4 configurations (a measurement only).  An exact S integral of the whole arrangement tree costs about 9,300
forwards per outer node (ChatGPT's optimized MF4 tree), so order n costs about 9,300 n^3: the error at each order is
set against sampling's at that cost.
Every 64th population network + the 6 zoo models.  Usage: python scripts/z21.py [nmax, default 6] [K, default 128] [nmin, default 1] [log2 reference size, default 22]
"""
import sys, itertools, numpy as np, torch as th
from scipy.special import roots_jacobi
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; nmax=int(sys.argv[1]) if len(sys.argv)>1 else 6; K=int(sys.argv[2]) if len(sys.argv)>2 else 128; nmin=int(sys.argv[3]) if len(sys.argv)>3 else 1; lref=int(sys.argv[4]) if len(sys.argv)>4 else 22
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev)                   # (5040, 4): positions of M, S, R, Q
nd=th.distributions.Normal(0.,1.)
def ndtri(u): return th.special.ndtri(u)
def beta_rule(n,a,b):
    x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()

@th.no_grad()
def G(Q,R,M):
    """mean over arrangements and K equal-mass S points of I0, for all networks: (N,)"""
    fR,fM=float(th.special.ndtr(th.tensor(R))),float(th.special.ndtr(th.tensor(M)))
    S=ndtri(th.tensor(fR+(fM-fR)*(np.arange(K)+.5)/K,dtype=th.float32,device=dev))          # (K,)
    mu=float(mu_below(th.tensor([[Q]]))[0,0]); tot=th.zeros(N,device=dev); A=len(ARR); ch=1260
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; a=len(arr)
        X=th.full((a,K,10),mu,device=dev)
        X.scatter_(-1,arr[:,None,0:1].expand(a,K,1),th.full((a,K,1),M,device=dev))
        X.scatter_(-1,arr[:,None,1:2].expand(a,K,1),S[None,:,None].expand(a,K,1))
        X.scatter_(-1,arr[:,None,2:3].expand(a,K,1),th.full((a,K,1),R,device=dev))
        X.scatter_(-1,arr[:,None,3:4].expand(a,K,1),th.full((a,K,1),Q,device=dev))
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,a,K,10)
        tot+=(L.argmax(-1)==arr[None,:,None,1]).float().sum((1,2))
    return tot/(A*K)

with th.no_grad():                                                                   # reference E[I0] by sampling
    acc=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**lref; g=th.Generator().manual_seed(71)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=g).to(dev); y=target(x); o=x.argsort(-1,descending=True); v4=x.gather(-1,o[:,3:4])
        base=mu_below(v4).expand(len(x),10).clone(); base.scatter_(-1,o[:,:4],x.gather(-1,o[:,:4]))
        acc+=(forward(*W,base[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    ref=(acc/n_ref).cpu().numpy(); se=np.sqrt(ref*(1-ref)/n_ref)
print(f"{N-6} population networks (every 64th) + 6 zoo; MF4 accuracy by tensor Gauss-Jacobi outer rules, K = {K} S points per line")
print(f"  reference E[I0] at 2^{lref} sampled configurations: pop median {np.median(ref[pop]):.4f}; SE about {100*np.median(se):.3f} points")
pq=ref*(1-ref)
for n in range(nmin,nmax+1):
    (uq,wq),(ur,wr),(ug,wg)=beta_rule(n,7,4),beta_rule(n,1,3),beta_rule(n,2,1); est=np.zeros(N)
    for a,wa in zip(uq,wq):
        Q=float(nd.icdf(th.tensor(a)))
        for b,wb in zip(ur,wr):
            pr=a+(1-a)*b; R=float(nd.icdf(th.tensor(pr)))
            for c,wc in zip(ug,wg):
                M=float(nd.icdf(th.tensor(pr+(1-pr)*c))); est+=wa*wb*wc*G(Q,R,M).cpu().numpy()
    e=100*(est-ref); cost=9300*n**3; samp=100*np.sqrt(np.mean(pq[pop])/cost)
    print(f"  order {n} ({n**3:3d} outer nodes, about {cost:>9,} forwards): error RMS {np.sqrt(np.mean(e[pop]**2)):.3f} points (median {np.median(e[pop]):+.3f})"
          f"   sampling at that cost {samp:.3f}   zoo "+" ".join(f"{v:+.3f}" for v in e[-6:]),flush=True)
