"""Sparse grids for the MF4 anchor integral: does a Smolyak rule beat z21's tensor rules?
Same integrand as z21 (CDF coordinates of Q, R, M; S integrated along each line over all 5,040 arrangements, K points),
same Gauss-Jacobi 1-D rules (order i in each coordinate), combined by the Smolyak combination technique:
  A(L) = sum over L-2 <= |i| <= L of (-1)^(L-|i|) C(2, L-|i|) Q_i1 x Q_i2 x Q_i3        (i_k >= 1, d = 3)
with each distinct outer node evaluated once.  Cost is charged at about 9,300 forwards per distinct node, as in z21.
Every 64th population network + the 6 zoo models.  Usage: python z28.py [Lmax, default 10] [K, default 256] [log2 reference, default 26]
"""
import sys, itertools, numpy as np, torch as th
from math import comb
from scipy.special import roots_jacobi
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; Lmax=int(sys.argv[1]) if len(sys.argv)>1 else 10; K=int(sys.argv[2]) if len(sys.argv)>2 else 256; lref=int(sys.argv[3]) if len(sys.argv)>3 else 26
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev)
nd=th.distributions.Normal(0.,1.)
def rule(n,a,b):
    x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()
LAWS=[(7,4),(1,3),(2,1)]
cache={}
@th.no_grad()
def G(c):
    if c in cache: return cache[c]
    a,b,g=c; Q=float(nd.icdf(th.tensor(a))); pr=a+(1-a)*b; R=float(nd.icdf(th.tensor(pr))); M=float(nd.icdf(th.tensor(pr+(1-pr)*g)))
    fR,fM=float(th.special.ndtr(th.tensor(R))),float(th.special.ndtr(th.tensor(M)))
    S=th.special.ndtri(th.tensor(fR+(fM-fR)*(np.arange(K)+.5)/K,dtype=th.float32,device=dev))
    mu=float(mu_below(th.tensor([[Q]]))[0,0]); tot=th.zeros(N,device=dev); A=len(ARR); ch=1260
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; m=len(arr); X=th.full((m,K,10),mu,device=dev)
        for col,v in ((0,M),(2,R),(3,Q)): X.scatter_(-1,arr[:,None,col:col+1].expand(m,K,1),th.full((m,K,1),v,device=dev))
        X.scatter_(-1,arr[:,None,1:2].expand(m,K,1),S[None,:,None].expand(m,K,1))
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,m,K,10)
        tot+=(L.argmax(-1)==arr[None,:,None,1]).float().sum((1,2))
    cache[c]=(tot/(A*K)).cpu().numpy(); return cache[c]
with th.no_grad():
    acc=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**lref; g=th.Generator().manual_seed(71)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=g).to(dev); y=target(x); o=x.argsort(-1,descending=True); v4=x.gather(-1,o[:,3:4])
        base=mu_below(v4).expand(len(x),10).clone(); base.scatter_(-1,o[:,:4],x.gather(-1,o[:,:4]))
        acc+=(forward(*W,base[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    ref=(acc/n_ref).cpu().numpy(); pq=ref*(1-ref)
print(f"{N-6} population networks (every 64th) + 6 zoo; MF4 anchor accuracy by Smolyak sparse grids of Gauss-Jacobi rules, K = {K}; reference 2^{lref}")
rows=[]
for L in range(3,Lmax+1):
    est=np.zeros(N); used=set()
    for i in itertools.product(range(1,L-1),repeat=3):
        s=sum(i)
        if not (L-2<=s<=L): continue
        coef=(-1)**(L-s)*comb(2,L-s); rs=[rule(n,*law) for n,law in zip(i,LAWS)]
        for (a,wa),(b,wb),(c,wc) in itertools.product(zip(*rs[0]),zip(*rs[1]),zip(*rs[2])):
            key=(round(float(a),12),round(float(b),12),round(float(c),12)); used.add(key); est+=coef*wa*wb*wc*G(key)
    nodes=len(used); cost=9300*nodes; e=100*(est-ref); rms=np.sqrt(np.mean(e[pop]**2)); rows.append((nodes,rms))
    print(f"  level {L:2d} ({nodes:4d} distinct outer nodes, about {cost:>10,} forwards): error RMS {rms:.3f} points (median {np.median(e[pop]):+.3f})"
          f"   sampling at that cost {100*np.sqrt(np.mean(pq[pop])/cost):.3f}",flush=True)
r=np.array(rows)
if len(r)>=3: print(f"  error ~ nodes^{np.polyfit(np.log(r[-3:,0]),np.log(r[-3:,1]),1)[0]:.2f} over the last three levels (sampling: nodes^-0.50)")
