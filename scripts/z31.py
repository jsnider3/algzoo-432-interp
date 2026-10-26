"""An adaptive outer rule for the MF4 anchor integral: does refining where the integrand is rough beat z21's tensor rules?
Same integrand as z21 (CDF coordinates a = Phi(Q), b, c of z21's Beta(7,4), Beta(1,3), Beta(2,1) laws; S integrated along
each line over all 5,040 arrangements, K points), written as G(a,b,c) times the three Beta densities on the unit cube.
Each box carries a 2 x 2 x 2 Gauss-Legendre estimate and a centre-point estimate; the box whose two estimates differ most
(mean over networks) is split into 8, until the node budget is spent.  The design is common to all networks (driven by
their mean), so it is deterministic and weight-dependent.  Cost is charged at about 9,300 forwards per node, as in z21.
Every 64th population network + the 6 zoo models.  Usage: python scripts/z31.py [node budget, default 800] [K, default 256] [log2 reference, default 26]
"""
import sys, itertools, heapq, numpy as np, torch as th
from scipy.stats import beta as B
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; budget=int(sys.argv[1]) if len(sys.argv)>1 else 800; K=int(sys.argv[2]) if len(sys.argv)>2 else 256; lref=int(sys.argv[3]) if len(sys.argv)>3 else 26
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev)
nd=th.distributions.Normal(0.,1.); dens=lambda a,b,c: B(7,4).pdf(a)*B(1,3).pdf(b)*B(2,1).pdf(c)
cache={}
@th.no_grad()
def G(a,b,g):
    key=(round(a,13),round(b,13),round(g,13))
    if key in cache: return cache[key]
    Q=float(nd.icdf(th.tensor(a))); pr=a+(1-a)*b; R=float(nd.icdf(th.tensor(pr))); M=float(nd.icdf(th.tensor(pr+(1-pr)*g)))
    fR,fM=float(th.special.ndtr(th.tensor(R))),float(th.special.ndtr(th.tensor(M)))
    S=th.special.ndtri(th.tensor(fR+(fM-fR)*(np.arange(K)+.5)/K,dtype=th.float32,device=dev))
    mu=float(mu_below(th.tensor([[Q]]))[0,0]); tot=th.zeros(N,device=dev); A=len(ARR); ch=1260
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; m=len(arr); X=th.full((m,K,10),mu,device=dev)
        for col,v in ((0,M),(2,R),(3,Q)): X.scatter_(-1,arr[:,None,col:col+1].expand(m,K,1),th.full((m,K,1),v,device=dev))
        X.scatter_(-1,arr[:,None,1:2].expand(m,K,1),S[None,:,None].expand(m,K,1))
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,m,K,10)
        tot+=(L.argmax(-1)==arr[None,:,None,1]).float().sum((1,2))
    cache[key]=(tot/(A*K)).cpu().numpy(); return cache[key]
gx=(np.array([-1,1])/np.sqrt(3)+1)/2
def box(lo,hi):
    """2x2x2 Gauss-Legendre and centre estimates of the integral over the box"""
    vol=np.prod(np.array(hi)-np.array(lo)); q2=np.zeros(N)
    for p in itertools.product(*[lo[d]+(hi[d]-lo[d])*gx for d in range(3)]): q2+=G(*p)*dens(*p)*vol/8
    c=[(lo[d]+hi[d])/2 for d in range(3)]; q1=G(*c)*dens(*c)*vol
    return q2,np.abs(q2-q1).mean()
with th.no_grad():
    acc=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**lref; g=th.Generator().manual_seed(71)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=g).to(dev); y=target(x); o=x.argsort(-1,descending=True); v4=x.gather(-1,o[:,3:4])
        base=mu_below(v4).expand(len(x),10).clone(); base.scatter_(-1,o[:,:4],x.gather(-1,o[:,:4]))
        acc+=(forward(*W,base[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    ref=(acc/n_ref).cpu().numpy(); pq=ref*(1-ref)
print(f"{N-6} population networks (every 64th) + 6 zoo; MF4 anchor accuracy by adaptive box refinement, K = {K}; reference 2^{lref}")
heap=[]; leaves={}; uid=0
for lo in itertools.product([0,.5],repeat=3):
    hi=tuple(l+.5 for l in lo); q,e=box(lo,hi); leaves[uid]=(lo,hi,q); heapq.heappush(heap,(-e,uid)); uid+=1
marks=[100,200,400,800,1600]; done=set()
while True:
    nodes=len(cache); est=sum(v[2] for v in leaves.values())
    for mk in marks:
        if nodes>=mk and mk not in done:
            done.add(mk); e=100*(est-ref); cost=9300*nodes
            print(f"  {nodes:5d} nodes ({len(leaves)} boxes, about {cost:>10,} forwards): error RMS {np.sqrt(np.mean(e[pop]**2)):.3f} points (median {np.median(e[pop]):+.3f})"
                  f"   sampling at that cost {100*np.sqrt(np.mean(pq[pop])/cost):.3f}",flush=True)
    if nodes>=budget: break
    _,k=heapq.heappop(heap); lo,hi,_=leaves.pop(k); mid=[(lo[d]+hi[d])/2 for d in range(3)]
    for corner in itertools.product([0,1],repeat=3):
        l=tuple(lo[d] if corner[d]==0 else mid[d] for d in range(3)); h=tuple(mid[d] if corner[d]==0 else hi[d] for d in range(3))
        q,e=box(l,h); leaves[uid]=(l,h,q); heapq.heappush(heap,(-e,uid)); uid+=1
