"""An end-to-end deterministic accuracy estimate with the correction included: the top-3 decomposition (z37) plus a
first-order lower-input model, all by quadrature, no sampling.
  estimate = sum over outer nodes (n x n Gauss-Jacobi in the CDF coordinates of R and M) of
             mean over the 720 arrangements and K equal-mass S points of
             clip(I0 + sum_j (E_j - I0), 0, 1),  j over the 7 smaller inputs (including the 4th largest),
             E_j = input j at q Gauss-Legendre nodes of its law below R (weights), the others at mu(R)
Cost: n^2 x 720 x K x (1 + 7 q) forwards (no prefix sharing charged).  Scored against each network's true accuracy
(2^24 sampled inputs; a measurement), against sampling at the same cost.
Every 64th population network + the 6 zoo models.  Usage: python scripts/z38.py [settings n:K:q,..., default 1:4:4,2:4:4,2:8:4,3:8:8]
"""
import sys, itertools, numpy as np, torch as th
from scipy.special import roots_jacobi
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; SET=[tuple(map(int,s.split(':'))) for s in (sys.argv[1] if len(sys.argv)>1 else "1:4:4,2:4:4,2:8:4,3:8:8").split(',')]
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),3)),device=dev); A=len(ARR)
LOW=th.stack([th.tensor([p for p in range(10) if p not in a.tolist()],device=dev) for a in ARR])     # (720, 7) lower positions
def rule(n,a,b): x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()
@th.no_grad()
def node(uR,g,K,q):
    """mean over arrangements and K S points of the clamped first-order model, for all networks"""
    fR=th.tensor(uR,dtype=th.float64,device=dev); fM=fR+(1-fR)*g
    S=th.special.ndtri(fR+(fM-fR)*th.tensor((np.arange(K)+.5)/K,device=dev)).float(); R=th.special.ndtri(fR).float(); M=th.special.ndtri(fM).float()
    mu=mu_below(R.view(1,1))[0,0]; ug,wg=np.polynomial.legendre.leggauss(q)
    zn=th.special.ndtri(fR.float()*th.tensor((ug+1)/2,dtype=th.float32,device=dev)); wq=th.tensor(wg/2,dtype=th.float32,device=dev)
    tot=th.zeros(N,device=dev); ch=40
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; low=LOW[a0:a0+ch]; m=len(arr)
        base=th.full((m,K,10),float(mu),device=dev)
        base.scatter_(-1,arr[:,None,0:1].expand(m,K,1),M.expand(m,K,1)); base.scatter_(-1,arr[:,None,2:3].expand(m,K,1),R.expand(m,K,1))
        base.scatter_(-1,arr[:,None,1:2].expand(m,K,1),S[None,:,None].expand(m,K,1))
        rows=base[:,:,None].repeat(1,1,1+7*q,1)                                          # (m, K, 1+7q, 10)
        for j in range(7):
            pos=low[:,j]
            rows[:,:,1+j*q:1+(j+1)*q].scatter_(-1,pos[:,None,None,None].expand(m,K,q,1),zn[None,None,:,None].expand(m,K,q,1))
        L=forward(*W,rows.reshape(1,-1,10).expand(N,-1,-1)).view(N,m,K,1+7*q,10)
        I=(L.argmax(-1)==arr[None,:,None,None,1]).float(); I0=I[...,0]; E=(I[...,1:].view(N,m,K,7,q)*wq).sum(-1)
        Pm=(I0+(E-I0[...,None]).sum(-1)).clamp(0,1); tot+=Pm.sum((1,2))
    return (tot/(A*K)).cpu().numpy()
with th.no_grad():
    tru=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**24; gg=th.Generator().manual_seed(157)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=gg).to(dev); y=target(x)
        tru+=(forward(*W,x[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    truth=(tru/n_ref).cpu().numpy(); pq=truth*(1-truth)
print(f"{N-6} population networks (every 64th) + 6 zoo; deterministic estimate = MF3 outer rule x first-order lower-input model; truth 2^24 samples")
for n,K,q in SET:
    (ur,wr),(ugg,wgg)=rule(n,8,3),rule(n,2,1); est=np.zeros(N)
    for a,wa in zip(ur,wr):
        for c,wc in zip(ugg,wgg): est+=wa*wc*node(float(a),float(c),K,q)
    e=100*(est-truth); cost=n*n*A*K*(1+7*q)
    print(f"  n = {n}, K = {K}, q = {q}: about {cost:>11,} forwards; error RMS {np.sqrt(np.mean(e[pop]**2)):.3f} points (mean {e[pop].mean():+.3f});"
          f" sampling at that cost {100*np.sqrt(np.mean(pq[pop])/cost):.3f}   zoo "+" ".join(f"{v:+.2f}" for v in e[-6:]),flush=True)
