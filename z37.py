"""Other decompositions: keep the top k inputs exact for k = 2 or 3 instead of 4.  Fewer exact inputs mean fewer
arrangements (90, 720, 5,040 for k = 2, 3, 4), a lower-dimensional outer integral (k - 1 after the S-line), and a cheaper
exact S tree (ChatGPT round 4: about 1,528 forwards per MF3 slice, 9,265 per MF4 slice; MF2 charged pro rata to its
arrangements, 191), but a mean-field anchor (the rest at their conditional mean) further from the truth.
For each k: (a) the anchor accuracy E[I0] by tensor Gauss-Jacobi rules of order n over the outer CDF coordinates, with
S on K equal-mass points per line and all arrangements, against a 2^lref sampled reference of the same anchor; (b) the
anchor's own bias against the network's true accuracy (2^lref sampled inputs).  Compare z21 for k = 4.
  k = 2: Phi(M) ~ Beta(10,1); given M, Phi(S)/Phi(M) ~ Beta(9,1) (the S-line); 8 lower inputs at mu(S)
  k = 3: Phi(R) ~ Beta(8,3); (Phi(M)-Phi(R))/(1-Phi(R)) ~ Beta(2,1); Phi(S) uniform between; 7 lower inputs at mu(R)
Every 64th population network + the 6 zoo models.  Usage: python z37.py [k, default 3] [nmax, default 10] [K, default 256] [log2 ref, default 24]
"""
import sys, itertools, numpy as np, torch as th
from scipy.special import roots_jacobi
from scipy.stats import beta as Bd
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; k=int(sys.argv[1]) if len(sys.argv)>1 else 3; nmax=int(sys.argv[2]) if len(sys.argv)>2 else 10
K=int(sys.argv[3]) if len(sys.argv)>3 else 256; lref=int(sys.argv[4]) if len(sys.argv)>4 else 24
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),k)),device=dev); A=len(ARR); cost_node={2:191,3:1528}[k]
nd=th.special.ndtri
def rule(n,a,b): x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()
@th.no_grad()
def G(vals_fn):
    """mean over arrangements and K line points of I0; vals_fn() -> (K, k) values M, S, (R), and the lower value (K,)"""
    V,low=vals_fn(); tot=th.zeros(N,device=dev); ch=max(1,5040//k)
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; m=len(arr); X=low[None,:,None].expand(m,K,10).clone()
        for c in range(k): X.scatter_(-1,arr[:,None,c:c+1].expand(m,K,1),V[None,:,c:c+1].expand(m,K,1))
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,m,K,10)
        tot+=(L.argmax(-1)==arr[None,:,None,1]).float().sum((1,2))
    return (tot/(A*K)).cpu().numpy()
qs=th.tensor((np.arange(K)+.5)/K,dtype=th.float64,device=dev)
def line2(uM):
    """k = 2: M fixed, S over equal-mass points of Phi(S)/Phi(M) ~ Beta(9,1); lower inputs at mu(S)"""
    fs=uM*th.tensor(Bd(9,1).ppf(qs.cpu().numpy()),device=dev); S=nd(fs).float(); M=nd(th.tensor(uM,dtype=th.float64,device=dev)).float()
    return th.stack([M.expand(K),S],-1), mu_below(S[:,None])[:,0]
def line3(uR,g):
    fR=th.tensor(uR,dtype=th.float64,device=dev); fM=fR+(1-fR)*g; S=nd(fR+(fM-fR)*qs).float(); R=nd(fR).float(); M=nd(fM).float()
    return th.stack([M.expand(K),S,R.expand(K)],-1), mu_below(R.view(1,1)).expand(K,1)[:,0]
with th.no_grad():
    acc=th.zeros(N,device=dev,dtype=th.float64); tru=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**lref; gg=th.Generator().manual_seed(151)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=gg).to(dev); y=target(x); o=x.argsort(-1,descending=True); vk=x.gather(-1,o[:,k-1:k])
        base=mu_below(vk).expand(len(x),10).clone(); base.scatter_(-1,o[:,:k],x.gather(-1,o[:,:k]))
        acc+=(forward(*W,base[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
        tru+=(forward(*W,x[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    ref=(acc/n_ref).cpu().numpy(); truth=(tru/n_ref).cpu().numpy(); pq=truth*(1-truth)
bias=100*(ref-truth)
print(f"{N-6} population networks (every 64th) + 6 zoo; MF{k} anchor ({A} arrangements, charged {cost_node:,} forwards per outer node), K = {K}; references 2^{lref}")
print(f"  MF{k} anchor bias against the true accuracy: mean {bias[pop].mean():+.3f}  RMS {np.sqrt(np.mean(bias[pop]**2)):.3f} points"
      f"  zoo "+" ".join(f"{v:+.3f}" for v in bias[-6:]))
for n in range(1,nmax+1):
    est=np.zeros(N)
    if k==2:
        u,wu=rule(n,10,1)
        for a,wa in zip(u,wu): est+=wa*G(lambda:line2(float(a)))
        nodes=n
    else:
        (ur,wr),(ug,wg)=rule(n,8,3),rule(n,2,1)
        for a,wa in zip(ur,wr):
            for c,wc in zip(ug,wg): est+=wa*wc*G(lambda:line3(float(a),float(c)))
        nodes=n*n
    e=100*(est-ref); cost=cost_node*nodes; samp=100*np.sqrt(np.mean(pq[pop])/cost)
    tot=np.sqrt(np.mean((100*(est-truth))[pop]**2))
    print(f"  order {n:2d} ({nodes:3d} outer nodes, about {cost:>9,} forwards): anchor integral error RMS {np.sqrt(np.mean(e[pop]**2)):.3f}"
          f"; error against the true accuracy RMS {tot:.3f}; sampling at that cost {samp:.3f}",flush=True)
