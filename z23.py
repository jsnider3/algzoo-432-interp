"""The top-4 integral in homogeneous coordinates: 2 outer dimensions instead of 3 (ChatGPT round 4's reduction).
The network is positively homogeneous, so correctness depends only on the direction of the input.  Write the top-4
values as rho*u (|u| = 1, u1 > u2 > u3 > u4) and the lower inputs as rho*z_j (z_j < u4).  Integrating rho out exactly,
the ordered directions u have density proportional to D(u4) on the sphere, with
  D(r) = int_0^inf rho^3 exp(-rho^2/2) Phi(rho r)^6 d rho,
and the homogeneous mean-field anchor puts every lower ratio at its conditional mean
  mu4(r) = -int rho^2 exp(-rho^2/2) phi(rho r) Phi(rho r)^5 d rho / D(r)          (mu4(0) = -1/2)
so the homogeneous MF4 accuracy is a target-independent-weighted integral over (theta, u3, u4), u1 = R cos theta,
u2 = R sin theta, R = sqrt(1 - u3^2 - u4^2), d sigma = d theta du3 du4, theta in (-3 pi/4, pi/4) with R sin theta > u3.
Here the theta integral (all 5,040 arrangements, K midpoints) is brute force, and the outer (u4, u3) integral uses an
n x n Gauss-Legendre rule (u4 over (-1, 1/2), u3 from max(u4, -sqrt(1 - u4^2)) to sqrt((1 - u4^2) / 3)); the reference is the same anchor's
accuracy at 2^22 sampled inputs.  An exact theta arc costs about what an exact S line does, so order n costs about
9,300 n^2 forwards.  Every 64th population network + the 6 zoo models.  Usage: python z23.py [nmax, default 8] [K, default 128 theta points over (-3 pi/4, pi/4)]
"""
import sys, itertools, numpy as np, torch as th
from scipy.integrate import quad
from scipy.special import ndtr
from batch_train import forward, target
from z2 import load
from seeds_clocks import load as zload
dev='cuda'; nmax=int(sys.argv[1]) if len(sys.argv)>1 else 8; K=int(sys.argv[2]) if len(sys.argv)>2 else 128
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev)
phi=lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)
def D(r): return quad(lambda p: p**3*np.exp(-p*p/2)*ndtr(p*r)**6,0,np.inf)[0]
def mu4(r): return -quad(lambda p: p**2*np.exp(-p*p/2)*phi(p*r)*ndtr(p*r)**5,0,np.inf)[0]/D(r)
assert abs(mu4(0.)+.5)<1e-7
rt=np.linspace(-1,1,4001); mut=th.tensor([mu4(r) for r in rt],dtype=th.float32,device=dev)   # table for the sampled reference
def mu_t(r): return th.from_numpy(np.interp(r.cpu().numpy(),rt,mut.cpu().numpy())).to(dev).float()

@th.no_grad()
def arc(u3,u4):
    """theta-integral of mean_arrangements I0 (absolute, d theta) and the arc length"""
    R=np.sqrt(max(0.,1-u3*u3-u4*u4))
    tg=-3*np.pi/4+np.pi*(np.arange(K)+.5)/K; tg=tg[R*np.sin(tg)>u3]                  # u1 > u2 (theta < pi/4) and u2 > u3
    if len(tg)==0: return np.zeros(N),0.
    L_=np.pi*len(tg)/K; th_=th.tensor(tg,dtype=th.float32,device=dev); Kt=len(tg)
    u1=R*th.cos(th_); u2=R*th.sin(th_); m=mu4(u4); tot=th.zeros(N,device=dev); A=len(ARR); ch=1260
    for a0 in range(0,A,ch):
        arr=ARR[a0:a0+ch]; a=len(arr); X=th.full((a,Kt,10),m,device=dev)
        X.scatter_(-1,arr[:,None,0:1].expand(a,Kt,1),u1[None,:,None].expand(a,Kt,1))
        X.scatter_(-1,arr[:,None,1:2].expand(a,Kt,1),u2[None,:,None].expand(a,Kt,1))
        X.scatter_(-1,arr[:,None,2:3].expand(a,Kt,1),th.full((a,Kt,1),u3,device=dev))
        X.scatter_(-1,arr[:,None,3:4].expand(a,Kt,1),th.full((a,Kt,1),u4,device=dev))
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,a,Kt,10)
        tot+=(L.argmax(-1)==arr[None,:,None,1]).float().sum((1,2))
    return (tot/(A*Kt)).cpu().numpy()*L_, L_

with th.no_grad():                                                                     # reference: sampled inputs, homogeneous anchor
    acc=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**22; g=th.Generator().manual_seed(79)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=g).to(dev); y=target(x); o=x.argsort(-1,descending=True); top=x.gather(-1,o[:,:4])
        rho=top.norm(dim=-1,keepdim=True); r=top[:,3]/rho[:,0]
        base=(mu_t(r)[:,None]*rho).expand(len(x),10).clone(); base.scatter_(-1,o[:,:4],top)
        acc+=(forward(*W,base[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    ref=(acc/n_ref).cpu().numpy()
print(f"{N-6} population networks (every 64th) + 6 zoo; homogeneous MF4 accuracy by n x n outer rules, K = {K} theta points")
print(f"  reference at 2^22 sampled inputs: pop median {np.median(ref[pop]):.4f}")
pq=ref*(1-ref)
for n in range(1,nmax+1):
    x4,w4=np.polynomial.legendre.leggauss(n); xs,ws=x4,w4
    num=np.zeros(N); den=0.
    for a,wa in zip(x4,w4):
        u4=-1+1.5*(a+1)/2; up=np.sqrt((1-u4*u4)/3); lo3=max(u4,-np.sqrt(1-u4*u4)); Dw=D(u4)*1.5/2
        for b,wb in zip(xs,ws):
            u3=lo3+(up-lo3)*(b+1)/2; jac=(up-lo3)/2
            v,Lr=arc(u3,u4); num+=wa*wb*Dw*jac*v; den+=wa*wb*Dw*jac*Lr
    est=num/den; e=100*(est-ref); cost=9300*n*n; samp=100*np.sqrt(np.mean(pq[pop])/cost)
    print(f"  order {n} ({n*n:3d} outer nodes, about {cost:>9,} forwards): error RMS {np.sqrt(np.mean(e[pop]**2)):.3f} points (median {np.median(e[pop]):+.3f})"
          f"   sampling at that cost {samp:.3f}   zoo "+" ".join(f"{v:+.3f}" for v in e[-6:]),flush=True)
