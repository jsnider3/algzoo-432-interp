"""The homogeneous-coordinate top-4 integral (z23) with its geometric kinks removed.
In z23 the outer (u4, u3) integrand had kinks of the coordinates' own making: the allowed theta set
{theta in (-3 pi/4, pi/4): R sin theta > u3}, c = u3 / R, is one arc (asin c, pi/4) for c > -1/sqrt 2, two arcs
(-3 pi/4, -pi - asin c) and (asin c, pi/4) for -1 < c < -1/sqrt 2, and the whole interval for c < -1; and the lower
end of u3 switches from -sqrt(1 - u4^2) to u4 at u4 = -1/sqrt 2.  Here the outer domain is split along those curves,
u3 = -sqrt((1 - u4^2) / 2) (c = -1) and u3 = -sqrt((1 - u4^2) / 3) (c = -1/sqrt 2), and at u4 = -1/sqrt 2, -1/sqrt 3,
-1/2, into 9 pieces, each with its own n x n Gauss-Legendre rule, and each theta arc gets its own K midpoints.
What is left is the network's own non-smoothness (5,040 arrangements' decision boundaries).  The convergence rate in
the number of outer nodes, against sampling's n^(-1/2), is the point of the test.
Every 64th population network + the 6 zoo models.  Usage: python scripts/z27.py [nmax, default 7] [K per arc, default 64] [log2 reference, default 24] [nmin, default 1]
"""
import sys, itertools, numpy as np, torch as th
from scipy.integrate import quad
from scipy.special import ndtr
from batch_train import forward, target
from z2 import load
from seeds_clocks import load as zload
dev='cuda'; nmax=int(sys.argv[1]) if len(sys.argv)>1 else 7; K=int(sys.argv[2]) if len(sys.argv)>2 else 64; lref=int(sys.argv[3]) if len(sys.argv)>3 else 24; nmin=int(sys.argv[4]) if len(sys.argv)>4 else 1
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev)
phi=lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)
def D(r): return quad(lambda p: p**3*np.exp(-p*p/2)*ndtr(p*r)**6,0,np.inf)[0]
def mu4(r): return -quad(lambda p: p**2*np.exp(-p*p/2)*phi(p*r)*ndtr(p*r)**5,0,np.inf)[0]/D(r)
rt=np.linspace(-1,1,4001); mut=np.array([mu4(r) for r in rt])
def mu_t(r): return th.from_numpy(np.interp(r.cpu().numpy(),rt,mut)).to(dev).float()

def arcs(u3,u4):
    R=np.sqrt(max(0.,1-u3*u3-u4*u4)); c=u3/R if R>0 else -np.inf
    if c<=-1: return [(-3*np.pi/4,np.pi/4)]
    a=np.arcsin(c)
    if c< -1/np.sqrt(2): return [(-3*np.pi/4,-np.pi-a),(a,np.pi/4)]
    return [(a,np.pi/4)] if a<np.pi/4 else []
@th.no_grad()
def arc_int(u3,u4):
    """sum over arcs of the theta-integral of mean_arrangements I0, and the total arc length"""
    R=np.sqrt(max(0.,1-u3*u3-u4*u4)); m=mu4(u4); tot=np.zeros(N); Ltot=0.
    for lo,hi in arcs(u3,u4):
        tg=th.tensor(lo+(hi-lo)*(np.arange(K)+.5)/K,dtype=th.float32,device=dev); u1=R*th.cos(tg); u2=R*th.sin(tg)
        acc=th.zeros(N,device=dev); A=len(ARR); ch=1260
        for a0 in range(0,A,ch):
            arr=ARR[a0:a0+ch]; a=len(arr); X=th.full((a,K,10),m,device=dev)
            X.scatter_(-1,arr[:,None,0:1].expand(a,K,1),u1[None,:,None].expand(a,K,1))
            X.scatter_(-1,arr[:,None,1:2].expand(a,K,1),u2[None,:,None].expand(a,K,1))
            X.scatter_(-1,arr[:,None,2:3].expand(a,K,1),th.full((a,K,1),u3,device=dev))
            X.scatter_(-1,arr[:,None,3:4].expand(a,K,1),th.full((a,K,1),u4,device=dev))
            L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,a,K,10)
            acc+=(L.argmax(-1)==arr[None,:,None,1]).float().sum((1,2))
        tot+=(acc/(A*K)).cpu().numpy()*(hi-lo); Ltot+=hi-lo
    return tot,Ltot
s2,s3=lambda u4:-np.sqrt((1-u4*u4)/2),lambda u4:-np.sqrt((1-u4*u4)/3)
hi3=lambda u4:np.sqrt((1-u4*u4)/3); lo_full=lambda u4:-np.sqrt(1-u4*u4); ident=lambda u4:u4
r2,r3=1/np.sqrt(2),1/np.sqrt(3)
PIECES=[((-1,-r2),lo_full,s2),((-1,-r2),s2,s3),((-1,-r2),s3,hi3),
        ((-r2,-r3),ident,s2),((-r2,-r3),s2,s3),((-r2,-r3),s3,hi3),
        ((-r3,-.5),ident,s3),((-r3,-.5),s3,hi3),((-.5,.5),ident,hi3)]

with th.no_grad():                                                                     # reference: the same anchor at sampled inputs
    acc=th.zeros(N,device=dev,dtype=th.float64); n_ref=2**lref; g=th.Generator().manual_seed(97)
    for i in range(0,n_ref,2**16):
        x=th.randn(2**16,10,generator=g).to(dev); y=target(x); o=x.argsort(-1,descending=True); top=x.gather(-1,o[:,:4])
        rho=top.norm(dim=-1,keepdim=True); r=top[:,3]/rho[:,0]
        base=(mu_t(r)[:,None]*rho).expand(len(x),10).clone(); base.scatter_(-1,o[:,:4],top)
        acc+=(forward(*W,base[None].expand(N,-1,-1)).argmax(-1)==y[None]).double().sum(1)
    ref=(acc/n_ref).cpu().numpy(); pq=ref*(1-ref)
print(f"{N-6} population networks (every 64th) + 6 zoo; homogeneous MF4 accuracy, outer domain split into {len(PIECES)} smooth pieces,"
      f" n x n Gauss-Legendre per piece, K = {K} theta points per arc; reference 2^{lref} sampled inputs (SE about {100*np.sqrt(np.median(pq[pop])/n_ref):.3f} points)")
rows=[]
for n in range(nmin,nmax+1):
    x_,w_=np.polynomial.legendre.leggauss(n); num=np.zeros(N); den=0.; nodes=0
    for (a4,b4),lo,hi in PIECES:
        for xa,wa in zip(x_,w_):
            u4=a4+(b4-a4)*(xa+1)/2; j4=(b4-a4)/2; l,h=lo(u4),hi(u4)
            if h<=l: continue
            Dw=D(u4)
            for xb,wb in zip(x_,w_):
                u3=l+(h-l)*(xb+1)/2; j3=(h-l)/2; v,Lr=arc_int(u3,u4); nodes+=1
                num+=wa*wb*j4*j3*Dw*v; den+=wa*wb*j4*j3*Dw*Lr
    e=100*(num/den-ref); cost=9300*nodes; samp=100*np.sqrt(np.mean(pq[pop])/cost); rms=np.sqrt(np.mean(e[pop]**2)); rows.append((nodes,rms))
    print(f"  order {n} ({nodes:4d} outer nodes, about {cost:>10,} forwards): error RMS {rms:.3f} points (median {np.median(e[pop]):+.3f})"
          f"   sampling at that cost {samp:.3f}   zoo "+" ".join(f"{v:+.3f}" for v in e[-6:]),flush=True)
r=np.array(rows); k=r[:,1]>0
if k.sum()>=3:
    sl=np.polyfit(np.log(r[k][-3:,0]),np.log(r[k][-3:,1]),1)[0]
    print(f"  error ~ nodes^{sl:.2f} over the last three orders (sampling: nodes^-0.50)")
