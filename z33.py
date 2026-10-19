"""Deterministic pattern discovery for the linear-region closed form.  z30's greedy oracle chose good patterns but needed
2^14 network evaluations per arrangement.  Here the patterns come from the S-lines that an exact-S integrator traces
anyway: at the n^3 outer nodes of z21's tensor Gauss-Jacobi rule, every arrangement's S-line (S over (R, M)) passes
through a sequence of linear regions, and each region's activation pattern gives an exact linear logit map.  All of them
are kept; each gets a centroid (the mean CDF coordinates of the line points in it).  The closed form answers a
configuration exactly if its pattern was discovered, and otherwise with the map of the nearest discovered centroid.
The line pieces are found here on a K-point S grid (an exact tracer would find every piece); the cost charged is that of
exact tracing, about 1.84 forwards per arrangement line (ChatGPT round 4's MF4 tree), so n^3 x 5,040 x 1.84.
Scored on held-out sampled configurations of NA arrangements per network (a measurement), arrangement noise removed.
Every 64th population network + the 6 zoo models.  Usage: python z33.py [nmax, default 3] [arrangements, default 100] [K, default 64]
"""
import sys, itertools, numpy as np, torch as th
from scipy.special import roots_jacobi
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; nmax=int(sys.argv[1]) if len(sys.argv)>1 else 3; NA=int(sys.argv[2]) if len(sys.argv)>2 else 100; K=int(sys.argv[3]) if len(sys.argv)>3 else 64
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=list(itertools.permutations(range(10),4)); sel=[ARR[i] for i in np.linspace(0,len(ARR)-1,NA).astype(int)]
HK=th.randint(1,2**31,(160,),generator=th.Generator().manual_seed(5)).to(dev)
def run(w,X):
    h=th.zeros(len(X),16,device=dev); G=[]
    for t in range(10):
        pre=X[:,t:t+1]*w[0]+h@w[1].T; G.append(pre>0); h=th.relu(pre)
    G=th.stack(G,1); J=w[2][None].expand(len(X),10,16)*G[:,9,None,:].float(); C=[None]*10
    for t in range(9,-1,-1):
        C[t]=J@w[0]
        if t>0: J=(J@w[1])*G[:,t-1,None,:].float()
    return h@w[2].T,(G.reshape(len(X),-1).long()*HK).sum(-1),th.stack(C,1)
def beta_nodes(n,a,b): x,_=roots_jacobi(n,b-1,a-1); return (x+1)/2
def to_input(arr,cc):
    """CDF coordinates (uQ, r, g, s) -> anchor inputs for arrangement arr"""
    a,b,g,s=cc.unbind(-1); pr=a+(1-a)*b; pm=pr+(1-pr)*g; ps=pr+(pm-pr)*s; nd=th.special.ndtri
    M,S,R,Q=nd(pm),nd(ps),nd(pr),nd(a); X=mu_below(Q[:,None]).expand(len(cc),10).clone()
    for c,v in zip(range(4),(M,S,R,Q)): X[:,arr[c]]=v
    return X
g=th.Generator().manual_seed(127); nh=2**13; res=np.zeros((N,nmax,NA,3))
with th.no_grad():
    for ai,arr in enumerate(sel):
        v=th.randn(nh*3,10,generator=g).to(dev).topk(4,-1).values[:nh]; f=th.special.ndtr(v)
        ch=th.stack([f[:,3],(f[:,2]-f[:,3])/(1-f[:,3]),(f[:,0]-f[:,2])/(1-f[:,2]),(f[:,1]-f[:,2])/(f[:,0]-f[:,2])],-1)
        Xh=to_input(arr,ch); tgt=arr[1]
        for n in range(1,nmax+1):
            uq,ur,ug=beta_nodes(n,7,4),beta_nodes(n,1,3),beta_nodes(n,2,1)
            grid=th.tensor([(a,b,c_,(k+.5)/K) for a in uq for b in ur for c_ in ug for k in range(K)],dtype=th.float32,device=dev)
            Xg=to_input(arr,grid)
            for i in range(N):
                w=[q[i] for q in W]
                _,hg,Cg=run(w,Xg); Lh,hh,_=run(w,Xh); Ih=Lh.argmax(-1)==tgt
                u,inv=th.unique(hg,return_inverse=True); M_=len(u)
                cen=th.zeros(M_,4,device=dev).index_add_(0,inv,grid)/th.bincount(inv,minlength=M_)[:,None]
                first=th.full((M_,),-1,dtype=th.long,device=dev); first.scatter_(0,inv,th.arange(len(grid),device=dev)); Cu=Cg[first]
                own=(hh[:,None]==u[None]).any(1)
                nn=((ch[:,None,:]-cen[None])**2).sum(-1).argmin(1)
                a_=((Xh[:,:,None]*Cu[nn]).sum(1).argmax(-1)==tgt); Icf=th.where(own,Ih,a_)
                res[i,n-1,ai]=[own.float().mean().item(),(Icf!=Ih).float().mean().item(),(Icf.float().mean()-Ih.float().mean()).item()]
print(f"{N-6} population networks (every 64th) + 6 zoo; patterns discovered along the S-lines at the n^3 outer nodes (K = {K} points per line);"
      f" {NA} arrangements, 2^13 held-out configurations each")
for n in range(1,nmax+1):
    r=res[pop,n-1]; e=100*r[:,:,2]; tot=e.mean(1); noise2=e.var(1)/NA; c=1.84*5040*n**3
    print(f"  order {n} ({n**3} lines per arrangement, about {c:>9,.0f} forwards): coverage {np.median(r[:,:,0].mean(1)):.3f}; disagreement {100*np.median(r[:,:,1].mean(1)):.2f}%;"
          f" net error RMS {np.sqrt(np.mean(tot**2)):.3f} (arrangement noise removed {np.sqrt(max(0,np.mean(tot**2)-noise2.mean())):.3f}) points; sampling at that cost {100*np.sqrt(0.07/c):.3f}")
np.save('pop1k/seed0_z33.npy',res)
