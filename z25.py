"""A closed form for the recurrent state: one linear map per activation pattern.  Within one activation pattern the
network is exactly linear: h_T = sum_t (D_T Whh ... D_{t+1} Whh) D_t Wih x_t, so the logits are sum_t c_t x_t.  If each
arrangement's value space were covered by a few patterns, the MF4 anchor accuracy would need one forward per
(arrangement, pattern) to build the maps, and the value integral (a polyhedral region under the order-statistic law)
no further network evaluations: the outer-node multiplier that keeps z21 off the sampling line would disappear.
Test: for each arrangement, the activation patterns at K representative value points (an n x n x n Gauss-Jacobi grid
in the CDF coordinates of Q, R, M of z21, times m equal-mass S points; lower inputs at mu(Q)).  On 2^20 sampled top-4
configurations (a measurement only):
  coverage   share whose true activation pattern is one of its arrangement's K patterns (the closed form is exact there)
  nearest    the anchor accuracy if each configuration uses the linear map of the representative nearest in CDF coordinates
  covered    exact where covered, nearest elsewhere
against the true MF4 anchor accuracy on the same configurations.  Every 64th population network + the 6 zoo models.
Usage: python z25.py [n, default 2] [m, default 2] [log2 configurations, default 20]
"""
import sys, itertools, numpy as np, torch as th
from scipy.special import roots_jacobi
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; n=int(sys.argv[1]) if len(sys.argv)>1 else 2; m=int(sys.argv[2]) if len(sys.argv)>2 else 2; lc=int(sys.argv[3]) if len(sys.argv)>3 else 20
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=th.tensor(list(itertools.permutations(range(10),4)),device=dev); A=len(ARR)
code=th.full((10**4,),-1,dtype=th.long,device=dev); code[ARR[:,0]*1000+ARR[:,1]*100+ARR[:,2]*10+ARR[:,3]]=th.arange(A,device=dev)
def beta_nodes(n,a,b): x,_=roots_jacobi(n,b-1,a-1); return (x+1)/2
uq,ur,ug=beta_nodes(n,7,4),beta_nodes(n,1,3),beta_nodes(n,2,1); us=(np.arange(m)+.5)/m
REP=np.array([(a,b,c,d) for a in uq for b in ur for c in ug for d in us]); K=len(REP)       # (K, 4) CDF coordinates
def values(c):
    """CDF coordinates (uQ, r, g, s) -> values M, S, R, Q"""
    a,b,g,s=c[...,0],c[...,1],c[...,2],c[...,3]; pr=a+(1-a)*b; pm=pr+(1-pr)*g; ps=pr+(pm-pr)*s
    nd=th.special.ndtri; return nd(pm),nd(ps),nd(pr),nd(a)
def inputs(arr,M,S,R,Q):
    X=mu_below(Q[:,None]).expand(len(arr),10).clone()
    for c,v in ((0,M),(1,S),(2,R),(3,Q)): X.scatter_(-1,arr[:,c:c+1],v[:,None])
    return X
def gates_maps(w,X):
    """true activation pattern (B, 10, 16) bool and the pattern's linear logit map c (B, 10 steps, 10 outputs)"""
    h=th.zeros(len(X),16,device=dev); G=[]
    for t in range(10):
        pre=X[:,t:t+1]*w[0]+h@w[1].T; G.append(pre>0); h=th.relu(pre)
    G=th.stack(G,1); J=w[2][None].expand(len(X),10,16)*G[:,9,None,:].float(); C=[None]*10
    for t in range(9,-1,-1):
        C[t]=J@w[0]                                                                      # (B, 10)
        if t>0: J=(J@w[1])*G[:,t-1,None,:].float()
    return G,th.stack(C,1)
def pack(G): return (G.reshape(len(G),-1).long()*(th.arange(160,device=dev)%61+1)*(th.arange(160,device=dev)//61*7919+1)).sum(-1)   # pattern hash
rc=th.tensor(REP,dtype=th.float32,device=dev)
x=th.randn(2**lc,10,generator=th.Generator().manual_seed(83)).to(dev)
out=np.zeros((N,6))
with th.no_grad():
    o=x.argsort(-1,descending=True); arr_i=code[o[:,0]*1000+o[:,1]*100+o[:,2]*10+o[:,3]]; top=x.gather(-1,o[:,:4])
    Mv,Sv,Rv,Qv=top.unbind(-1); y=o[:,1]
    fQ,fR,fM,fS=[th.special.ndtr(v) for v in (Qv,Rv,Mv,Sv)]
    cc=th.stack([fQ,(fR-fQ)/(1-fQ),(fM-fR)/(1-fR),(fS-fR)/(fM-fR)],-1)                   # sampled configurations' CDF coordinates
    near=((cc[:,None,:]-rc[None])**2).sum(-1).argmin(1)                                 # nearest representative
    X=inputs(ARR[arr_i],Mv,Sv,Rv,Qv)
    for i in range(N):
        w=[q[i] for q in W]
        repG=th.empty(A,K,dtype=th.long,device=dev); repC=th.empty(A,K,10,10,device=dev)
        for k in range(K):
            Mr,Sr,Rr,Qr=values(rc[k][None].expand(A,4)); G,C=gates_maps(w,inputs(ARR,Mr,Sr,Rr,Qr)); repG[:,k]=pack(G); repC[:,k]=C
        tot=th.zeros(6,device=dev,dtype=th.float64)
        for s0 in range(0,len(X),2**16):
            sl=slice(s0,s0+2**16); Gs,_=gates_maps(w,X[sl]); h=pack(Gs); ai=arr_i[sl]
            I0=((X[sl,:,None]*0).sum(1)==0)                                               # placeholder shape
            L=forward(w[0][None],w[1][None],w[2][None],X[sl][None])[0]; I0=(L.argmax(-1)==y[sl])
            match=(repG[ai]==h[:,None]); cov=match.any(1)
            Cn=repC[ai,near[sl]]; Ln=(X[sl,:,None]*Cn).sum(1); In=(Ln.argmax(-1)==y[sl])
            Icov=th.where(cov,I0,In)
            tot+=th.stack([I0.sum(),cov.sum(),In.sum(),Icov.sum(),(In!=I0).sum(),th.zeros((),device=dev)]).double()
        out[i]=(tot/len(X)).cpu().numpy()
print(f"{N-6} population networks (every 64th) + 6 zoo; K = {K} representative patterns per arrangement ({n}^3 x {m}); "
      f"building the maps costs about {K*A:,} forwards; 2^{lc} sampled configurations")
def show(lab,v,fmt):
    print(f"  {lab:56s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  RMS {fmt.format(np.sqrt(np.mean(v[pop]**2)))}  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
show("true MF4 anchor accuracy",out[:,0],"{:.4f}")
show("coverage: configurations in one of the K patterns",out[:,1],"{:.4f}")
show("nearest-pattern closed form: error (points)",100*(out[:,2]-out[:,0]),"{:+.3f}")
show("  answers differing from the network (% of configurations)",100*out[:,4],"{:.3f}")
show("exact where covered, nearest elsewhere: error (points)",100*(out[:,3]-out[:,0]),"{:+.3f}")
np.save(f'pop1k/seed0_z25_n{n}m{m}.npy',out)
