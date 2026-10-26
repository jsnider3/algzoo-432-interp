import numpy as np, itertools, time
from scipy.stats import norm
from seeds_clocks import load, run, T
from j5 import run_jac, gl
def estimate_trunc(path, nr=10, ng=8, nq=24):
    Wih,Whh,Wo=load(path)
    r,wr=gl(-2.8,3.2,nr); u,wu=gl(0,1,ng); g=4*u**2; wg=wu*8*u
    R,G1,G2=[a.ravel() for a in np.meshgrid(r,g,g,indexing='ij')]; W=np.einsum('i,j,k->ijk',wr,wg,wg).ravel()
    S=R+G1; M=S+G2; dens=norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.cdf(R)**7
    lam=norm.pdf(R)/np.maximum(norm.cdf(R),1e-300); mu=-lam; var=np.maximum(1-R*lam-lam**2,1e-12)
    perms=np.array(list(itertools.permutations(range(10),3)))
    # quadrature nodes for truncated normal Z<R (conditional): use CDF-uniform nodes
    qu,qw=gl(0,1,nq)
    num=0; mass=0; step=max(1,150000//720)
    for c0 in range(0,len(R),step):
        cs=np.arange(c0,min(c0+step,len(R))); nc=len(cs)
        X=np.repeat(mu[cs][:,None],10,1)[:,None,:].repeat(720,1); ii=np.arange(nc)[:,None]; jj=np.arange(720)[None,:]
        X[ii,jj,perms[None,:,0]]=M[cs][:,None]; X[ii,jj,perms[None,:,1]]=S[cs][:,None]; X[ii,jj,perms[None,:,2]]=R[cs][:,None]
        X=X.reshape(-1,10); L,J=run_jac(X,Wih,Whh,Wo); N=len(X); n=np.arange(N)
        P1,P2,P3=np.tile(perms[:,0],nc),np.tile(perms[:,1],nc),np.tile(perms[:,2],nc)
        low=np.ones((N,10),bool); low[n,P1]=low[n,P2]=low[n,P3]=False
        Rr=np.repeat(R[cs],720); MU=np.repeat(mu[cs],720); V=np.repeat(var[cs],720)
        D=L[n,P2][:,None]-L; JD=J[n,P2][:,None,:]-J            # D_j and its sensitivity to each input
        pc=np.ones(N)
        for j in range(10):
            isj=(j!=P2)
            a=JD[:,j,:]                                         # sensitivity of D_j to each input
            own=low[:,j]                                        # competitor j is a lower input: its own value truncated
            a_own=np.where(own,a[n,j],0.0)
            others=low.copy(); others[n,j]=False
            s_oth=np.sqrt(np.sum(a**2*others,1)*V)+1e-9
            # integrate over own input z ~ N(0,1) | z<R  (nodes via inverse CDF), delta = z - MU
            Fz=norm.cdf(Rr)
            pl=np.zeros(N)
            for q,w in zip(qu,qw):
                z=norm.ppf(np.clip(q*Fz,1e-12,1-1e-12)); dlt=np.where(own,z-MU,0.0)
                pl+=w*norm.sf((D[:,j]+a_own*dlt)/s_oth)
            pl=np.where(isj,pl,0.0)
            pc*=1-pl
        wgt=np.repeat(W[cs]*dens[cs],720); num+=np.sum(wgt*pc); mass+=np.sum(wgt)
    return num/mass
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); xt=np.random.default_rng(5).standard_normal((2000000,T)); true=np.mean((run(xt,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(xt,1)[:,-2])
    t0=time.time(); est=estimate_trunc(path)
    print(f"{name:8s} true {true:.4f} | sampling-free mean-field + truncated linear leakage {est:.4f} ({est-true:+.4f}) | {time.time()-t0:.0f}s")
