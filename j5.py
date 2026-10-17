import numpy as np, itertools, time
from scipy.stats import norm
from seeds_clocks import load, T
def run_jac(x,Wih,Whh,Wo):
    N=len(x); h=np.zeros((N,16)); dh=np.zeros((N,16,T))
    for t in range(T):
        pre=x[:,t:t+1]*Wih+h@Whh.T; m=(pre>0)
        dp=np.einsum('ij,njk->nik',Whh,dh); dp[:,:,t]+=Wih
        dh=dp*m[:,:,None]; h=pre*m
    return h@Wo.T, np.einsum('oh,nht->not',Wo,dh)
def gl(a,b,n):
    u,w=np.polynomial.legendre.leggauss(n); return (b-a)/2*u+(a+b)/2, w*(b-a)/2
def estimate(path, nr=20, ng=16, chunk=150000):
    Wih,Whh,Wo=load(path)
    r,wr=gl(-2.8,3.2,nr); u,wu=gl(0,1,ng); g=4*u**2; wg=wu*8*u
    R,G1,G2=np.meshgrid(r,g,g,indexing='ij'); W=np.einsum('i,j,k->ijk',wr,wg,wg)
    R,G1,G2,W=R.ravel(),G1.ravel(),G2.ravel(),W.ravel()
    S=R+G1; M=S+G2
    dens=norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.cdf(R)**7
    lam=norm.pdf(R)/np.maximum(norm.cdf(R),1e-300); mu=-lam; var=np.maximum(1-R*lam-lam**2,0)
    perms=np.array(list(itertools.permutations(range(10),3)))   # (720,3) positions of M,S,R3
    totU=0; totP=0; totMF=0; mass=0
    cells=np.arange(len(R))
    for c0 in range(0,len(cells),max(1,chunk//720)):
        cs=cells[c0:c0+max(1,chunk//720)]
        nc=len(cs); X=np.repeat(mu[cs][:,None],10,1)[:,None,:].repeat(720,1)   # (nc,720,10)
        ii=np.arange(nc)[:,None]
        X[ii,np.arange(720)[None,:],perms[None,:,0]]=M[cs][:,None]
        X[ii,np.arange(720)[None,:],perms[None,:,1]]=S[cs][:,None]
        X[ii,np.arange(720)[None,:],perms[None,:,2]]=R[cs][:,None]
        X=X.reshape(-1,10); L,J=run_jac(X,Wih,Whh,Wo)
        p2=np.tile(perms[:,1],nc); n=np.arange(len(X))
        low=np.ones((len(X),10),bool); low[n,np.tile(perms[:,0],nc)]=False; low[n,p2]=False; low[n,np.tile(perms[:,2],nc)]=False
        v=np.repeat(var[cs],720)
        D=L[n,p2][:,None]-L                              # margins (10)
        JD=J[n,p2][:,None,:]-J                           # (N,10 competitors,10 inputs)
        sd=np.sqrt(np.sum((JD**2)*low[:,None,:],2)*v[:,None])+1e-9
        z=D/sd; z[n,p2]=np.inf
        pe=norm.sf(z)                                    # pairwise loss probabilities
        pc_union=np.clip(1-pe.sum(1),0,1); pc_prod=np.prod(1-pe,1); pc_mf=(L.argmax(1)==p2).astype(float)
        wgt=np.repeat(W[cs]*dens[cs],720)
        totU+=np.sum(wgt*pc_union); totP+=np.sum(wgt*pc_prod); totMF+=np.sum(wgt*pc_mf); mass+=np.sum(wgt)
    return totMF/mass, totU/mass, totP/mass, mass*720
if __name__=="__main__":
    for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
        Wih,Whh,Wo=load(path); xt=np.random.default_rng(5).standard_normal((2000000,T))
        from seeds_clocks import run
        true=np.mean((run(xt,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(xt,1)[:,-2])
        t0=time.time(); mf,un,pr,mass=estimate(path)
        print(f"{name:8s} true {true:.4f} | sampling-free: mean-field {mf:.4f} ({mf-true:+.4f}), +linearized leakage (union) {un:.4f} ({un-true:+.4f}), (product) {pr:.4f} ({pr-true:+.4f}) | quadrature mass check {mass:.4f} | {time.time()-t0:.0f}s")
