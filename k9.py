import numpy as np, itertools, time, torch as th
from scipy.stats import norm
from seeds_clocks import load, T
from j5 import gl
th.set_num_threads(2)
def logits(X,Wih,Whh,Wo):
    h=np.zeros((len(X),16))
    for t in range(T): h=np.maximum(X[:,t:t+1]*Wih+h@Whh.T,0)
    return h@Wo.T
def estimate(path, nr=8, ng=6, nq=5):
    Wih,Whh,Wo=load(path)
    r,wr=gl(-2.8,3.2,nr); u,wu=gl(0,1,ng); g=4*u**2; wg=wu*8*u
    R,G1,G2=[a.ravel() for a in np.meshgrid(r,g,g,indexing='ij')]; Wc=np.einsum('i,j,k->ijk',wr,wg,wg).ravel()
    S=R+G1; M=S+G2; dens=norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.cdf(R)**7
    lam=norm.pdf(R)/np.maximum(norm.cdf(R),1e-300); mu=-lam
    perms=np.array(list(itertools.permutations(range(10),3)))
    qu,qw=gl(0,1,nq)                                    # quantile nodes of truncated normal
    mf=0; nl=0; mass=0; step=max(1,60000//720)
    for c0 in range(0,len(R),step):
        cs=np.arange(c0,min(c0+step,len(R))); nc=len(cs)
        X=np.repeat(mu[cs][:,None],10,1)[:,None,:].repeat(720,1); ii=np.arange(nc)[:,None]; jj=np.arange(720)[None,:]
        X[ii,jj,perms[None,:,0]]=M[cs][:,None]; X[ii,jj,perms[None,:,1]]=S[cs][:,None]; X[ii,jj,perms[None,:,2]]=R[cs][:,None]
        X=X.reshape(-1,10); N=len(X); n=np.arange(N)
        P2=np.tile(perms[:,1],nc); Rr=np.repeat(R[cs],720)
        low=np.ones((N,10),bool); low[n,np.tile(perms[:,0],nc)]=False; low[n,P2]=False; low[n,np.tile(perms[:,2],nc)]=False
        L0=logits(X,Wih,Whh,Wo); D0=L0[n,P2][:,None]-L0          # mean-field margins
        Emean=np.zeros_like(D0); Evar=np.zeros_like(D0)
        lowidx=np.where(low)[1].reshape(N,7)
        for k in range(7):
            pos=lowidx[:,k]; f1=np.zeros_like(D0); f2=np.zeros_like(D0)
            for q,w in zip(qu,qw):
                z=norm.ppf(np.clip(q*norm.cdf(Rr),1e-12,1-1e-12)); Xk=X.copy(); Xk[n,pos]=z
                Lk=logits(Xk,Wih,Whh,Wo); fk=(Lk[n,P2][:,None]-Lk)-D0
                f1+=w*fk; f2+=w*fk**2
            Emean+=f1; Evar+=np.maximum(f2-f1**2,0)
        mean=D0+Emean; sd=np.sqrt(Evar)+1e-9
        pc_nl=np.prod(np.where(np.arange(10)[None,:]==P2[:,None],1.0,norm.cdf(mean/sd)),1)
        pc_mf=(L0.argmax(1)==P2).astype(float)
        wgt=np.repeat(Wc[cs]*dens[cs],720); mf+=np.sum(wgt*pc_mf); nl+=np.sum(wgt*pc_nl); mass+=np.sum(wgt)
    return mf/mass, nl/mass
truth={"mystery":(0.9529,-0.0030),"seed 2":(0.9483,-0.0015),"seed 0":(0.9100,-0.0015)}   # (true acc, true - fine-grid mean-field)
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
    t0=time.time(); mf,nl=estimate(path)
    print(f"{name:8s} coarse grid: mean-field {mf:.4f}, additive-nonlinear leakage {nl:.4f} -> leakage shift {nl-mf:+.4f} (target shift from fine grid {truth[name][1]:+.4f}) [{time.time()-t0:.0f}s]", flush=True)
