import sys, numpy as np, itertools, time
from scipy.stats import norm
from seeds_clocks import load
from j5 import gl
def logits(X,Wih,Whh,Wo):
    h=np.zeros((len(X),16))
    for t in range(10): h=np.maximum(X[:,t:t+1]*Wih+h@Whh.T,0)
    return h@Wo.T
def run(path, nr=10, ng=8, nq=4, part=0, nparts=1, raw=False):
    Wih,Whh,Wo=load(path)
    r,wr=gl(-2.8,3.2,nr); u,wu=gl(0,1,ng); g=4*u**2; wg=wu*8*u
    R,G1,G2=[a.ravel() for a in np.meshgrid(r,g,g,indexing='ij')]; Wc=np.einsum('i,j,k->ijk',wr,wg,wg).ravel()
    S=R+G1; M=S+G2; dens=norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.cdf(R)**7
    mu=-norm.pdf(R)/np.maximum(norm.cdf(R),1e-300)
    perms=np.array(list(itertools.permutations(range(10),3))); qu,qw=gl(0,1,nq)
    mf=0; hd=0; mass=0; step=max(1,40000//720)
    lo=len(R)*part//nparts; hi=len(R)*(part+1)//nparts
    for c0 in range(lo,hi,step):
        cs=np.arange(c0,min(c0+step,hi)); nc=len(cs)
        X=np.repeat(mu[cs][:,None],10,1)[:,None,:].repeat(720,1); ii=np.arange(nc)[:,None]; jj=np.arange(720)[None,:]
        X[ii,jj,perms[None,:,0]]=M[cs][:,None]; X[ii,jj,perms[None,:,1]]=S[cs][:,None]; X[ii,jj,perms[None,:,2]]=R[cs][:,None]
        X=X.reshape(-1,10); N=len(X); n=np.arange(N); P2=np.tile(perms[:,1],nc); Rr=np.repeat(R[cs],720)
        low=np.ones((N,10),bool); low[n,np.tile(perms[:,0],nc)]=False; low[n,P2]=False; low[n,np.tile(perms[:,2],nc)]=False
        lowidx=np.where(low)[1].reshape(N,7)
        I=lambda X_: (logits(X_,Wih,Whh,Wo).argmax(1)==P2).astype(float)
        I0=I(X); h=I0.copy()
        for k in range(7):
            Ek=np.zeros(N)
            for q,w in zip(qu,qw):
                Xk=X.copy(); Xk[n,lowidx[:,k]]=norm.ppf(np.clip(q*norm.cdf(Rr),1e-12,1-1e-12)); Ek+=w*I(Xk)
            h+=Ek-I0
        wgt=np.repeat(Wc[cs]*dens[cs],720); mf+=np.sum(wgt*I0); hd+=np.sum(wgt*np.clip(h,0,1)); mass+=np.sum(wgt)
    return (mf,hd,mass) if raw else (mf/mass, hd/mass)
if __name__=="__main__":
    name,path,mf_fine,true=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
    t0=time.time(); mfc,hdc=run(path); est=mf_fine+(hdc-mfc)
    print(f"{name}: true {true:.4f} | fine-grid mean-field {mf_fine:.4f} (err {mf_fine-true:+.4f}) | coarse-grid leakage correction {hdc-mfc:+.4f} | sampling-free combined {est:.4f} (err {est-true:+.4f}) [{time.time()-t0:.0f}s]",flush=True)
