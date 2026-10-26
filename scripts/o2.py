import sys, numpy as np, itertools, time
from scipy.stats import norm
from seeds_clocks import load
from j5 import gl
from l3 import logits
def mf_quad(path, nr, ng, power=2):
    Wih,Whh,Wo=load(path)
    r,wr=gl(-3.2,3.6,nr); u,wu=gl(0,1,ng); g=4*u**power; wg=wu*4*power*u**(power-1)
    R,G1,G2=[a.ravel() for a in np.meshgrid(r,g,g,indexing='ij')]; Wc=np.einsum('i,j,k->ijk',wr,wg,wg).ravel()
    S=R+G1; M=S+G2; dens=norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.cdf(R)**7; mu=-norm.pdf(R)/np.maximum(norm.cdf(R),1e-300)
    perms=np.array(list(itertools.permutations(range(10),3))); num=0; mass=0; step=max(1,200000//720)
    for c0 in range(0,len(R),step):
        cs=np.arange(c0,min(c0+step,len(R))); nc=len(cs)
        X=np.repeat(mu[cs][:,None],10,1)[:,None,:].repeat(720,1); ii=np.arange(nc)[:,None]; jj=np.arange(720)[None,:]
        X[ii,jj,perms[None,:,0]]=M[cs][:,None]; X[ii,jj,perms[None,:,1]]=S[cs][:,None]; X[ii,jj,perms[None,:,2]]=R[cs][:,None]
        P2=np.tile(perms[:,1],nc); ok=(logits(X.reshape(-1,10),Wih,Whh,Wo).argmax(1)==P2)
        wgt=np.repeat(Wc[cs]*dens[cs],720); num+=np.sum(wgt*ok); mass+=np.sum(wgt)
    return num/mass, mass
name,path,target=sys.argv[1],sys.argv[2],float(sys.argv[3])
for nr,ng,pw in [(20,16,2),(20,16,3),(24,24,3),(28,32,3)]:
    t0=time.time(); v,m=mf_quad(path,nr,ng,pw)
    print(f"{name}: nr={nr} ng={ng} g=4u^{pw}: mean-field {v:.4f} (vs sampled MF {target:.4f}: {v-target:+.4f}) mass {m:.4f} evals {nr*ng*ng*720/1e6:.1f}M [{time.time()-t0:.0f}s]",flush=True)
