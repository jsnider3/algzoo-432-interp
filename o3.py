import sys, numpy as np, itertools, time
from scipy.stats import norm
from seeds_clocks import load
from j5 import gl
from l3 import logits
def boundary_quad(path, nr=20, ng2=16, nscan=48, nbis=14, gmax=4.0, chunk=40):
    Wih,Whh,Wo=load(path)
    r,wr=gl(-3.2,3.6,nr); u,wu=gl(0,1,ng2); g2=gmax*u**2; wg2=wu*2*gmax*u
    perms=np.array(list(itertools.permutations(range(10),3)))
    scan=gmax*np.linspace(0,1,nscan)**2
    fine=gmax*np.linspace(0,1,4001)**2; dfine=np.gradient(fine)
    total=0; mass=0; nev=0
    lines=[(i,j) for i in range(nr) for j in range(ng2)]
    for c0 in range(0,len(lines),chunk):
        L=lines[c0:c0+chunk]; nl=len(L)
        Rl=np.array([r[i] for i,j in L]); G2=np.array([g2[j] for i,j in L]); Wl=np.array([wr[i]*wg2[j] for i,j in L])
        mu=-norm.pdf(Rl)/np.maximum(norm.cdf(Rl),1e-300)
        def ind(g1):  # g1: (nl,720) -> indicator (nl,720)
            S=Rl[:,None]+g1; M=S+G2[:,None]
            X=np.repeat(mu[:,None,None],10,2).repeat(720,1).copy()
            ii=np.arange(nl)[:,None]; jj=np.arange(720)[None,:]
            X[ii,jj,perms[None,:,0]]=M; X[ii,jj,perms[None,:,1]]=S; X[ii,jj,perms[None,:,2]]=Rl[:,None].repeat(720,1)
            return (logits(X.reshape(-1,10),Wih,Whh,Wo).argmax(1)==np.tile(perms[:,1],nl)).reshape(nl,720)
        I=np.stack([ind(np.full((nl,720),g)) for g in scan],2)    # (nl,720,nscan)
        nonlocal_ev=nl*720*nscan
        # switch points: refine by bisection in each interval where I changes
        flips=np.where(I[:,:,1:]!=I[:,:,:-1])
        lo=scan[flips[2]].copy(); hi=scan[flips[2]+1].copy(); Ilo=I[flips[0],flips[1],flips[2]]
        for _ in range(nbis):
            mid=(lo+hi)/2
            # evaluate only the flagged (line,perm) pairs
            g1=np.full((nl,720),np.nan); 
            # vectorized evaluation for flagged pairs
            S=Rl[flips[0]]+mid; M=S+G2[flips[0]]
            X=np.repeat(mu[flips[0]][:,None],10,1)
            k=np.arange(len(mid)); P=perms[flips[1]]
            X[k,P[:,0]]=M; X[k,P[:,1]]=S; X[k,P[:,2]]=Rl[flips[0]]
            Im=(logits(X,Wih,Whh,Wo).argmax(1)==P[:,1])
            same=Im==Ilo; lo=np.where(same,mid,lo); hi=np.where(same,hi,mid)
        bpt=(lo+hi)/2; nev+=nonlocal_ev+nbis*len(mid)
        # integrate density * step indicator over fine grid
        for li in range(nl):
            S=Rl[li]+fine; M=S+G2[li]; dens=norm.pdf(M)*norm.pdf(S)*norm.pdf(Rl[li])*norm.cdf(Rl[li])**7
            base=np.sum(dens*dfine)
            sel=flips[0]==li
            for p in range(720):
                # build indicator on fine grid from scan values and refined switch points
                sp=sel&(flips[1]==p)
                Ifine=np.interp(fine,scan,I[li,p].astype(float))>0.5 if not sp.any() else None
                if sp.any():
                    pts=np.sort(bpt[sp]); state=bool(I[li,p,0]); Ifine=np.empty(len(fine),bool); prev=0
                    idx=np.searchsorted(fine,pts)
                    for b in idx: Ifine[prev:b]=state; state=not state; prev=b
                    Ifine[prev:]=state
                total+=Wl[li]*np.sum(dens*dfine*Ifine)
            mass+=Wl[li]*base*720
    return total/mass, mass, nev
if __name__=="__main__":
    name,path,target,true=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
    t0=time.time(); v,m,nev=boundary_quad(path)
    print(f"{name}: boundary-aware mean-field {v:.4f} (vs sampled MF {target:.4f}: {v-target:+.4f}; vs TRUE acc {true:.4f}: {v-true:+.4f}) | mass {m:.4f} | {nev/1e6:.1f}M evals [{time.time()-t0:.0f}s]",flush=True)
