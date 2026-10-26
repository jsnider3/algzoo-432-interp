import sys, numpy as np, itertools, time
from scipy.stats import norm
from seeds_clocks import load
from j5 import gl
from l3 import logits
perms=np.array(list(itertools.permutations(range(10),3)))
def dens(M,S,R): return norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.cdf(R)**7
def lines_integral(W, Ms, As, wline, Rmin=-4.5, nscan=8, nbis=10, nfine=600):
    """for each line (M, a): integrate over b in [0,bmax] of dens*M^2*I, boundary-aware. returns sum over lines*perms and evals"""
    Wih,Whh,Wo=W; nl=len(Ms); aM=np.abs(Ms); S=Ms-As*aM; bmax=np.maximum((S-Rmin)/aM,1e-6)
    tq=np.linspace(0,1,nscan)**2                       # scan in t=b/bmax
    def indicator(Mv,Sv,Rv,pidx):
        mu=-norm.pdf(Rv)/np.maximum(norm.cdf(Rv),1e-300)
        X=np.repeat(mu[:,None],10,1); k=np.arange(len(Mv)); P=perms[pidx]
        X[k,P[:,0]]=Mv; X[k,P[:,1]]=Sv; X[k,P[:,2]]=Rv
        return logits(X,Wih,Whh,Wo).argmax(1)==P[:,1]
    L=np.repeat(np.arange(nl),720); Pp=np.tile(np.arange(720),nl)
    I=np.zeros((nl*720,nscan),bool); ev=0
    for j,t in enumerate(tq):
        b=t*bmax[L]; R=S[L]-b*aM[L]; I[:,j]=indicator(Ms[L],S[L],R,Pp); ev+=len(L)
    fl=np.where(I[:,1:]!=I[:,:-1]); lo=tq[fl[1]]; hi=tq[fl[1]+1]; Ilo=I[fl[0],fl[1]]; rows=fl[0]
    for _ in range(nbis):
        mid=(lo+hi)/2; li=L[rows]; R=S[li]-mid*bmax[li]*aM[li]
        Im=indicator(Ms[li],S[li],R,Pp[rows]); ev+=len(rows); same=Im==Ilo; lo=np.where(same,mid,lo); hi=np.where(same,hi,mid)
    bp=(lo+hi)/2
    tf=np.linspace(0,1,nfine)**2; dt=np.gradient(tf); total=0; mass=0
    for l in range(nl):
        R=S[l]-tf*bmax[l]*aM[l]; d=dens(Ms[l],S[l],R)*aM[l]**2*bmax[l]*dt*wline[l]
        mass+=d.sum()*720
        base=I[l*720:(l+1)*720]
        sel=(L[rows]==l)
        for p in range(720):
            r=l*720+p; spp=bp[sel & (Pp[rows]==p)]
            if len(spp)==0: total+= d.sum() if base[p,0] else 0.0; continue
            pts=np.sort(spp); state=bool(base[p,0]); idx=np.searchsorted(tf,pts); prev=0; acc=0.0
            for k in idx:
                if state: acc+=d[prev:k].sum()
                state=not state; prev=k
            if state: acc+=d[prev:].sum()
            total+=acc
    return total, mass, ev
def estimate(path, nM, na, nscan=8, nbis=10):
    W=load(path)
    # region M>0 : outer M in (0, 4.8], a with u^2 spacing in [0, amax(M)] where S >= -4.5
    m1,w1=gl(0,4.8,nM); m2,w2=gl(-3.0,0,max(3,nM//3)); m=np.r_[m1,m2]; wm=np.r_[w1,w2]; u,wu=gl(0,1,na)
    Ms=[];As=[];Wl=[]
    for Mi,wi in zip(m,wm):
        amax=max((Mi+4.5)/abs(Mi),1e-6); a=amax*u**2; wa=wu*2*amax*u
        Ms+=list(np.full(na,Mi)); As+=list(a); Wl+=list(wi*wa)
    tot,mass,ev=0,0,0
    Ms=np.array(Ms);As=np.array(As);Wl=np.array(Wl)
    for c in range(0,len(Ms),40):
        t,mm,e=lines_integral(W,Ms[c:c+40],As[c:c+40],Wl[c:c+40],nscan=nscan,nbis=nbis); tot+=t;mass+=mm;ev+=e
    return tot, mass, ev
if __name__=="__main__":
    path=sys.argv[1]; true=float(sys.argv[2])
    pneg=1/1024    # P(max<0); contribution handled separately below
    for nM,na in [(6,6),(8,8),(10,10),(12,12)]:
        t0=time.time(); tot,mass,ev=estimate(path,nM,na)
        print(f"{path}: nM={nM} na={na}: M>0 mass {mass:.5f} (exact {1-pneg:.5f}) | P(correct, M>0) contribution {tot:.5f} | evals {ev/1e6:.2f}M [{time.time()-t0:.0f}s]",flush=True)
