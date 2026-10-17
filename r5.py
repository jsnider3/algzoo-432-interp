import sys, time, itertools, numpy as np
from scipy.stats import norm
from seeds_clocks import load
from j5 import gl
P4=np.array(list(itertools.permutations(range(10),4)))         # positions of M,S,R,R4
seqs=np.zeros((len(P4),10),int)
for i,p in enumerate(P4):
    for k in range(4): seqs[i,p[k]]=k+1                      # 0=mu (lower 6), 1=M,2=S,3=R,4=R4
levels=[]; prev=None
for d in range(10):
    nodes={}; par=[]; sym=[]
    for s in seqs:
        pre=tuple(s[:d+1])
        if pre not in nodes: nodes[pre]=len(nodes); par.append(prev[tuple(s[:d])] if d>0 else 0); sym.append(s[d])
    levels.append((np.array(par),np.array(sym))); prev=nodes
leaf=np.array([prev[tuple(s)] for s in seqs]); NSTEP=sum(len(p) for p,_ in levels)
def logits5(vals,Wih,Whh,Wo):
    h=np.zeros((len(vals),1,16))
    for par,sym in levels: h=np.maximum(vals[:,sym][:,:,None]*Wih[None,None,:]+h[:,par,:]@Whh.T,0)
    return h[:,leaf,:]@Wo.T
def direct(vals_rows,pidx,Wih,Whh,Wo):
    n=len(pidx); X=np.repeat(vals_rows[:,0:1],10,1); P=P4[pidx]
    for k in range(4): X[np.arange(n),P[:,k]]=vals_rows[:,k+1]
    h=np.zeros((n,16))
    for t in range(10): h=np.maximum(X[:,t:t+1]*Wih+h@Whh.T,0)
    return (h@Wo.T).argmax(1)==P[:,1]
def dens(M,S,R,R4): return norm.pdf(M)*norm.pdf(S)*norm.pdf(R)*norm.pdf(R4)*norm.cdf(R4)**6
def estimate(path,nM,na,nc,nscan=16,nbis=8,Rmin=-5.0,nfine=300):
    Wih,Whh,Wo=load(path)
    m1,w1=gl(0,4.8,nM); m2,w2=gl(-3.0,0,max(3,nM//3)); m=np.r_[m1,m2]; wm=np.r_[w1,w2]; u,wu=gl(0,1,na); v,wv=gl(0,1,nc)
    tq=np.linspace(0,1,nscan)**2; tf=np.linspace(0,1,nfine)**2; dt=np.gradient(tf)
    tot=0; mass=0; ev=0; NP=len(P4)
    for Mi,wi in zip(m,wm):
        aM=abs(Mi)
        for uj,wuj in zip(u,wu):
            amax=max((Mi-Rmin)/aM,1e-6); a=amax*uj**2; wa=wuj*2*amax*uj; S=Mi-a*aM
            bmax=max((S-Rmin)/aM,1e-6)
            for vk,wvk in zip(v,wv):
                # b in [0,bmax] inner (boundary-aware); c = (R-R4)/|M| in [0, cmax(b)] mapped by fraction vk of remaining room
                def R_R4(t):
                    R=S-t*bmax*aM; cmax=np.maximum((R-Rmin)/aM,1e-6); c=cmax*vk**2; return R, R-c*aM, cmax
                I=np.zeros((NP,nscan),bool)
                for j,t in enumerate(tq):
                    R,R4,_=R_R4(t); mu=-norm.pdf(R4)/max(norm.cdf(R4),1e-300)
                    I[:,j]=(logits5(np.array([[mu,Mi,S,R,R4]]),Wih,Whh,Wo)[0].argmax(1)==P4[:,1]); ev+=NSTEP/10
                fl=np.where(I[:,1:]!=I[:,:-1]); lo=tq[fl[1]]; hi=tq[fl[1]+1]; Ilo=I[fl[0],fl[1]]
                for _ in range(nbis):
                    mid=(lo+hi)/2; R,R4,_=R_R4(mid); mu=-norm.pdf(R4)/np.maximum(norm.cdf(R4),1e-300)
                    rows=np.c_[mu,np.full(len(mid),Mi),np.full(len(mid),S),R,R4]
                    Im=direct(rows,fl[0],Wih,Whh,Wo); ev+=len(mid); same=Im==Ilo; lo=np.where(same,mid,lo); hi=np.where(same,hi,mid)
                bp=(lo+hi)/2
                R,R4,cmax=R_R4(tf); d=dens(Mi,S,R,R4)*aM**3*bmax*cmax*2*vk*wvk*dt*wa*wi
                mass+=d.sum()*NP; cum=np.r_[0,np.cumsum(d)]
                # integrate step function per arrangement
                acc=np.where(I[:,0],cum[-1],0.0)
                if len(bp):
                    idx=np.searchsorted(tf,bp); order=np.lexsort((bp,fl[0])); rws=fl[0][order]; ids=idx[order]
                    for r in np.unique(rws):
                        cuts=ids[rws==r]; state=bool(I[r,0]); prev_=0; s_=0.0
                        for k in cuts:
                            if state: s_+=cum[k]-cum[prev_]
                            state=not state; prev_=k
                        if state: s_+=cum[-1]-cum[prev_]
                        acc[r]=s_
                tot+=acc.sum()
    return tot/mass, mass, ev
if __name__=="__main__":
    path,ref=sys.argv[1],float(sys.argv[2]); nM,na,nc=[int(a) for a in sys.argv[3:6]]
    t0=time.time(); est,mass,ev=estimate(path,nM,na,nc)
    print(f"R4-EXACT {path} nM={nM} na={na} nc={nc}: estimate {est:.5f} (ref {ref:.5f}: {(est-ref)*100:+.3f} pts) mass {mass:.4f} | {ev/1e6:.1f}M seq-evals | trie {NSTEP} steps vs {len(P4)*10} [{time.time()-t0:.0f}s]",flush=True)
