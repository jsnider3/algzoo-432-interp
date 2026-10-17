import sys, time, numpy as np
from r4 import lines_integral
from j5 import gl
from seeds_clocks import load
def replicate(W, n, rng, nscan=24):
    Mmax=rng.uniform(4.6,5.2); Mmin=-rng.uniform(2.8,3.4); pw=rng.uniform(1.7,2.5); Rmin=-rng.uniform(4.3,4.9)
    m1,w1=gl(0,Mmax,n); m2,w2=gl(Mmin,0,max(3,n//3)); m=np.r_[m1,m2]; wm=np.r_[w1,w2]; u,wu=gl(0,1,n)
    Ms=[];As=[];Wl=[]
    for Mi,wi in zip(m,wm):
        amax=max((Mi-Rmin)/abs(Mi),1e-6); a=amax*u**pw; wa=wu*pw*amax*u**(pw-1)
        Ms+=list(np.full(n,Mi)); As+=list(a); Wl+=list(wi*wa)
    Ms=np.array(Ms);As=np.array(As);Wl=np.array(Wl); tot=mass=ev=0
    for c in range(0,len(Ms),40):
        t,mm,e=lines_integral(W,Ms[c:c+40],As[c:c+40],Wl[c:c+40],Rmin=Rmin,nscan=nscan,nbis=10); tot+=t;mass+=mm;ev+=e
    return tot, mass, ev
if __name__=="__main__":
    path,ref,n,K,seed=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
    W=load(path); rng=np.random.default_rng(seed); vals=[]; EV=0; t0=time.time()
    for k in range(K):
        t,m,e=replicate(W,n,rng); vals.append(t/m); EV+=e
    vals=np.array(vals); est=vals.mean(); se=vals.std(ddof=1)/np.sqrt(K); sd=np.sqrt(ref*(1-ref)/EV)
    print(f"RGL {path} n={n} K={K} seed={seed}: replicates {np.round(vals,5)} | estimate {est:.5f} ± {se*100:.4f} pts (err {(est-ref)*100:+.4f}, z={(est-ref)/se:+.2f}) | {EV/1e6:.2f}M evals | sampling sd {sd*100:.4f} | |err|/sd {abs(est-ref)/sd:.2f}x se/sd {se/sd:.2f}x [{time.time()-t0:.0f}s]",flush=True)
