import sys, time, numpy as np
from r4 import lines_integral
from seeds_clocks import load
def replicate(W, n, rng, nscan=24):
    Ms=[];As=[];Wl=[]
    # M>0: (0,4.8], M<0: (-3,0) with n//3 strata ; a=amax*u^2, u stratified
    for lo,hi,nm in [(0,4.8,n),(-3.0,0,max(3,n//3))]:
        mcell=(hi-lo)/nm
        for i in range(nm):
            Mi=lo+(i+rng.random())*mcell
            amax=max((Mi+4.5)/abs(Mi),1e-6)
            for j in range(n):
                u=(j+rng.random())/n; Ms.append(Mi); As.append(amax*u**2); Wl.append(mcell*(1/n)*2*amax*u)
    Ms=np.array(Ms);As=np.array(As);Wl=np.array(Wl); tot=mass=ev=0
    for c in range(0,len(Ms),40):
        t,m,e=lines_integral(W,Ms[c:c+40],As[c:c+40],Wl[c:c+40],nscan=nscan,nbis=10); tot+=t;mass+=m;ev+=e
    return tot, mass, ev
if __name__=="__main__":
    path,ref,n,K,seed=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
    W=load(path); rng=np.random.default_rng(seed); vals=[]; EV=0; t0=time.time()
    for k in range(K):
        t,m,e=replicate(W,n,rng); vals.append(t/m); EV+=e
    vals=np.array(vals); est=vals.mean(); se=vals.std(ddof=1)/np.sqrt(K)
    sd_samp=np.sqrt(ref*(1-ref)/EV)
    print(f"STRAT {path} n={n} K={K} seed={seed}: estimate {est:.5f} ± {se*100:.4f} pts (error vs ref {(est-ref)*100:+.4f}, z={(est-ref)/se:+.2f}) | {EV/1e6:.2f}M evals | sampling sd {sd_samp*100:.4f} | ratio |err|/sd {abs(est-ref)/sd_samp:.2f}x, se/sd {se/sd_samp:.2f}x [{time.time()-t0:.0f}s]",flush=True)
