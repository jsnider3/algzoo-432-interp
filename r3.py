import sys, time, numpy as np
from r2 import estimate
name,path,true=sys.argv[1],sys.argv[2],float(sys.argv[3])
for nM,na in [(6,6),(8,8),(10,10),(12,12),(16,16)]:
    t0=time.time(); tot,mass,ev=estimate(path,nM,na); est=tot/mass
    sd=np.sqrt(true*(1-true)/ev)
    print(f"{name}: nM={nM:2d} na={na:2d}: {ev/1e6:5.2f}M evals | mass {mass:.4f} | estimate {est:.4f} error {abs(est-true)*100:.3f} pts | sampling sd {sd*100:.3f} pts | ratio {abs(est-true)/sd:5.1f}x [{time.time()-t0:.0f}s]",flush=True)
