import numpy as np, time
from o3 import boundary_quad
true=0.9532
rows=[]
for nr,ng2 in [(8,6),(12,8),(16,12),(20,16)]:
    for ns in [8,16,32]:
        t0=time.time(); v,m,nev=boundary_quad("weights.npz",nr=nr,ng2=ng2,nscan=ns,nbis=10)
        err=abs(v-true); eff=nev/2.56
        sd=np.sqrt(true*(1-true)/eff)
        rows.append((nev,err,sd))
        print(f"r{nr:2d} g{ng2:2d} scan{ns:2d}: {nev/1e6:5.2f}M evals ({eff/1e6:5.2f}M with caching) | estimate {v:.4f} error {err*100:.3f} pts | sampling sd at cached cost {sd*100:.3f} pts | ratio {err/sd:5.1f}x [{time.time()-t0:.0f}s]",flush=True)
