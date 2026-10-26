import time
from r2 import estimate
for nM,na,ns in [(20,20,24),(32,32,24)]:
    t0=time.time(); tot,mass,ev=estimate("weights.npz",nM,na,nscan=ns); print(f"nM={nM} na={na} nscan={ns}: estimate {tot/mass:.5f} (ref 0.95313: {(tot/mass-0.95313)*100:+.3f}) {ev/1e6:.1f}M evals [{time.time()-t0:.0f}s]",flush=True)
