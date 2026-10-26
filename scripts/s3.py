import time
import r2
from r2 import estimate
for ns in [8,16,32,64]:
    t0=time.time(); tot,mass,ev=estimate("weights.npz",12,12,nscan=ns); print(f"nM=12 na=12 nscan={ns:2d}: estimate {tot/mass:.5f} (ref 0.95313: {(tot/mass-0.95313)*100:+.3f} pts) {ev/1e6:.2f}M evals [{time.time()-t0:.0f}s]",flush=True)
