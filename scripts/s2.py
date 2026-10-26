import time
from r2 import estimate
true=0.9532
for nM,na in [(24,6),(24,8),(24,12),(24,16),(6,24),(8,24),(12,24),(16,24)]:
    t0=time.time(); tot,mass,ev=estimate("weights.npz",nM,na); print(f"nM={nM:2d} na={na:2d}: estimate {tot/mass:.4f} ({ev/1e6:.2f}M evals) [{time.time()-t0:.0f}s]",flush=True)
