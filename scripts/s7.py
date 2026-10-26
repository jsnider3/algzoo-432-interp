import sys, time
from r2 import estimate
path,ref=sys.argv[1],float(sys.argv[2])
for nM,na,ns in [(12,12,24),(20,20,24),(32,32,24)]:
    t0=time.time(); tot,mass,ev=estimate(path,nM,na,nscan=ns); print(f"{path} nM={nM} na={na} nscan={ns}: estimate {tot/mass:.5f} (ref {ref:.5f}: {(tot/mass-ref)*100:+.3f}) {ev/1e6:.1f}M evals [{time.time()-t0:.0f}s]",flush=True)
