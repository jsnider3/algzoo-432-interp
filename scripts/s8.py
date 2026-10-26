import sys, time
import r4
path,ref=sys.argv[1],float(sys.argv[2])
for nM,na,ns in [(8,8,16),(12,12,16),(12,12,24),(16,16,24),(20,20,24),(24,24,24),(32,32,24)]:
    t0=time.time(); tot,mass,ev=r4.estimate(path,nM,na,nscan=ns); print(f"CACHED {path} nM={nM} na={na} nscan={ns}: estimate {tot/mass:.5f} (ref {ref:.5f}: {(tot/mass-ref)*100:+.4f}) {ev/1e6:.2f}M seq-equivalent evals [{time.time()-t0:.0f}s]",flush=True)
