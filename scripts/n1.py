import sys, time
from l3 import run
name,path,true=sys.argv[1],sys.argv[2],float(sys.argv[3])
t0=time.time(); mf,hd=run(path,nr=20,ng=16,nq=4)
print(f"{name}: true {true:.4f} | fine-grid mean-field {mf:.4f} (err {mf-true:+.4f}) | fine-grid mean-field + cut-HDMR leakage {hd:.4f} (err {hd-true:+.4f}) | leakage correction {hd-mf:+.4f} [{time.time()-t0:.0f}s]",flush=True)
