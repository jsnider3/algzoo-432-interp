import sys, numpy as np
from l3 import run
part=int(sys.argv[1]); mf,hd,mass=run("weights.npz",nr=20,ng=16,nq=12,part=part,nparts=3,raw=True)
print(f"PART {part} {mf} {hd} {mass}",flush=True)
