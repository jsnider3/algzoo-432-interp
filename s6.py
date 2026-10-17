import sys, numpy as np
from seeds_clocks import load, run, T
path,seed=sys.argv[1],int(sys.argv[2]); Wih,Whh,Wo=load(path); rng=np.random.default_rng(seed); c=0; n=0
for _ in range(100):
    x=rng.standard_normal((500000,T)); c+=((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2]).sum(); n+=len(x)
print(f"PART {c} {n}",flush=True)
