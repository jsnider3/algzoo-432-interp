import sys, numpy as np
from seeds_clocks import load, run, T
path=sys.argv[1]; Wih,Whh,Wo=load(path); rng=np.random.default_rng(12345); c=0; n=0
for _ in range(40):
    x=rng.standard_normal((500000,T)); c+=((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2]).sum(); n+=len(x)
p=c/n; print(f"REF {path} acc {p:.5f} ± {np.sqrt(p*(1-p)/n)*100:.4f} pts (n={n})",flush=True)
