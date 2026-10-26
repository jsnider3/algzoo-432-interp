import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=1, suppress=True, linewidth=220)
for name,path in [("seed 0","weights_seed0.npz"),("mystery","weights.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(2).standard_normal((400000,T)); L=run(x,Wih,Whh)[:,-1]@Wo.T
    L=L-L.mean(1,keepdims=True)
    o=np.argsort(-x,1); M=-np.sort(-x,1)[:,0]; aM=9-o[:,0]
    for a in [9,4]:
        s=aM==a; F=np.c_[x,M][s]; c,*_=np.linalg.lstsq(F,L[s],rcond=None)
        print(f"\n{name}: max at position {9-a}; centered-logit weights (rows logit i, cols x_0..x_9, M)"); print(c.T)
