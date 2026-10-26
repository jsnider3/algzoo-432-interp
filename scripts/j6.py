import numpy as np, time
from j5 import estimate
true=0.9529; sdN=lambda N: np.sqrt(true*(1-true)/N)
for nr,ng in [(6,4),(8,6),(10,8),(14,10)]:
    t0=time.time(); mf,_,_,_=estimate("weights.npz",nr=nr,ng=ng); N=nr*ng*ng*720
    print(f"grid r{nr} x g{ng} x g{ng}: {N} model evals | mean-field estimate {mf:.4f} (error {mf-true:+.4f}) | sampling sd at same cost {sdN(N):.5f} | {time.time()-t0:.0f}s")
