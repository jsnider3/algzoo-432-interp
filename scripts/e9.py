import numpy as np
from seeds_clocks import load, run, T
print(f"{'model':8s} acc   | err S<0  | errors in regions: S<0   near-tie M~S (<.05M)  near-tie S~R3 (<.05M)  neither | err rate 'clean' (S>0, both gaps>.1M)")
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(14).standard_normal((800000,T)); pm=(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S,R3=srt[:,0],srt[:,1],srt[:,2]; err=pm!=o[:,1]
    neg=S<0; t12=~neg&((M-S)<.05*M); t23=~neg&~t12&((S-R3)<.05*M); rest=~(neg|t12|t23)
    E=err.sum(); clean=(S>0)&((M-S)>.1*M)&((S-R3)>.1*M)
    print(f"{name:8s} {1-err.mean():.3f} | {err[neg].mean():.3f}    | {'':17s}{err[neg].sum()/E:.2f}   {err[t12].sum()/E:.2f}{'':18s}{err[t23].sum()/E:.2f}{'':21s}{err[rest].sum()/E:.2f}   | {err[clean].mean():.4f} (covers {clean.mean():.2f} of inputs)")
