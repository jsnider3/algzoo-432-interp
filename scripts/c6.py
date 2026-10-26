import numpy as np
from seeds_clocks import load, run, T
print(f"{'model':8s} {'acc':>6s} | err by gap S-R3: {'<.05':>6s} {'.05-.2':>6s} {'.2-.5':>6s} {'>.5':>6s} | wrong picks rank: max  3rd  4th+ | err by gap M-S >1")
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(3).standard_normal((600000,T)); pm=(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)
    o=np.argsort(-x,1); srt=-np.sort(-x,1); g=srt[:,1]-srt[:,2]; gm=srt[:,0]-srt[:,1]
    err=pm!=o[:,1]; rk=np.argsort(o,1)[np.arange(len(x)),pm]
    e=lambda s: err[s].mean()
    w=err
    print(f"{name:8s} {1-err.mean():.4f} | {'':16s}{e(g<.05):6.3f} {e((g>=.05)&(g<.2)):6.3f} {e((g>=.2)&(g<.5)):6.3f} {e(g>=.5):6.3f} | {'':17s}{(rk[w]==0).mean():.2f} {(rk[w]==2).mean():.2f} {(rk[w]>=3).mean():.2f} | {e(gm>1):.3f}")
