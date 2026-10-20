"""Would more training networks help the learned-row estimator?  Learning curve: rows selected (z68's greedy OMP) and
the ridge readout fitted on n of the 1,024 training networks (n = 128, 256, 512, 768, 1,024; three random subsets for
each n below 1,024), from the K = 8 one-box MF4 pool (5,760 rows) with the pulses and the 20 weight statistics; each
fit is scored on the 512 new networks (independent labels).
Usage: python z70.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60
from z62 import ridge
from z68 import omp, feats
KS=(16,64,128)
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz')
    W=[th.cat([a.float(),b.float()]) for a,b in zip(Wt,Wn)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy')]
    new=np.arange(1024,1536); pqn=(y[new]*(1-y[new])).mean()
    base=np.c_[z60.pulses(*Wnp),gfeat.stats(*Wnp)]; _,_,_,c=gfeat.summary(W,1,8,cells=True); M=c[2].astype(float); B=c[0].astype(float)
    print("RMS on the 512 new networks (points), mean over subsets (range)")
    for n in (128,256,512,768,1024):
        res={k:[] for k in KS}
        for rep in range(1 if n==1024 else 3):
            tr=np.sort(np.random.default_rng(100+rep).permutation(1024)[:n]); sel=omp(M[tr],y[tr],base[tr],max(KS))
            for k in KS:
                F=feats(M,B,sel[:k],base); res[k].append(100*np.sqrt(((ridge(F[tr],y[tr],F[new])[0]-y[new])**2).mean()))
        print(f"  n = {n:4d}: "+" | ".join(f"{k} rows {np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})" for k,v in res.items()),flush=True)
