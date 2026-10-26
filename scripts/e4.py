import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def lsq_probe(F,y,k,n):
    Y=np.eye(k)[y]; A=np.c_[F,np.ones(len(F))]; c,*_=np.linalg.lstsq(A[:n],Y[:n],rcond=None)
    return ((A[n:]@c).argmax(1)==y[n:]).mean()
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(6).standard_normal((300000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    o=np.argsort(-x,1); M=-np.sort(-x,1)[:,0]; aS=9-o[:,1]; aM=9-o[:,0]
    # features: h/M, plus per-neuron squared terms? keep linear but on h/M and active indicators
    F=np.c_[h/M[:,None], (h>0).astype(float)]
    n=225000; base=lsq_probe(F,aS,10,n)
    drops=[]
    for j in range(16):
        keep=[c for c in range(32) if c not in (j,16+j)]
        drops.append(base-lsq_probe(F[:,keep],aS,10,n))
    print(f"\n{name}: lsq probe ageS on [h/M, active masks] acc {base:.3f}; acc drop when removing each neuron:"); print(np.array(drops).round(3))
    # also timing: does ageS info come from history of sparse neurons? correlation of each neuron's 'ever fired since S' ...
    print(" active frac at final step:", (h>0).mean(0).round(2))
    print(" Wih:", Wih.round(1))
