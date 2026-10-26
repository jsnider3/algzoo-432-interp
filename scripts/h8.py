import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=250)
Wih,Whh,Wo=load("weights_h32.npz")
x=np.random.default_rng(9).standard_normal((600000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
o=np.argsort(-x,1); aM=9-o[:,0]; aS=9-o[:,1]
fr=(h>0).mean(0); sparse=np.where(fr<0.7)[0]
print("Wih:", Wih.round(1)); print("final active fraction:", fr.round(2))
def tuning(age):
    A=np.stack([(h[age==a]>0).mean(0) for a in range(10)])  # rows age
    return A
for nm,ag in [("ageM",aM),("ageS",aS)]:
    A=tuning(ag)[:,sparse]
    sel=(A.max(0)-A.min(0))
    print(f"\nsparse neurons {sparse.tolist()}: firing rate by {nm} (rows 0..9)"); print(A)
    print(f" preferred {nm} per sparse neuron:", A.argmax(0), " modulation depth:", sel.round(2))
