import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
Wih,Whh,Wo=load("weights_seed2.npz")
x=np.random.default_rng(81).standard_normal((2000000,T)); o=np.argsort(-x,1); keep=o[:,0]==0; x=x[keep]; o=o[keep]
hs=run(x,Wih,Whh); aS=9-o[:,1]; S=-np.sort(-x,1)[:,1]
print("seed 2, max at position 0. Final-step active fraction by ageS (rows 0..8) per neuron:")
A=np.stack([(hs[aS==a,-1]>0).mean(0) for a in range(9)]); print(A)
print("\nWhich neurons' on/off state across ALL time steps predicts ageS? lsq probe on masks of neuron n over t=0..9:")
def lsq(F,y,n=len(x)*3//4):
    Y=np.eye(9)[y]; A_=np.c_[F,np.ones(len(F))]; c,*_=np.linalg.lstsq(A_[:n],Y[:n],rcond=None); return ((A_[n:]@c).argmax(1)==y[n:]).mean()
masks=(hs>0).astype(float)
print(" per neuron:", np.array([lsq(masks[:,:,nn],aS) for nn in range(16)]).round(2))
print(" all masks:", round(lsq(masks.reshape(len(x),-1),aS),3), " final values h/S:", round(lsq(hs[:,-1]/S[:,None],aS),3))
