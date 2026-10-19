import numpy as np, torch as th
from seeds_clocks import load, T
# REPORT 7.14 (p1.py/p2.py): step-0 gradient on n13's incoming recurrent row for readout transplants into
# seed 4's init with seed 3's W_hh, by two statistics: lead over the next neuron, and size over the median (as in m5.py).
A=load("ckpt/s3_e0.npz"); B=load("ckpt/s4_e0.npz")
th.manual_seed(0); x=th.randn(200000,10); y=x.argsort(-1)[:,-2]
def grads(Wih,Whh,Wo):
    P=[th.tensor(a,dtype=th.float64,requires_grad=True) for a in (Wih,Whh,Wo)]
    h=th.zeros(len(x),16,dtype=th.float64); X=x.double()
    for t in range(T): h=th.relu(X[:,t:t+1]*P[0]+h@P[1].T)
    l=th.nn.functional.cross_entropy(h@P[2].T,y); l.backward(); return [p.grad.numpy() for p in P]
def readout(kind):
    Wo=B[2].copy()
    if kind=="full seed 3 Wo": Wo=A[2].copy()
    if kind=="p1 columns": Wo[:,list(np.load("p1_cols.npy"))]=A[2][:,list(np.load("p1_cols.npy"))]
    return Wo
for kind in ["full seed 3 Wo","seed 4 Wo","p1 columns"]:
    nr=np.linalg.norm(grads(B[0],A[1],readout(kind))[1],axis=1)
    print(f"{kind:15s}: n13 rank {int((nr>nr[13]).sum())}, size/median {nr[13]/np.median(nr):.2f}, lead over next {nr[13]/np.delete(nr,13).max():.2f}")
