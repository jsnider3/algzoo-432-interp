import numpy as np
from seeds_clocks import load, run, T
x=np.random.default_rng(8).standard_normal((200000,T)); o=np.argsort(-x,1); aM=9-o[:,0]
g=-np.diff(-np.sort(-x,1),axis=1)[:,1]
def diag_ratio(path):
    Wih,Whh,Wo=load(path); L=run(x,Wih,Whh)[:,-1]@Wo.T; pm=L.argmax(1); L=L-L.mean(1,keepdims=True); dg=[]
    for a in range(10):
        s=aM==a; c,*_=np.linalg.lstsq(x[s],L[s],rcond=None); W=c.T; p=9-a
        for i in range(10):
            if i!=p: dg.append(abs(W[i,i])/(np.abs(np.delete(W[i],[i,p])).sum()+1e-9))
    return np.median(dg), (pm==o[:,1])[g>0.3].mean()
for s in [2,0,3]:
    print(f"seed {s}: train seqs -> (readout diag/offdiag, acc on gap>.3)")
    print("  "+"  ".join(f"2^{e}:{diag_ratio(f'ckpt/s{s}_e{e}.npz')[0]:.2f}/{diag_ratio(f'ckpt/s{s}_e{e}.npz')[1]:.3f}" for e in range(22,31)))
