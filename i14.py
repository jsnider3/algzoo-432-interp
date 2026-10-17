import numpy as np
from seeds_clocks import load, run, T
print("firing rate at step t (pooled t=3..9) by what x_t (delay 0) or x_(t-1) (delay 1) was")
for name,path,n,dl in [("mystery n5","weights.npz",5,1),("seed 2 n10","weights_seed2.npz",10,0),("seed 0 n0","weights_seed0.npz",0,1),("seed 0 n7","weights_seed0.npz",7,0),("seed 3 n9","weights_seed3.npz",9,0)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(151).standard_normal((200000,T)); hs=run(x,Wih,Whh)
    a={'new max':[],'new 2nd':[],'neither':[]}
    for t in range(3,T):
        e=t-dl; P=x[:,:e]; srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; xe=x[:,e]; f=hs[:,t,n]>0
        a['new max'].append(f[xe>M].mean()); a['new 2nd'].append(f[(xe>S)&(xe<M)].mean()); a['neither'].append(f[xe<S].mean())
    print(f"  {name:11s}: " + "  ".join(f"{k} {np.mean(v):.2f}" for k,v in a.items()))
