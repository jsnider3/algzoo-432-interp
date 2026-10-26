import numpy as np
from seeds_clocks import load
from e5 import runL
print(f"{'model':8s} | max also in last 10:  L=12   L=17   L=30 | max older than 10 steps: L=12  L=17  L=30")
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); a=[];b=[]
    for L in [12,17,30]:
        x=np.random.default_rng(L).standard_normal((400000,L)); p=(runL(x,Wih,Whh)@Wo.T).argmax(1)
        o=np.argsort(x,1); p2=o[:,-2]; p1=o[:,-1]; ok=(p+L-10)==p2; s=p2>=L-10
        a.append(ok[s&(p1>=L-10)].mean()); b.append(ok[s&(p1<L-10)].mean())
    print(f"{name:8s} | {'':20s}"+"  ".join(f"{v:.3f}" for v in a)+" | "+" "*24+"  ".join(f"{v:.3f}" for v in b))
