import numpy as np, torch as th
from m1 import A, B
from seeds_clocks import T
th.manual_seed(0); x=th.randn(60000,10).double(); y=x.argsort(-1)[:,-2]
Wih=th.tensor(B[0]).double(); Whh=th.tensor(A[1]).double()
def dom(Wo_np):
    Whh_=Whh.clone().requires_grad_(True); Wo=th.tensor(Wo_np).double()
    h=th.zeros(len(x),16,dtype=th.float64)
    for t in range(T): h=th.relu(x[:,t:t+1]*Wih+h@Whh_.T)
    th.nn.functional.cross_entropy(h@Wo.T,y).backward()
    nr=Whh_.grad.norm(dim=1).numpy(); return nr[13]/np.max(np.delete(nr,13)), nr
full,_=dom(A[2]); base,_=dom(B[2]); print(f"n13 lead (n13 grad / best other): full s3 Wo {full:.2f}, s4 Wo {base:.2f}")
Wo=B[2].copy(); Wo[:,13]=A[2][:,13]; chosen=[13]; cur,_=dom(Wo); print(f"  + col 13: {cur:.2f}")
while cur<full*0.95 and len(chosen)<16:
    best=None
    for j in range(16):
        if j in chosen: continue
        W2=Wo.copy(); W2[:,j]=A[2][:,j]; d,_=dom(W2)
        if best is None or d>best[0]: best=(d,j)
    chosen.append(best[1]); Wo[:,best[1]]=A[2][:,best[1]]; cur=best[0]; print(f"  + col {best[1]}: lead {cur:.2f}",flush=True)
np.save("p1_cols.npy",np.array(chosen)); print("minimal column set:",chosen)
