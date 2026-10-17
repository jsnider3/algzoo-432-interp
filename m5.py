import numpy as np
from m1 import A, B
from m4 import grads
def mixWo(kind):
    Wo=B[2].copy()
    if kind=="full": Wo=A[2].copy()
    elif kind=="Wo13_only": Wo[:,13]=A[2][:,13]
    elif kind=="Wo_all_but13": Wo=A[2].copy(); Wo[:,13]=B[2][:,13]
    return Wo
gF=grads(B[0],A[1],A[2])
for kind,rate in [("full","6/6"),("s4 Wo","0/4"),("Wo13_only","0/4"),("Wo_all_but13","1/4")]:
    g=grads(B[0],A[1],mixWo(kind)); nr=np.linalg.norm(g[1],axis=1)
    c=(g[1][13]*gF[1][13]).sum()/np.linalg.norm(g[1][13])/np.linalg.norm(gF[1][13])
    print(f"{kind:13s} (switch {rate}): n13 row-grad rank {int((nr>nr[13]).sum())}, |row grad| {nr[13]:.4f} (ratio to median {nr[13]/np.median(nr):.2f}), input-grad {g[0][13]:+.4f}, cos to full {c:+.2f}")
