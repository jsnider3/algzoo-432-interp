import numpy as np
from m1 import delay1_neither, A, B
def mix(idx):
    Whh=B[1].copy(); Wo=B[2].copy()
    for j in idx: Whh[j,:]=A[1][j,:]; Whh[:,j]=A[1][:,j]; Wo[:,j]=A[2][:,j]
    return B[0],Whh
conds={"full":(list(range(16)),"6/6"),"half0-7":(list(range(8)),"2/4"),"half8-15":(list(range(8,16)),"1/4"),"only13":([13],"0/4"),"all_but13":([j for j in range(16) if j!=13],"1/4"),"none (pure s4)":([],"2/15")}
print("n13 one-step-late firing at INIT (no training): neither-rate, new-2nd rate | observed switch rate")
for k,(idx,rate) in conds.items():
    ne,ev=delay1_neither(*mix(idx),13); print(f"  {k:15s}: neither {ne:.2f} new-2nd {ev:.2f}  | {rate}")
# also: Whh-only and Wo-only transplants (switch 0/4 and 1/4)
ne,ev=delay1_neither(B[0],A[1],13); print(f"  {'Whh only':15s}: neither {ne:.2f} new-2nd {ev:.2f}  | 0/4")
ne,ev=delay1_neither(B[0],B[1],13); print(f"  {'Wo only (=s4 dyn)':15s}: neither {ne:.2f} new-2nd {ev:.2f}  | 1/4")
