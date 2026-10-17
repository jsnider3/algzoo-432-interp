import sys, torch as th
from m1 import A, B, train
from k1 import best_pointer_sel
import numpy as np
th.set_num_threads(1)
which=sys.argv[1]
Wo=B[2].copy()
if which=="Wo13_only": Wo[:,13]=A[2][:,13]
elif which=="Wo_all_but13": Wo=A[2].copy(); Wo[:,13]=B[2][:,13]
W=[B[0],A[1],Wo]
for ds in [91,92,93,94]:
    out=train(W,ds,2**15,snaps=(2**15,)); Wf=out[2**15]; bp=best_pointer_sel(*Wf)
    print(f"s3 Whh + {which} (rest s4) | data {ds}: pointer {bp[0]:.2f} neuron {bp[1][0] if bp[1] else None} delay {bp[1][1] if bp[1] else None} switch={bp[0]<0.4}",flush=True)
