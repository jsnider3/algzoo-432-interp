import sys, numpy as np, torch as th
from m1 import A, B, train
from k1 import best_pointer_sel
th.set_num_threads(1)
for j in [int(a) for a in sys.argv[1].split(',')]:
    Whh=A[1].copy(); Wo=A[2].copy()
    Whh[j,:]=B[1][j,:]; Whh[:,j]=B[1][:,j]; Wo[:,j]=B[2][:,j]     # revert neuron j to seed 4
    for ds in [91,92]:
        out=train([B[0],Whh,Wo],ds,2**15,snaps=(2**15,)); bp=best_pointer_sel(*out[2**15])
        print(f"LOO revert n{j} | data {ds}: pointer {bp[0]:.2f} neuron {bp[1][0] if bp[1] else None} delay {bp[1][1] if bp[1] else None} switch={bp[0]<0.4}",flush=True)
