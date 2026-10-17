import sys, numpy as np, torch as th
from m1 import A, B, train
from k1 import best_pointer_sel
th.set_num_threads(1)
cols=list(np.load("p1_cols.npy")); Wo=B[2].copy()
for j in cols: Wo[:,j]=A[2][:,j]
W=[B[0],A[1],Wo]
for ds in [int(a) for a in sys.argv[1].split(',')]:
    out=train(W,ds,2**15,snaps=(2**15,)); bp=best_pointer_sel(*out[2**15])
    print(f"s3 Whh + s3 Wo cols {cols} (rest s4) | data {ds}: pointer {bp[0]:.2f} neuron {bp[1][0] if bp[1] else None} delay {bp[1][1] if bp[1] else None} switch={bp[0]<0.4}",flush=True)
