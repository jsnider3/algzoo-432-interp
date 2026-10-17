import sys, time, torch as th
from k4 import train_from
from k1 import clean_err, best_pointer_sel
th.set_num_threads(1)
seeds=[int(a) for a in sys.argv[1].split(',')]
for s in seeds:
    for ds in range(61,67):
        t0=time.time(); W=train_from(f"ckpt/s{s}_e0.npz",ds,steps=2**15,warmup=2**14); a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
        print(f"seed {s} data {ds}: acc {a:.3f} clean-err {ce:.4f} pointer {bp[0]:.2f} switch={bp[0]<0.4} [{time.time()-t0:.0f}s]", flush=True)
