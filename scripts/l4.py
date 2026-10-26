import sys, time, torch as th
from k4 import train_from
from k1 import clean_err, best_pointer_sel
th.set_num_threads(1)
for s in range(5):
    for ds in range(71,77):
        t0=time.time(); W=train_from(f"ckpt/s{s}_e0.npz",ds,steps=2**15,warmup=2**14); a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
        print(f"seed {s} data {ds}: acc {a:.3f} pointer {bp[0]:.2f} switch={bp[0]<0.4} [{time.time()-t0:.0f}s]", flush=True)
