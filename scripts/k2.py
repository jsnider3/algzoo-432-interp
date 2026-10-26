import numpy as np, time
from k1 import finetune, clean_err, best_pointer_sel, load
for s in [2,0]:
    for e in [20,22]:
        path=f"ckpt/s{s}_e{e}.npz"
        for ds in [21,22,23]:
            t0=time.time(); W=finetune(path,ds,steps=2**15,start_step=2**(e-10)); a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
            print(f"seed {s} from 2^{e} (data seed {ds}) +2^25 seqs: acc {a:.3f} clean-err {ce:.4f} | most selective pointer neither-rate {bp[0]:.2f} {bp[1][:2] if bp[1] else None} [{time.time()-t0:.0f}s]", flush=True)
