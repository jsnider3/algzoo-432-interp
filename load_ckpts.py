import numpy as np
from alg_zoo import zoo_2nd_argmax
for s in range(5):
    for e in [0]+list(range(20,31)):
        n = 0 if e==0 else 2**e
        m = zoo_2nd_argmax(16, 10, seed=s, n_seqs=n)
        np.savez(f"ckpt/s{s}_e{e}.npz", **{k: v.numpy() for k, v in m.state_dict().items()})
    print("seed", s, "done", flush=True)
