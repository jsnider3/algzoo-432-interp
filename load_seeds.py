import numpy as np, torch as th
from alg_zoo import zoo_2nd_argmax
for s in range(5):
    m = zoo_2nd_argmax(16, 10, seed=s)
    sd = {k: v.numpy() for k, v in m.state_dict().items()}
    np.savez(f"work/weights_seed{s}.npz", **sd)
    th.manual_seed(0); x = th.randn(200000, 10)
    with th.no_grad(): a = (m(x).argmax(-1) == x.argsort(-1)[:, -2]).float().mean().item()
    print(s, {k: v.shape for k, v in sd.items()}, "acc %.4f" % a)
