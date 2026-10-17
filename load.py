import torch as th, numpy as np
from alg_zoo import example_2nd_argmax
m = example_2nd_argmax()
sd = {k: v.numpy() for k, v in m.state_dict().items()}
np.savez("work/weights.npz", **sd)
for k,v in sd.items(): print(k, v.shape)
th.manual_seed(0)
x = th.randn(200000, 10)
with th.no_grad(): out = m(x)
print("acc", (out.argmax(-1)==x.argsort(-1)[:,-2]).float().mean().item())
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("W_ih", sd['rnn.weight_ih_l0'].ravel())
print("W_hh\n", sd['rnn.weight_hh_l0'])
print("W_out\n", sd['linear.weight'])
