# Train a population of fresh 16-neuron seeds (AlgZoo schedule, 2^27 sequences) and save weights to pop/.
import sys, numpy as np, torch as th
from alg_zoo import DistRNN
th.set_num_threads(1)
steps = 2**17
for seed in [int(a) for a in sys.argv[1].split(',')]:
    th.manual_seed(70000+seed); m = DistRNN(hidden_size=16, seq_len=10)
    opt = th.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.001)
    for s in range(steps):
        mult = min(max(s, 1)/2**14, 1)*max(0.01**(2*s/steps), 0.01)
        for g in opt.param_groups: g['lr'] = 0.01*mult
        x = th.randn(1024, 10); loss = th.nn.functional.cross_entropy(m(x), x.argsort(-1)[:, -2])
        opt.zero_grad(); loss.backward(); opt.step()
    np.savez(f"pop/p{seed}.npz", **{k: v.detach().numpy() for k, v in m.state_dict().items()})
    print("done", seed, flush=True)
