import sys, numpy as np, torch as th
from alg_zoo import DistRNN
from k1 import best_pointer_sel, clean_err
th.set_num_threads(1)
K=int(sys.argv[2]); steps=2**17
for init in [int(a) for a in sys.argv[1].split(',')]:
    th.manual_seed(1000+init); m=DistRNN(hidden_size=16,seq_len=10)
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(5000+init)
    for s in range(steps):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward()
        if s<K: m.linear.weight.grad=None
        opt.step()
    W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
    bp=best_pointer_sel(W[0],W[1]); a,ce=clean_err(*W)
    print(f"2^27 seqs | init {init}, freeze-readout-steps {K}: acc {a:.3f} clean-err {ce:.4f} pointer {bp[0]:.2f} switch={bp[0]<0.4}",flush=True)
