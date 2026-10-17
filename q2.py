import sys, numpy as np, torch as th
from alg_zoo import DistRNN
from seeds_clocks import load
from k1 import best_pointer_sel, clean_err
th.set_num_threads(1)
def go(W, ds, freeze_ro, tag):
    m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(W[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(W[1],dtype=th.float32),'linear.weight':th.tensor(W[2],dtype=th.float32)})
    ps=[m.rnn.weight_ih_l0,m.rnn.weight_hh_l0]+([] if freeze_ro else [m.linear.weight])
    opt=th.optim.AdamW(ps,lr=0.01,weight_decay=0.001); th.manual_seed(ds)
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    Wf=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
    bp=best_pointer_sel(Wf[0],Wf[1]); a,ce=clean_err(*Wf)
    print(f"{tag} | data {ds}: acc {a:.3f} clean-err {ce:.4f} pointer {bp[0]:.2f} switch={bp[0]<0.4}",flush=True)
job=sys.argv[1]
if job=="s4own":
    W=load("ckpt/s4_e0.npz")
    for ds in [91,92,93,94]: go(W,ds,True,"seed 4 init, OWN readout frozen")
else:
    for init in [int(a) for a in job.split(',')]:
        th.manual_seed(1000+init); m=DistRNN(hidden_size=16,seq_len=10)
        W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
        go(W,5000+init,True,f"fresh init {init}, readout frozen")
