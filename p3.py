import sys, numpy as np, torch as th
from alg_zoo import DistRNN
from m1 import A, B
from k1 import best_pointer_sel
th.set_num_threads(1)
mode=sys.argv[1]; W=[B[0],A[1],A[2]]
for ds in [91,92,93]:
    m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(W[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(W[1],dtype=th.float32),'linear.weight':th.tensor(W[2],dtype=th.float32)})
    params={"freeze_Wo":[m.rnn.weight_ih_l0,m.rnn.weight_hh_l0],"freeze_Wih":[m.rnn.weight_hh_l0,m.linear.weight],"only_row13+Wo":[m.rnn.weight_hh_l0,m.linear.weight,m.rnn.weight_ih_l0]}[mode]
    mask=None
    if mode=="only_row13+Wo":
        mask=th.zeros(16,16); mask[13]=1   # only n13's incoming row of Whh trains (+ its input weight + readout)
        maskih=th.zeros(16,1); maskih[13]=1
    opt=th.optim.AdamW(params,lr=0.01,weight_decay=0.001); th.manual_seed(ds)
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward()
        if mask is not None: m.rnn.weight_hh_l0.grad*=mask; m.rnn.weight_ih_l0.grad*=maskih
        opt.step()
        if mask is not None:   # undo weight decay drift on frozen entries
            pass
    Wf=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float)]
    bp=best_pointer_sel(*Wf); acc=None
    print(f"full s3 transplant, {mode} | data {ds}: pointer {bp[0]:.2f} neuron {bp[1][0] if bp[1] else None} delay {bp[1][1] if bp[1] else None} switch={bp[0]<0.4}",flush=True)
