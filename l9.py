import sys, time, numpy as np, torch as th
from alg_zoo import DistRNN
from seeds_clocks import load
from k1 import best_pointer_sel
th.set_num_threads(1)
A=load("ckpt/s3_e0.npz"); B=load("ckpt/s4_e0.npz")
conds={"s3 Wih + s4 rest":[A[0],B[1],B[2]], "s3 Whh + s4 rest":[B[0],A[1],B[2]], "s3 Wo + s4 rest":[B[0],B[1],A[2]], "s3 Wih+Whh + s4 Wo":[A[0],A[1],B[2]], "s3 Wih+Wo + s4 Whh":[A[0],B[1],A[2]], "s3 Whh+Wo + s4 Wih":[B[0],A[1],A[2]]}
which=sys.argv[1]; W=conds[which]
for ds in [81,82,83,84]:
    m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(W[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(W[1],dtype=th.float32),'linear.weight':th.tensor(W[2],dtype=th.float32)})
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(ds)
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    bp=best_pointer_sel(m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float))
    print(f"{which} | data {ds}: pointer {bp[0]:.2f} switch={bp[0]<0.4}",flush=True)
