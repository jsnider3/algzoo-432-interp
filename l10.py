import sys, numpy as np, torch as th
from alg_zoo import DistRNN
from seeds_clocks import load
from k1 import best_pointer_sel
th.set_num_threads(1)
A=load("ckpt/s3_e0.npz"); B=load("ckpt/s4_e0.npz")
def mix(idx):
    Whh=B[1].copy(); Wo=B[2].copy()
    if idx is None: Whh=A[1].copy(); Wo=A[2].copy()
    else:
        for j in idx: Whh[j,:]=A[1][j,:]; Whh[:,j]=A[1][:,j]; Wo[:,j]=A[2][:,j]
    return [B[0],Whh,Wo]
which=sys.argv[1]; idx={"all":None,"half0-7":list(range(0,8)),"half8-15":list(range(8,16)),"only13":[13],"all_but13":[j for j in range(16) if j!=13]}[which]
W=mix(idx); seeds=[91,92] if which=="all" else [91,92,93,94]
for ds in seeds:
    m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(W[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(W[1],dtype=th.float32),'linear.weight':th.tensor(W[2],dtype=th.float32)})
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(ds)
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    bp=best_pointer_sel(m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float))
    print(f"s3 Whh+Wo on neurons [{which}] + s4 rest | data {ds}: pointer {bp[0]:.2f} neuron {bp[1][0] if bp[1] else None} delay {bp[1][1] if bp[1] else None} switch={bp[0]<0.4}",flush=True)
