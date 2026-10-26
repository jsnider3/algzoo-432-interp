import sys, json, numpy as np, torch as th
from alg_zoo import DistRNN
from seeds_clocks import run, T
from k1 import best_pointer_sel
th.set_num_threads(1)
xd=np.random.default_rng(701).standard_normal((60000,T))
def decod(Wih,Whh):
    hs=run(xd,Wih,Whh); out=[]
    for which in [0,1]:
        H=np.concatenate([hs[:,t] for t in range(2,T)]); Y=np.concatenate([(-np.sort(-xd[:,:t+1],1))[:,which] for t in range(2,T)])
        A=np.c_[H,np.ones(len(H))]; c,*_=np.linalg.lstsq(A,Y,rcond=None); out.append(1-np.mean((A@c-Y)**2)/Y.var())
    return out
for init in [int(a) for a in sys.argv[1].split(',')]:
    th.manual_seed(4000+init); m=DistRNN(hidden_size=16,seq_len=10)
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(9000+init); rec={'init':init}
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
        if s+1 in (2048,4096):
            W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float)]
            dm,ds=decod(*W); rec[f"M@{s+1}"]=round(dm,3); rec[f"S@{s+1}"]=round(ds,3)
    W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float)]
    bp=best_pointer_sel(*W); rec['switch']=bool(bp[0]<0.4); print(json.dumps(rec),flush=True)
