import sys, json, numpy as np, torch as th
from alg_zoo import DistRNN
from k1 import best_pointer_sel
th.set_num_threads(1)
for init_seed in [int(a) for a in sys.argv[1].split(',')]:
    th.manual_seed(2000+init_seed); m=DistRNN(hidden_size=16,seq_len=10)
    snap={}
    def grab(): return m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float).copy(), np.diag(m.rnn.weight_hh_l0.detach().numpy()).astype(float).copy()
    snap[0]=grab()
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(7000+init_seed)
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
        if s+1 in (256,1024,4096): snap[s+1]=grab()
    Wih=m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float); Whh=m.rnn.weight_hh_l0.detach().numpy().astype(float)
    bp=best_pointer_sel(Wih,Whh); sw=bool(bp[0]<0.4); n=bp[1][0] if bp[1] else -1
    rank=lambda v,i: int((v>v[i]).sum())   # 0 = largest
    rec=dict(init=init_seed,switch=sw,pointer=n,final_in=float(Wih[n]),
             **{f"in_rank@{k}":rank(v[0],n) for k,v in snap.items()},
             **{f"self_rank@{k}":rank(v[1],n) for k,v in snap.items()},
             growth_rank_0_1024=rank(snap[1024][0]-snap[0][0],n), growth_rank_0_4096=rank(snap[4096][0]-snap[0][0],n))
    print(json.dumps(rec),flush=True)
