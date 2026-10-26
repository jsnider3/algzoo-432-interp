import sys, time, json, numpy as np, torch as th
from alg_zoo import DistRNN
from k1 import clean_err, best_pointer_sel
th.set_num_threads(1)
for init_seed in [int(a) for a in sys.argv[1].split(',')]:
    th.manual_seed(1000+init_seed); m=DistRNN(hidden_size=16,seq_len=10)
    Wih0=m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float).copy(); Whh0=m.rnn.weight_hh_l0.detach().numpy().astype(float).copy(); Wo0=m.linear.weight.detach().numpy().astype(float).copy()
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); n_steps=2**20
    th.manual_seed(5000+init_seed); t0=time.time()
    for s in range(2**15):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/n_steps),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
    a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
    rec=dict(init=init_seed,acc=a,clean=ce,pointer=bp[0],switch=bool(bp[0]<0.4),
             n_pos_in=int((Wih0>0).sum()), max_in=float(Wih0.max()), min_in=float(Wih0.min()), mean_abs_in=float(np.abs(Wih0).mean()),
             spec=float(np.max(np.abs(np.linalg.eigvals(Whh0)))), diag_mean=float(np.diag(Whh0).mean()),
             pos_in_self=float(np.diag(Whh0)[Wih0>0].mean()) if (Wih0>0).any() else 0.0, max_in_x_self=float((Wih0*np.diag(Whh0)).max()))
    print(json.dumps(rec), flush=True)
