import numpy as np, torch as th
from seeds_clocks import load, run, T
th.manual_seed(0)
xtr=np.random.default_rng(161).standard_normal((400000,T)); xte=np.random.default_rng(162).standard_normal((200000,T))
ytr=np.argsort(xtr,1)[:,-2]; yte=np.argsort(xte,1)[:,-2]
def fit_readout(Wih,Whh,iters=1500):
    Htr=th.tensor(run(xtr,Wih,Whh)[:,-1],dtype=th.float32); Hte=th.tensor(run(xte,Wih,Whh)[:,-1],dtype=th.float32)
    W=th.zeros(10,16,requires_grad=True); opt=th.optim.LBFGS([W],max_iter=iters,line_search_fn='strong_wolfe')
    y=th.tensor(ytr)
    def clo():
        opt.zero_grad(); l=th.nn.functional.cross_entropy(Htr@W.T,y); l.backward(); return l
    opt.step(clo)
    W=W.detach().numpy(); acc=((Hte.numpy()@W.T).argmax(1)==yte).mean(); return W,acc
cos=lambda a,b: ((a-a.mean(0))*(b-b.mean(0))).sum()/np.linalg.norm(a-a.mean(0))/np.linalg.norm(b-b.mean(0))
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); acc_tr=((run(xte,Wih,Whh)[:,-1]@Wo.T).argmax(1)==yte).mean()
    W,acc=fit_readout(Wih,Whh); 
    agree=((run(xte,Wih,Whh)[:,-1]@W.T).argmax(1)==(run(xte,Wih,Whh)[:,-1]@Wo.T).argmax(1)).mean()
    print(f"{name}: trained readout acc {acc_tr:.4f} | readout refit from scratch, recurrence fixed: acc {acc:.4f}, cosine to trained Wo {cos(W,Wo):.3f}, output agreement {agree:.3f}")
print("\nseed 2 across its switch: recurrence from checkpoint A + optimally re-fit readout")
for e in [23,24,25,26,30]:
    Wih,Whh,Wo=load(f"ckpt/s2_e{e}.npz"); acc_tr=((run(xte,Wih,Whh)[:,-1]@Wo.T).argmax(1)==yte).mean(); W,acc=fit_readout(Wih,Whh)
    print(f"  2^{e}: own readout {acc_tr:.4f} | optimal readout on this recurrence {acc:.4f}")
