import numpy as np, torch as th
from seeds_clocks import load, run, T
th.set_num_threads(1)
x=np.random.default_rng(501).standard_normal((60000,T)); y=np.argsort(x,1)[:,-2]
xt=np.random.default_rng(502).standard_normal((60000,T)); yt=np.argsort(xt,1)[:,-2]
def refit_eval(Wih,Whh):
    H=th.tensor(run(x,Wih,Whh)[:,-1],dtype=th.float32); Ht=th.tensor(run(xt,Wih,Whh)[:,-1],dtype=th.float32)
    W=th.zeros(10,16,requires_grad=True); opt=th.optim.LBFGS([W],max_iter=300,line_search_fn='strong_wolfe'); yy=th.tensor(y)
    def clo():
        opt.zero_grad(); l=th.nn.functional.cross_entropy(H@W.T,yy); l.backward(); return l
    opt.step(clo)
    with th.no_grad(): L=Ht@W.T; loss=th.nn.functional.cross_entropy(L,th.tensor(yt)).item(); acc=(L.argmax(1).numpy()==yt).mean()
    return loss,acc
def prefix_MS(z):
    out=[]
    for t in range(1,T):
        s=-np.sort(-z[:,:t],1); out.append((s[:,0], s[:,1] if t>1 else np.full(len(z),-3.0)))
    return out
for path in ["ckpt/s2_e22.npz","ckpt/s0_e22.npz","ckpt/s2_e23.npz","ckpt/s0_e23.npz"]:
    Wih,Whh,Wo=load(path); n=int(np.argmax(Wih)); hs=run(x,Wih,Whh); MS=prefix_MS(x)
    Hprev=np.concatenate([hs[:,t-1] for t in range(2,T)]); Mv=np.concatenate([MS[t-1][0] for t in range(2,T)]); Sv=np.concatenate([MS[t-1][1] for t in range(2,T)])
    base=refit_eval(Wih,Whh)
    print(f"\n{path}: leader n{n} (Wih {Wih[n]:+.2f}); original row -> loss {base[0]:.3f} acc {base[1]:.3f} (readout refit)")
    for nm,target in [("always-on (thr=-2M)",-2*Mv)]+[(f"thr={l:.2f}S+{1-l:.2f}M",l*Sv+(1-l)*Mv) for l in [0,0.25,0.5,0.75,1.0]]:
        r,*_=np.linalg.lstsq(Hprev,-target*Wih[n],rcond=None)     # want -(r.h)/Wih = target
        fitR2=1-np.mean((-(Hprev@r)/Wih[n]-target)**2)/target.var()
        Whh2=Whh.copy(); Whh2[n]=r
        loss,acc=refit_eval(Wih,Whh2)
        print(f"   {nm:22s}: row-fit R2 {fitR2:.2f} -> loss {loss:.3f} acc {acc:.3f}  (Δloss {loss-base[0]:+.3f})",flush=True)
