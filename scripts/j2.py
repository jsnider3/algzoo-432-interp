import numpy as np, torch as th
from seeds_clocks import load, run, T
x=np.random.default_rng(171).standard_normal((300000,T)); y=th.tensor(np.argsort(x,1)[:,-2]); X=th.tensor(x,dtype=th.float64)
def loss_fn(Wih,Whh,Wo):
    h=th.zeros(len(X),16,dtype=th.float64)
    for t in range(T): h=th.relu(X[:,t:t+1]*Wih+h@Whh.T)
    return th.nn.functional.cross_entropy(h@Wo.T,y)
def selectivity(Wih,Whh,n):
    hs=run(x[:100000],Wih,Whh); r={'new max':[],'new 2nd':[],'neither':[]}
    for t in range(3,T):
        P=x[:100000,:t]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]; xe=x[:100000,t]; f=hs[:,t,n]>0
        r['new max'].append(f[xe>M].mean()); r['new 2nd'].append(f[(xe>S)&(xe<M)].mean()); r['neither'].append(f[xe<S].mean())
    return {k:round(float(np.mean(v)),3) for k,v in r.items()}
if __name__=="__main__":
    for e in [22,23,24,25]:
        Wih,Whh,Wo=[th.tensor(a,dtype=th.float64,requires_grad=True) for a in load(f"ckpt/s2_e{e}.npz")]
        L=loss_fn(Wih,Whh,Wo); L.backward()
        n=10
        g_in=Wih.grad[n].item(); g_rec=Whh.grad[n].numpy()
        print(f"\nseed 2 @2^{e}: loss {L.item():.4f}; n10 selectivity {selectivity(Wih.detach().numpy(),Whh.detach().numpy(),n)}")
        g=np.r_[g_in,g_rec]; w=np.r_[Wih.detach().numpy()[n],Whh.detach().numpy()[n]]
        print(f"   cos(-grad, current n10 input weights) {(-g@w)/np.linalg.norm(g)/np.linalg.norm(w):+.3f}")
        for rel in [0.01,0.03,0.1]:
            Wi=Wih.detach().numpy().copy(); Wh=Whh.detach().numpy().copy(); step=rel*np.linalg.norm(w)/np.linalg.norm(g)
            Wi[n]-=step*g_in; Wh[n]-=step*g_rec     # step only on n10's incoming weights, size rel*|w|
            with th.no_grad(): L2=loss_fn(th.tensor(Wi),th.tensor(Wh),Wo.detach()).item()
            print(f"   gradient step on n10 inputs (step {rel:.0%} of |w|): loss {L2:.4f} ({L2-L.item():+.4f}); n10 selectivity -> {selectivity(Wi,Wh,n)}")
