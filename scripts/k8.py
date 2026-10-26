import numpy as np, torch as th, time
from k1 import finetune, load, best_pointer_sel
from seeds_clocks import run, T
th.set_num_threads(2); th.manual_seed(0)
Wf=finetune("ckpt/s0_e22.npz",21,steps=3*2**12,start_step=2**12); np.savez("fork_s0.npz",*Wf)
W0=load("ckpt/s0_e24.npz")
A=[th.tensor(a,dtype=th.float32) for a in W0]; B=[th.tensor(b,dtype=th.float32) for b in Wf]
phi=[((a+b)/2).clone().requires_grad_(True) for a,b in zip(A,B)]
def model_loss(Wih,Whh,Wo,x):
    h=th.zeros(len(x),16)
    for t in range(T): h=th.relu(x[:,t:t+1]*Wih+h@Whh.T)
    return th.nn.functional.cross_entropy(h@Wo.T, x.argsort(-1)[:,-2])
def curve(t): return [(1-t)**2*a+2*t*(1-t)*p+t**2*b for a,p,b in zip(A,phi,B)]
opt=th.optim.Adam(phi,lr=3e-3)
t0=time.time()
for it in range(6000):
    x=th.randn(2048,10); t=th.rand(()).item()
    l=model_loss(*curve(t),x); opt.zero_grad(); l.backward(); opt.step()
    if it%2000==1999: print(f"  iter {it+1}: sampled loss {l.item():.3f} [{time.time()-t0:.0f}s]",flush=True)
x=np.random.default_rng(301).standard_normal((200000,T)); y=np.argsort(x,1)[:,-2]
def ev(W):
    W=[w.detach().numpy().astype(float) for w in W]; L=run(x,W[0],W[1])[:,-1]@W[2].T; L=L-L.max(1,keepdims=True)
    return -np.mean(L[np.arange(len(x)),y]-np.log(np.exp(L).sum(1))), (L.argmax(1)==y).mean(), best_pointer_sel(W[0],W[1])[0]
print("Optimized Bezier path (0 = memory 2^24, 1 = switched fork):")
for t in np.linspace(0,1,11):
    l,a,p=ev(curve(float(t))); print(f"  t {t:.1f}: loss {l:.3f} acc {a:.3f} pointer {p:.2f}")
