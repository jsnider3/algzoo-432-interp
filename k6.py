import numpy as np, torch as th
from k1 import finetune, best_pointer_sel, load, clean_err
from seeds_clocks import run, T
th.set_num_threads(4)
Wf=finetune("ckpt/s0_e22.npz",21,steps=3*2**12,start_step=2**12)   # ~2^22 + 3*2^22 seqs = 2^24 total
Wo_=load("ckpt/s0_e24.npz")
print("fork pointer:", best_pointer_sel(Wf[0],Wf[1])[0].round(2), " original 2^24 pointer:", best_pointer_sel(Wo_[0],Wo_[1])[0].round(2))
x=np.random.default_rng(301).standard_normal((200000,T)); y=np.argsort(x,1)[:,-2]
def loss(W):
    L=run(x,W[0],W[1])[:,-1]@W[2].T; L=L-L.max(1,keepdims=True); return -np.mean(L[np.arange(len(x)),y]-np.log(np.exp(L).sum(1))), (L.argmax(1)==y).mean()
for a in np.linspace(0,1,11):
    W=[(1-a)*p+a*q for p,q in zip(Wo_,Wf)]; l,ac=loss(W)
    print(f"  alpha {a:.1f} (0=original memory 2^24, 1=switched fork): loss {l:.3f} acc {ac:.3f} pointer {best_pointer_sel(W[0],W[1])[0]:.2f}")
