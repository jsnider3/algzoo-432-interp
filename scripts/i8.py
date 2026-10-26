import numpy as np
from seeds_clocks import load, run, T
import i7
from i7 import strat, W, alloc, draw, K
rng=np.random.default_rng(121)
Wih0,Whh0,Wo0=load("weights.npz")
res=[]
for sc,trial in [(s_,t_) for s_ in (0.003,0.01,0.03) for t_ in range(4)]:
    Wih=Wih0*(1+sc*rng.standard_normal(Wih0.shape)); Whh=Whh0+sc*np.abs(Whh0).mean()*rng.standard_normal(Whh0.shape); Wo=Wo0*(1+sc*rng.standard_normal(Wo0.shape))
    f=lambda x:(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2]
    xt=np.random.default_rng(5).standard_normal((2000000,T)); true=np.mean(f(xt))
    budget=4000
    nv=[np.mean(f(rng.standard_normal((budget,T)))) for _ in range(100)]
    st=[sum(W[h]*np.mean(f(draw(h,max(3,int(round(alloc[h]*budget)))))) for h in range(K)) for _ in range(100)]
    mn=np.mean((np.array(nv)-true)**2); ms=np.mean((np.array(st)-true)**2); res.append((sc,mn/ms))
    print(f"noise {sc}: model {trial}: true acc {true:.4f} | naive MSE {mn:.2e} | stratified MSE {ms:.2e} -> {mn/ms:.1f}x")
for sc in sorted({s_ for s_,_ in res}):
    print(f"noise {sc}: geometric-mean gain", np.exp(np.mean(np.log([g for s_,g in res if s_==sc]))).round(2))
