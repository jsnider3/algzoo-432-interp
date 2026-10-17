import numpy as np
from seeds_clocks import load, run, T
# strata defined from the mechanism (no model needed to assign): S<0 ; normalized near-ties ; clean
def stratum(x):
    srt=-np.sort(-x,1); M,S,R=srt[:,0],srt[:,1],srt[:,2]; d=np.minimum(M-S,S-R)/np.abs(M)
    return np.where(S<0,0,np.where(d<0.02,1,np.where(d<0.05,2,np.where(d<0.1,3,np.where(d<0.2,4,5)))))
K=6
rng=np.random.default_rng(0)
# stratum weights from a huge model-free sample (cheap: just sorting)
W=np.bincount(stratum(rng.standard_normal((20_000_000,T))),minlength=K)/20_000_000
print("stratum weights:", W.round(4))
def gen_stratum(h,n,rng):
    out=[]; got=0
    while got<n:
        x=rng.standard_normal((max(4*n,10000),T)); x=x[stratum(x)==h]; out.append(x); got+=len(x)
    return np.concatenate(out)[:n]
for name,path in [("mystery","weights.npz"),("seed 0","weights_seed0.npz"),("seed 2","weights_seed2.npz")]:
    Wih,Whh,Wo=load(path)
    acc_fn=lambda x: ((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2])
    # ground truth & per-stratum error sd (pilot)
    true=np.mean(acc_fn(np.random.default_rng(99).standard_normal((3_000_000,T))))
    pil=[acc_fn(gen_stratum(h,40000,rng)).mean() for h in range(K)]
    sd=np.sqrt(np.array(pil)*(1-np.array(pil)))
    alloc=W*sd/np.sum(W*sd)
    budget=2000; reps=300
    naive=[acc_fn(rng.standard_normal((budget,T))).mean() for _ in range(reps)]
    strat=[]
    for _ in range(reps):
        est=0
        for h in range(K):
            n=max(2,int(round(alloc[h]*budget))); est+=W[h]*acc_fn(gen_stratum(h,n,rng)).mean()
        strat.append(est)
    mn=np.mean((np.array(naive)-true)**2); ms=np.mean((np.array(strat)-true)**2)
    print(f"{name:8s} true acc {true:.4f} | per-stratum acc {np.round(pil,3)} | MSE @ {budget} model evals: naive {mn:.2e}, mechanism-stratified {ms:.2e}  -> {mn/ms:.1f}x lower")
