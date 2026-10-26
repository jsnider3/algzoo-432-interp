import numpy as np
from scipy.stats import norm
from seeds_clocks import load, run
# REPORT 7.13 (r5.py): sampled, quadrature-free check of the mean-field bias with 3 vs 4 exact order statistics.
# Lower inputs are replaced by their conditional mean below the k-th largest; paired with the true output on the same x.
for name,path in [("seed 0","weights_seed0.npz"),("mystery","weights.npz")]:
    Wih,Whh,Wo=load(path); rng=np.random.default_rng(2024); N=0; sf=0; sk={3:0,4:0}; D={3:[],4:[]}
    for _ in range(16):
        x=rng.standard_normal((250000,10)); o=np.argsort(-x,1); s=-np.sort(-x,1); n=np.arange(len(x)); p2=o[:,1]
        f=(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==p2; sf+=f.sum(); N+=len(x)
        for k in (3,4):
            Rk=s[:,k-1]; mu=-norm.pdf(Rk)/norm.cdf(Rk); xm=x.copy(); low=np.ones_like(x,bool)
            for j in range(k): low[n,o[:,j]]=False
            xm[low]=np.repeat(mu[:,None],10,1)[low]
            g=(run(xm,Wih,Whh)[:,-1]@Wo.T).argmax(1)==p2; sk[k]+=g.sum(); D[k].append(g.astype(float)-f)
    out=[f"{name}: true {sf/N:.5f}"]
    for k in (3,4):
        d=np.concatenate(D[k]); out.append(f"MF{k} {sk[k]/N:.5f} (bias {100*d.mean():+.3f} ± {100*d.std()/np.sqrt(N):.3f} points)")
    print(" | ".join(out),flush=True)
