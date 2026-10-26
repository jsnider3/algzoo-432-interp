import numpy as np
from scipy.stats import norm
from seeds_clocks import load, run, T
Wih,Whh,Wo=load("weights.npz"); rng=np.random.default_rng(777); c=0; n=0
for _ in range(40):
    x=rng.standard_normal((500000,T)); o=np.argsort(-x,1); R=-np.sort(-x,1)[:,2]; mu=-norm.pdf(R)/norm.cdf(R)
    xm=x.copy(); low=np.ones_like(x,bool); k=np.arange(len(x)); low[k,o[:,0]]=low[k,o[:,1]]=low[k,o[:,2]]=False
    xm[low]=np.repeat(mu[:,None],10,1)[low]; c+=((run(xm,Wih,Whh)[:,-1]@Wo.T).argmax(1)==o[:,1]).sum(); n+=len(x)
p=c/n; print(f"SAMPLED-MF mystery {p:.5f} ± {np.sqrt(p*(1-p)/n)*100:.4f} pts",flush=True)
