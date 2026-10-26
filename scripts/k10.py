import numpy as np, itertools
from scipy.stats import norm, truncnorm
from seeds_clocks import load
from k9 import logits
Wih,Whh,Wo=load("weights.npz"); rng=np.random.default_rng(0)
perms=np.array(list(itertools.permutations(range(10),3)))
# random cells: draw top-3 from true distribution via sorting real samples
z=rng.standard_normal((3000,10)); s=-np.sort(-z,1); M,S,R=s[:,0],s[:,1],s[:,2]
pi=perms[rng.integers(0,720,3000)]
mu=-norm.pdf(R)/norm.cdf(R)
X=np.repeat(mu[:,None],10,1); n=np.arange(3000)
X[n,pi[:,0]]=M; X[n,pi[:,1]]=S; X[n,pi[:,2]]=R
low=np.ones((3000,10),bool); low[n,pi[:,0]]=low[n,pi[:,1]]=low[n,pi[:,2]]=False
P2=pi[:,1]
L0=logits(X,Wih,Whh,Wo); mf=(L0.argmax(1)==P2)
# Monte Carlo truth over lower inputs
mc=np.zeros(3000)
for rep in range(200):
    Xr=X.copy(); zz=truncnorm.rvs(-np.inf,R[:,None].repeat(10,1),size=(3000,10),random_state=rng); Xr[low]=zz[low]
    mc+=(logits(Xr,Wih,Whh,Wo).argmax(1)==P2)/200
print("mean-field P(correct) avg", mf.mean().round(4), "| Monte-Carlo-over-lower-inputs avg", mc.mean().round(4))
# where do they differ?
d=mc-mf
print("fraction of cells: MF=1 & MC<0.5:", np.mean(mf&(mc<0.5)).round(4), "  MF=0 & MC>0.5:", np.mean(~mf&(mc>0.5)).round(4))
print("among MF-correct cells, mean MC:", mc[mf].mean().round(4))
