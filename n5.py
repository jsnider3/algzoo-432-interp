import numpy as np, itertools
from scipy.stats import norm, truncnorm
from seeds_clocks import load
from j5 import gl
from l3 import logits
perms=np.array(list(itertools.permutations(range(10),3)))
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz")]:
    Wih,Whh,Wo=load(path); rng=np.random.default_rng(0); K=3000
    z=rng.standard_normal((K,10)); s=-np.sort(-z,1); M,S,R=s[:,0],s[:,1],s[:,2]; pi=perms[rng.integers(0,720,K)]
    mu=-norm.pdf(R)/norm.cdf(R); n=np.arange(K)
    X=np.repeat(mu[:,None],10,1); X[n,pi[:,0]]=M; X[n,pi[:,1]]=S; X[n,pi[:,2]]=R
    low=np.ones((K,10),bool); low[n,pi[:,0]]=low[n,pi[:,1]]=low[n,pi[:,2]]=False; li=np.where(low)[1].reshape(K,7); P2=pi[:,1]
    I=lambda X_: (logits(X_,Wih,Whh,Wo).argmax(1)==P2).astype(float)
    nodes=lambda q: norm.ppf(np.clip(q*norm.cdf(R),1e-12,1-1e-12))
    I0=I(X)
    mc=np.zeros(K)
    for rep in range(300):
        Xr=X.copy(); zz=truncnorm.rvs(-np.inf,R[:,None].repeat(10,1),size=(K,10),random_state=rng); Xr[low]=zz[low]; mc+=I(Xr)/300
    for nq in [8]:
        qu,qw=gl(0,1,nq)
        E1=np.zeros((K,7))
        for k in range(7):
            for q,w in zip(qu,qw):
                Xk=X.copy(); Xk[n,li[:,k]]=nodes(q); E1[:,k]+=w*I(Xk)
        h1=I0+np.sum(E1-I0[:,None],1)
        E2=np.zeros(K)
        for a in range(7):
            for b in range(a+1,7):
                Eab=np.zeros(K)
                for q1,w1 in zip(qu,qw):
                    for q2,w2 in zip(qu,qw):
                        Xk=X.copy(); Xk[n,li[:,a]]=nodes(q1); Xk[n,li[:,b]]=nodes(q2); Eab+=w1*w2*I(Xk)
                E2+=Eab-E1[:,a]-E1[:,b]+I0
        h2=h1+E2
        c=lambda v: np.clip(v,0,1)
        print(f"{name}: MC {mc.mean():.4f} | MF {I0.mean():.4f} ({I0.mean()-mc.mean():+.4f}) | 1st order (nq={nq}) {c(h1).mean():.4f} ({c(h1).mean()-mc.mean():+.4f}) | 2nd order (nq={nq}) {c(h2).mean():.4f} ({c(h2).mean()-mc.mean():+.4f}) | per-cell RMSE MF/1st/2nd {np.sqrt(np.mean((I0-mc)**2)):.3f}/{np.sqrt(np.mean((c(h1)-mc)**2)):.3f}/{np.sqrt(np.mean((c(h2)-mc)**2)):.3f}",flush=True)
