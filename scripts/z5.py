"""Mean field across a population: how much of each network's accuracy is carried by its function of the top-3
values and positions alone, and whether evaluating that function beats sampling at equal cost.

Mean field (REPORT 7.13): the 7 lower inputs are replaced by E[x | x < R] = -phi(R)/Phi(R); the top-3 values and
their positions are kept. Measured per network on a shared pool, paired with the true inputs:
  ceiling   accuracy(mean field) - accuracy(true): its mean and SD across networks, and how much of it network
            features predict (the leakage part a mean-field circuit model would have to model separately)
  cost      per-network RMSE against the 2^18-pool accuracy (z2) at K forward passes, for plain sampling, ordered
            Sobol (exact top-3 transport, w1.py), and mean field on the same Sobol top-3 design; each raw and after
            one affine calibration fitted on half the networks and scored on the other half
Usage: python scripts/z5.py pop1k/snap_2e28.npz [pop1k/snap_2e28.roles.npz]
"""
import sys, numpy as np, torch as th
from scipy.stats import norm, qmc
from batch_train import forward, target
from z2 import load
from z3 import cv_rmse, roles, bag
from w1 import transport

def meanfield(x):
    o=np.argsort(-x,1); R=np.take_along_axis(x,o[:,2:3],1)[:,0]; mu=-norm.pdf(R)/norm.cdf(R)
    xm=np.repeat(mu[:,None],10,1); np.put_along_axis(xm,o[:,:3],np.take_along_axis(x,o[:,:3],1),1); return xm

@th.no_grad()
def accuracy(P, x, chunk=8192):
    N=P[0].shape[0]; xt=th.tensor(x,dtype=th.float32); out=np.zeros(N)
    for i in range(0,len(x),chunk):
        xc=xt[i:i+chunk]; out+=(forward(*P,xc[None].expand(N,-1,-1)).argmax(-1)==target(xc)[None]).float().sum(1).numpy()
    return out/len(x)

if __name__=="__main__":
    th.set_num_threads(6); P,s=load(sys.argv[1]); N=P[0].shape[0]
    lab=sys.argv[2] if len(sys.argv)>2 else sys.argv[1].replace(".npz",".roles.npz")
    z=np.load(lab); truth=z["acc"]; F,nm=z["feats"],list(z["names"])
    x=np.random.default_rng(21).standard_normal((2**16,10)).astype(np.float32)
    at=accuracy(P,x); am=accuracy(P,meanfield(x)); d=(am-at)*100
    print(f"{N} networks (s{s}); accuracy {at.mean():.4f} +- {at.std()*100:.2f} points; mean field {am.mean():.4f}, corr {np.corrcoef(at,am)[0,1]:.4f}")
    print(f"mean field - true (points): mean {d.mean():+.3f}, SD {d.std():.3f}, p5/p50/p95 {np.percentile(d,[5,50,95]).round(2)}")
    A=np.c_[np.ones(N),am]; c=np.linalg.lstsq(A,at,rcond=None)[0]; print(f"  after affine calibration: RMSE {np.sqrt(np.mean((A@c-at)**2))*100:.3f} points")
    Wi,Wh,Wo=(p.numpy() for p in P)
    Wf=np.concatenate([np.sort(Wi,1),np.sort(np.diagonal(Wh,0,1,2),1),np.sort(np.abs(np.linalg.eigvals(Wh)),1),np.sort(np.linalg.norm(Wo,axis=1),1)],1)
    for fn,X in [("weights",Wf),("roles + means",bag(roles(F,nm),F,8)),("sorted z2 features + MF accuracy",np.c_[np.sort(F,1).reshape(N,-1),am])]:
        print(f"  leakage predicted from {fn:34s} CV RMSE "+"  ".join(f"{k} {v*100:.3f}" for k,v in cv_rmse(X,d/100).items()))
    half=np.random.default_rng(0).permutation(N)<N//2; p=truth.mean()
    def score(est):
        A=np.c_[np.ones(N),est]; c=np.linalg.lstsq(A[half],truth[half],rcond=None)[0]
        return np.sqrt(np.mean((est-truth)[~half]**2))*100, np.sqrt(np.mean((A[~half]@c-truth[~half])**2))*100
    print("\ncost: per-network RMSE (points) against the 2^18-pool accuracy, raw / calibrated")
    for m in (8,10,12):
        K=2**m; u=qmc.Sobol(11,scramble=True,seed=m).random_base2(m); xs=transport(u,10)
        res={"plain sampling":score(accuracy(P,np.random.default_rng(100+m).standard_normal((K,10)))),
             "ordered Sobol":score(accuracy(P,xs)),"mean field, Sobol top-3":score(accuracy(P,meanfield(xs)))}
        print(f"  K={K:5d} (sampling SE {np.sqrt(p*(1-p)/K)*100:.2f}): "+"   ".join(f"{k} {a:.3f}/{b:.3f}" for k,(a,b) in res.items()))
