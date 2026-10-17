from common import *
x = sample(400000, seed=4); lg,_ = run(x); pm=lg.argmax(1); N=len(x)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1)
rng=np.random.default_rng(9)
def resample_below(x, k):
    """keep top-k values in place; replace others with fresh N(0,1) values conditioned < k-th largest"""
    x2=x.copy(); thr=Xr[:,k-1:k]
    mask=x<thr
    new=rng.standard_normal(x.shape)
    bad=mask&(new>=thr)
    while bad.any():
        new[bad]=rng.standard_normal(bad.sum()); bad=mask&(new>=thr)
    x2[mask]=new[mask]; return x2
wrong=pm!=o[:,1]
for k in [3,4,5]:
    x2=resample_below(x,k); p2=run(x2)[0].argmax(1)
    print(f"keep top-{k}: output unchanged overall {(p2==pm).mean():.4f}; among model errors {(p2==pm)[wrong].mean():.3f}; among correct {(p2==pm)[~wrong].mean():.4f}")
