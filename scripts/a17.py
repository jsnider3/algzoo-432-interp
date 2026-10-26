from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(1000000); lg, hs = run(x); pm=lg.argmax(1)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1); p1,p2,p3=o[:,0],o[:,1],o[:,2]
ok=pm==p2; g=Xr[:,1]-Xr[:,2]; gm=Xr[:,0]-Xr[:,1]
bins=[0,.02,.05,.1,.2,.3,.5,1,10]
print("err rate by gap S-R3 (rows) split by order of 2nd/3rd")
for lo,hi in zip(bins[:-1],bins[1:]):
    s=(g>=lo)&(g<hi)
    print(f" [{lo},{hi}) n={s.sum():7d} err={1-ok[s].mean():.3f} | 3rd after 2nd: {1-ok[s&(p3>p2)].mean():.3f}  3rd before 2nd: {1-ok[s&(p3<p2)].mean():.3f}  | pred=3rd: {(pm[s]==p3[s]).mean():.3f} pred=max {(pm[s]==p1[s]).mean():.3f}")
print("err rate by gap M-S")
for lo,hi in zip(bins[:-1],bins[1:]):
    s=(gm>=lo)&(gm<hi); print(f" [{lo},{hi}) n={s.sum():7d} err={1-ok[s].mean():.3f}  pred=max {(pm[s]==p1[s]).mean():.3f}")
# errors with large gaps both sides
s=(g>.3)&(gm>.3); print("err when both gaps >.3:", 1-ok[s].mean(), " n", s.sum())
w=s&~ok
print(" those errors: pred position minus p2 dist:", np.bincount(pm[w]-p2[w]+9,minlength=19))
print(" those errors: rank of predicted:", np.bincount((np.argsort(o,1)[np.arange(len(x)),pm])[w],minlength=10))
print(" p1,p2 positions of those errors (count matrix):\n", np.histogram2d(p1[w],p2[w],bins=10,range=[[0,10],[0,10]])[0].astype(int))
