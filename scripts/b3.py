from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(300000, seed=2); lg, hs, ms, J = run_jac(x); pm=lg.argmax(1); N=len(x)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1); ages=9-o; p1,p2,p3=o[:,0],o[:,1],o[:,2]
n=np.arange(N)
D = lg[n,p2]-lg[n,p3]
dD = J[n,p2,:]-J[n,p3,:]          # gradient of D wrt x
slope = (dD[n,p2]-dD[n,p3])/2
g = Xr[:,1]-Xr[:,2]
gstar = g - D/slope              # gap at which tie would occur (local linear)
near = g<0.15
print("slope>0 frac (near):", (slope[near]>0).mean())
print("g* quantiles (near ties):", np.quantile(gstar[near],[.05,.25,.5,.75,.95]).round(3))
# relative to M
print("g*/M quantiles:", np.quantile((gstar/Xr[:,0])[near],[.05,.25,.5,.75,.95]).round(3))
# dependence on ages: median g* by (aS,a3)
tab=np.full((10,10),np.nan)
for a in range(10):
    for b in range(10):
        s=near&(ages[:,1]==a)&(ages[:,2]==b)
        if s.sum()>200: tab[a,b]=np.median(gstar[s])
print("median g* by ageS (rows) x age3 (cols):\n", tab)
# what else explains g*? regress on M, S, x9, R4 within near
A=np.c_[Xr[:,0],Xr[:,1],Xr[:,3],x[:,9],np.ones(N)][near]
c,*_=np.linalg.lstsq(A,gstar[near],rcond=None); r=gstar[near]-A@c
print("lin fit g* ~ M,S,R4,x9,1:", c.round(3), "R2", 1-r.var()/gstar[near].var())
