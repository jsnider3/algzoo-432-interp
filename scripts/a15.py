from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(300000); lg, hs = run(x)
o=np.argsort(-x,1)  # rank order desc
L = lg - lg.mean(1,keepdims=True)
Lr = np.take_along_axis(L,o,1); Xr=np.take_along_axis(x,o,1)
print("mean centered logit by rank (1=max):", Lr.mean(0))
print("std:", Lr.std(0))
# within-sample: among non-max positions, corr of logit with x
c=[]
for i in range(2000):
    m=np.ones(10,bool); m[o[i,0]]=False; c.append(np.corrcoef(lg[i,m],x[i,m])[0,1])
print("median within-sample corr(logit, x) over non-max positions:", np.median(c))
# logit for rank r as function of value: fit logit_rank_r ≈ a*x_r + b*M + c*S
for r in range(1,6):
    A=np.c_[Xr[:,r],Xr[:,0],Xr[:,1],np.ones(len(x))]; cf,*_=np.linalg.lstsq(A,Lr[:,r],rcond=None)
    r2=1-((A@cf-Lr[:,r])**2).mean()/Lr[:,r].var()
    print(f"rank {r+1}: logit ≈ {cf[0]:.2f}*x_self + {cf[1]:.2f}*M + {cf[2]:.2f}*S + {cf[3]:.2f}  R2={r2:.2f}")
