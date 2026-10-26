from common import *
x=sample(1500000,seed=12); pm=run(x)[0].argmax(1)
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]; err=pm!=o[:,1]
r=(M-S)/np.abs(S).clip(0.05)
print("error rate by M-S gap (cols) and max age (rows); only cases where 2nd max came AFTER max (so its admission test ran while max was old)")
bins=[0,.3,.6,1,1.5,10]
print("ageM  "+"  ".join(f"[{a},{b})" for a,b in zip(bins[:-1],bins[1:])))
for a in range(1,10):
    row=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        s=(aM==a)&(aS<aM)&(M-S>=lo)&(M-S<hi); row.append(f"{err[s].mean():.3f}" if s.sum()>500 else "  -  ")
    print(f"{a:4d}  "+"   ".join(row))
print("same for 2nd max BEFORE max (S was displaced old max; no admission test vs S needed)")
for a in [1,3,5,7]:
    row=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        s=(aM==a)&(aS>aM)&(M-S>=lo)&(M-S<hi); row.append(f"{err[s].mean():.3f}" if s.sum()>500 else "  -  ")
    print(f"{a:4d}  "+"   ".join(row))
