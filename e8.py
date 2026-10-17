from common import *
x=sample(1500000,seed=13); pm=run(x)[0].argmax(1)
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S,R3=srt[:,0],srt[:,1],srt[:,2]; err=pm!=o[:,1]
q=S/M; g3=(S-R3)/M
print("error by S/M (rows) and (S-R3)/M (cols)")
qb=[-1,0,.2,.4,.6,.8,1.0]; gb=[0,.02,.05,.1,.2,1]
print("S/M        "+" ".join(f"[{a},{b})".rjust(10) for a,b in zip(gb[:-1],gb[1:]))+"   all   n")
for lo,hi in zip(qb[:-1],qb[1:]):
    s=(q>=lo)&(q<hi)
    row=[f"{err[s&(g3>=a)&(g3<b)].mean():.3f}".rjust(10) if (s&(g3>=a)&(g3<b)).sum()>300 else "-".rjust(10) for a,b in zip(gb[:-1],gb[1:])]
    print(f"[{lo:.1f},{hi:.1f})  "+" ".join(row)+f"  {err[s].mean():.3f} {s.sum()}")
