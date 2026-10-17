from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(1000000); lg, hs = run(x); pred=lg.argmax(1)
P=x[:,:9]; o=np.argsort(P,1); srt=np.sort(P,1); M,S=srt[:,-1],srt[:,-2]; aM=8-o[:,-1]; aS=8-o[:,-2]; x9=x[:,9]
tgt=target(x); ok=pred==tgt
print("overall acc", ok.mean())
case = np.where(x9>M,0,np.where(x9>S,1,2))
for c,nm in enumerate(['x9 new max','x9 new 2nd','x9 below S']):
    s=case==c; print(f"{nm}: frac {s.mean():.3f} acc {ok[s].mean():.4f}")
# acc vs x9-S binned, for x9<M
s=x9<M; d=(x9-S)[s]; bins=np.linspace(-1,1,21); ib=np.digitize(d,bins)
print("acc vs (x9 - S) when x9<M:")
for b in range(1,len(bins)):
    m=ib==b; print(f"  [{bins[b-1]:+.1f},{bins[b]:+.1f}) n={m.sum():6d} acc={ok[s][m].mean():.3f}  pred=9 frac={(pred[s][m]==9).mean():.3f}")
# acc for x9 < S by ageM
s2=case==2
print("x9<S: acc by ageM", [round(ok[s2&(aM==a)].mean(),3) for a in range(9)])
print("x9<S: acc by ageS", [round(ok[s2&(aS==a)].mean(),3) for a in range(9)])
print("all x9: acc by ageS(at t=8)", [round(ok[(aS==a)].mean(),3) for a in range(9)])
