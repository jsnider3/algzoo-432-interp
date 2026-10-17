from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(500000); lg, hs = run(x)
for t in [9]:
    P=x[:,:t+1]; o=np.argsort(-P,1); srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; aM=t-o[:,0]; aS=t-o[:,1]
    h=hs[:,t]; pos=M>0.5
    print("mean h_n / M  (rows ageM 0..9, cols neurons 0..15), M>0.5")
    for a in range(t+1):
        s=pos&(aM==a); print(a, (h[s]/M[s,None]).mean(0))
    print("mean h_n / M by ageS (rows), restricted to ageM >= 5 ")
    for a in range(t+1):
        s=pos&(aS==a)&(aM>=5); 
        if s.sum()>100: print(a, (h[s]/M[s,None]).mean(0))
