from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(200000); lg, hs = run(x)
rows=[]
for t in range(3,T):
    P=x[:,:t]; srt=np.sort(P,1); M=srt[:,-1]; S=srt[:,-2]; xt=x[:,t]
    Mz=np.maximum(M,0)
    cand={'relu(x-M)':np.maximum(xt-M,0),'relu(M-x)':np.maximum(M-xt,0),'relu(x-S)':np.maximum(xt-S,0),'relu(S-x)':np.maximum(S-xt,0),
          'relu(min(x,M)-S)':np.maximum(np.minimum(xt,M)-S,0),'M':M,'S':S,'x':xt,'newS?':((xt>S)&(xt<M)).astype(float),'newM?':(xt>M).astype(float)}
    rows.append((t,cand,hs[:,t]))
print("corr(neuron, candidate) averaged over t=3..9")
names=list(rows[0][1])
C=np.zeros((H,len(names)))
for t,cand,h in rows:
    for j,k in enumerate(names):
        for n in range(H): C[n,j]+=np.corrcoef(h[:,n],cand[k])[0,1]/len(rows)
print("        "+" ".join(f"{k[:10]:>10}" for k in names))
for n in range(H): print(f"n{n:2d}    "+" ".join(f"{v:10.2f}" for v in C[n]))
