import torch as th
from common import *
x = sample(200000); lg, hs = run(x)
def probe(Hm, y, k):
    Hm=th.tensor(Hm,dtype=th.float32); y=th.tensor(y)
    mu,sd=Hm[:150000].mean(0),Hm[:150000].std(0)+1e-6; Hm=(Hm-mu)/sd
    lin=th.nn.Linear(Hm.shape[1],k); opt=th.optim.Adam(lin.parameters(),lr=0.05)
    for i in range(400):
        opt.zero_grad(); l=th.nn.functional.cross_entropy(lin(Hm[:150000]),y[:150000]); l.backward(); opt.step()
    return (lin(Hm[150000:]).argmax(1)==y[150000:]).float().mean().item()
for t in [5,9]:
    X=x[:,:t+1]; o=np.argsort(X,1)
    ageM=t-o[:,-1]; ageS=t-o[:,-2]; age3=t-o[:,-3]
    print(t, "linear probe acc ageM %.3f ageS %.3f age3rd %.3f" % (probe(hs[:,t],ageM,t+1), probe(hs[:,t],ageS,t+1), probe(hs[:,t],age3,t+1)))
    print("  chance (most common):", np.bincount(ageM).max()/len(x))
