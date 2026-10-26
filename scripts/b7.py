from common import *
x = sample(300000, seed=5); lg,hs=run(x); pm=lg.argmax(1)
c=1/hs[:,-1].std(0)          # rescale each neuron to unit std at final step (exact symmetry of ReLU)
Wih2=Wih*c; Whh2=Whh*c[:,None]/c[None,:]; Wo2=Wo/c[None,:]
print("rescaled exact agreement", (run(x,Wih2,Whh2,Wo2)[0].argmax(1)==pm).mean())
allw=np.r_[np.abs(Wih2),np.abs(Whh2).ravel(),np.abs(Wo2).ravel()]
for tau in [0.003,0.01,0.03,0.1]:
    f=lambda W: np.where(np.abs(W)<tau,0,W)
    p=run(x,f(Wih2),f(Whh2),f(Wo2))[0].argmax(1)
    print(f"prune |w|<{tau}: nonzero {(allw>=tau).sum()}/432  agreement {(p==pm).mean():.4f}  acc {(p==target(x)).mean():.4f}")
# greedy: prune weights one at a time by smallest effect? estimate each recurrent weight's individual effect
eff=[]
for i in range(16):
    for j in range(16):
        W=Whh2.copy(); W[i,j]=0; eff.append(((run(x[:50000],Wih2,W,Wo2)[0].argmax(1)==pm[:50000]).mean(),i,j))
eff.sort(reverse=True); print("recurrent weights whose individual removal keeps agreement >=99.5%:", sum(e[0]>=0.995 for e in eff), "/256; >=99.9%:", sum(e[0]>=0.999 for e in eff))
