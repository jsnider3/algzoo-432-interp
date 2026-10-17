import numpy as np
from seeds_clocks import load, run, T
rng=np.random.default_rng(21)
def pairs(N):
    x=rng.standard_normal((N,T)); o=np.argsort(-x,1); p2=o[:,1]
    # swap the 2nd max with a random non-top-2 element
    j=o[np.arange(N), rng.integers(2,10,N)]
    x2=x.copy(); x2[np.arange(N),p2]=x[np.arange(N),j]; x2[np.arange(N),j]=x[np.arange(N),p2]
    return x,x2
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); x,x2=pairs(100000)
    h=run(x,Wih,Whh)[:,-1]; h2=run(x2,Wih,Whh)[:,-1]; p=(h@Wo.T).argmax(1); q=(h2@Wo.T).argmax(1)
    d=p!=q
    def transfer(S):
        hp=h.copy(); hp[:,S]=h2[:,S]; r=(hp@Wo.T).argmax(1); return (r==q)[d].mean()
    single=np.array([transfer([n]) for n in range(16)])
    # greedy set
    S=[]; hist=[]
    for _ in range(8):
        best=max([n for n in range(16) if n not in S], key=lambda n: transfer(S+[n])); S.append(best); hist.append(round(transfer(S),2))
    print(f"\n{name}: outputs differ in {d.mean():.2f} of pairs. final-state single-neuron patch transfer:\n  {np.round(single,2)}")
    print(f"  greedy neuron set: {S}  cumulative transfer {hist}")
