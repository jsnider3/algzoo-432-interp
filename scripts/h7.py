import numpy as np
from seeds_clocks import load, run, analyze, T
for hsz in [4,6,8,32]:
    path=f"weights_h{hsz}.npz"; Wih,Whh,Wo=load(path)
    x=np.random.default_rng(3).standard_normal((400000,T)); pm=(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S,R=srt[:,0],srt[:,1],srt[:,2]; err=pm!=o[:,1]
    clean=(S>0)&((M-S)>.1*M)&((S-R)>.1*M)
    # readout diagonality given max position
    L=(run(x,Wih,Whh)[:,-1]@Wo.T); L=L-L.mean(1,keepdims=True); aM=9-o[:,0]; dg=[]
    for a in range(10):
        s=aM==a; c,*_=np.linalg.lstsq(x[s],L[s],rcond=None); W=c.T; p=9-a
        for i in range(10):
            if i!=p: dg.append(abs(W[i,i])/(np.abs(np.delete(W[i],[i,p])).sum()+1e-9))
    print(f"\n##### hidden {hsz}: acc {1-err.mean():.4f}, clean-case error {err[clean].mean():.4f}, error when S<0 {err[S<0].mean():.2f}, readout diag ratio {np.median(dg):.2f}")
    analyze(f"hidden {hsz}", path)
