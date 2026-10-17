import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path)
    x=np.random.default_rng(4).standard_normal((1500000,T)); o=np.argsort(-x,1); keep=o[:,0]==0
    x=x[keep]; o=o[keep]; h=run(x,Wih,Whh)[:,-1]; S=-np.sort(-x,1)[:,1]; aS=9-o[:,1]
    k=S>0.3; hn=h[k]/S[k,None]; a=aS[k]
    C=np.stack([hn[a==j].mean(0) for j in range(9)]); Cc=C-C.mean(0)
    U,s,Vt=np.linalg.svd(Cc,full_matrices=False); var=s**2/(s**2).sum()
    P=Cc@Vt[:4].T
    ang12=np.degrees(np.unwrap(np.arctan2(P[:,1],P[:,0]))); ang34=np.degrees(np.unwrap(np.arctan2(P[:,3],P[:,2])))
    rad=np.hypot(P[:,0],P[:,1])
    # within-age spread vs between-age spread (how separable)
    within=np.mean([hn[a==j].std(0).mean() for j in range(9)])
    print(f"\n=== {name} (max fixed at position 0; h/S centroids by ageS 0..8)")
    print(" PCA var of centroids:", var[:5].round(2))
    print(" PC1-2 angle by ageS:", ang12.round(0), " steps", np.diff(ang12).round(0))
    print(" PC1-2 radius:", rad.round(2))
    print(" PC3-4 angle steps:", np.diff(ang34).round(0))
    print(" which neurons span PC1-2 (|loading|):", np.abs(Vt[:2]).sum(0).round(2))
    print(" neuron centroid table (rows ageS, cols neurons):"); print(C)
