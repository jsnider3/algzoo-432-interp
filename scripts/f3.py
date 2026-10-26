import numpy as np
from seeds_clocks import load, run, T
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(31).standard_normal((300000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    act=np.where((hs[:,5:]>0).mean((0,1))>0.7)[0]; ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Z=h[:,act]@np.linalg.inv(R).T
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]
    print(f"\n{name}: complex fit of each rotating mode z ≈ A·M·λ^ageM + B·S·λ^ageS (+ per-age-cell free fit as ceiling)")
    for k in np.argsort(-abs(ev)):
        if ev[k].imag<=1e-9: continue
        lam=ev[k]; z=Z[:,k]
        def r2(F):
            c,*_=np.linalg.lstsq(F,z,rcond=None); return 1-np.mean(np.abs(F@c-z)**2)/np.var(z)
        fM=r2(np.c_[M*lam**aM]); fS=r2(np.c_[S*lam**aS]); fMS=r2(np.c_[M*lam**aM,S*lam**aS])
        # ceiling: free complex coefficient per (aM,aS) cell times M and S
        P=np.zeros_like(z)
        for c_ in np.unique(aM*10+aS):
            s=(aM*10+aS)==c_; F=np.c_[M[s],S[s]]; cc,*_=np.linalg.lstsq(F,z[s],rcond=None); P[s]=F@cc
        ceil=1-np.mean(np.abs(P-z)**2)/np.var(z)
        print(f"  mode |{abs(lam):.2f}| {np.degrees(np.angle(lam)):6.1f}°: R2 M-hand {fM:.2f}  S-hand {fS:.2f}  both hands {fMS:.2f}   ceiling {ceil:.2f}")
