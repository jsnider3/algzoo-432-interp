import numpy as np
from seeds_clocks import load, run, T
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(31).standard_normal((300000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    act=np.where((hs[:,5:]>0).mean((0,1))>0.7)[0]; ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Z=h[:,act]@np.linalg.inv(R).T
    o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]
    EM=np.eye(10)[aM]*M[:,None]; ES=np.eye(10)[aS]*S[:,None]
    print(f"\n{name}: additive two-hand fit z ≈ A[ageM]·M + B[ageS]·S")
    for k in np.argsort(-abs(ev)):
        if ev[k].imag<=1e-9: continue
        z=Z[:,k]; V=np.var(z)
        r2=lambda F: 1-np.mean(np.abs(F@np.linalg.lstsq(F,z,rcond=None)[0]-z)**2)/V
        fM=r2(EM); fS=r2(ES); fMS=r2(np.c_[EM,ES])
        # S-hand phase steps
        c=np.linalg.lstsq(np.c_[EM,ES],z,rcond=None)[0]; B=c[10:]; A=c[:10]
        stepB=np.degrees(np.diff(np.unwrap(np.angle(B[2:])))); stepA=np.degrees(np.diff(np.unwrap(np.angle(A[2:]))))
        print(f"  mode |{abs(ev[k]):.2f}| {np.degrees(np.angle(ev[k])):6.1f}°: R2 M-hand {fM:.2f} S-hand {fS:.2f} both {fMS:.2f} | S-hand gain over M-hand {fMS-fM:+.2f} | "
              f"S-hand |B| {np.abs(B[2:]).round(1)} steps {stepB.round(0)} | M-hand steps {stepA.round(0)}")
