import numpy as np
from seeds_clocks import load, run, T
for name,path,target in [("seed 0","weights_seed0.npz",-0.88+0.19j),("seed 4","weights_seed4.npz",-0.79+0.20j)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(41).standard_normal((200000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    act=np.where((hs[:,5:]>0).mean((0,1))>0.7)[0]; ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Li=np.linalg.inv(R); Z=h[:,act]@Li.T
    ks=[k for k in range(len(ev)) if abs(ev[k]-target)<0.02 or abs(ev[k]-np.conj(target))<0.02]
    o=np.argsort(-x,1); p1=o[:,0]; N=len(x)
    L0=h@Wo.T; Lc=L0-L0.mean(1,keepdims=True)
    # per-slot gain c: logit_i ≈ c*x_i for non-max slots
    m=np.ones_like(x,bool); m[np.arange(N),p1]=False; c=np.polyfit(x[m],Lc[m],1)[0]
    def hole(L):  # slot with most-negative residual = suppressed slot
        Lc=L-L.mean(1,keepdims=True); return np.argmin(Lc-c*x,1)
    print(f"\n{name}: max clock modes {[np.round(ev[k],2) for k in ks]}; gain c={c:.1f}")
    print(f"  baseline: hole at max position {(hole(L0)==p1).mean():.3f}; model picks the max {(L0.argmax(1)==p1).mean():.3f}")
    for st in [1,2,3]:
        Z2=Z.copy()
        for k in ks: Z2[:,k]=Z[:,k]*np.exp(1j*np.angle(ev[k])*st)
        h2=h.copy(); h2[:,act]=np.real(Z2@R.T); L=h2@Wo.T; hl=hole(L); sel=p1>=st
        print(f"  rotate max clock {st} step(s): hole at p1 {(hl==p1)[sel].mean():.3f}, at p1-{st} {(hl==p1-st)[sel].mean():.3f}; model now picks the true max {(L.argmax(1)==p1)[sel].mean():.3f}")
