import numpy as np
from seeds_clocks import load, run, T
rng=np.random.default_rng(51)
for name,path in [("seed 0","weights_seed0.npz"),("seed 4","weights_seed4.npz"),("mystery","weights.npz")]:
    Wih,Whh,Wo=load(path); N=150000; x=rng.standard_normal((N,T)); o=np.argsort(-x,1); p1=o[:,0]
    j=o[np.arange(N),rng.integers(2,10,N)]; x2=x.copy(); x2[np.arange(N),p1]=x[np.arange(N),j]; x2[np.arange(N),j]=x[np.arange(N),p1]
    hs=run(x,Wih,Whh); h=hs[:,-1]; h2=run(x2,Wih,Whh)[:,-1]
    act=np.where((hs[:,5:]>0).mean((0,1))>0.7)[0]; ina=[n for n in range(16) if n not in act]
    ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Li=np.linalg.inv(R); Z=h[:,act]@Li.T; Z2=h2[:,act]@Li.T
    base=((h@Wo.T).argmax(1)==p1).mean()
    print(f"\n{name}: picks old max position p1 — baseline {base:.3f}; after patching from max-moved run:")
    full=h2.copy(); print(f"  whole state (= run on x2): {((full@Wo.T).argmax(1)==p1).mean():.3f}")
    res=[]
    done=set()
    for k in np.argsort(-abs(ev)):
        if k in done: continue
        g=[k]+[q for q in range(len(ev)) if q!=k and abs(ev[q]-np.conj(ev[k]))<1e-8 and abs(ev[k].imag)>1e-9]; done.update(g)
        Zp=Z.copy(); Zp[:,g]=Z2[:,g]; hp=h.copy(); hp[:,act]=np.real(Zp@R.T)
        res.append((((hp@Wo.T).argmax(1)==p1).mean(), ev[k]))
    for v,e in sorted(res,key=lambda r:-r[0])[:5]: print(f"  mode |{abs(e):.2f}| {np.degrees(abs(np.angle(e))):5.1f}°: {v:.3f}")
    hp=h.copy(); hp[:,ina]=h2[:,ina]; print(f"  sparse neurons {ina}: {((hp@Wo.T).argmax(1)==p1).mean():.3f}")
