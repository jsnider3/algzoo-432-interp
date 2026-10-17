import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz")]:
    Wih,Whh,Wo=load(path)
    x=np.random.default_rng(5).standard_normal((400000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    frac=(hs[:,5:]>0).mean((0,1)); act=np.where(frac>0.7)[0]
    ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Z=h[:,act]@np.linalg.inv(R).T
    o=np.argsort(-x,1); srt=-np.sort(-x,1); S=srt[:,1]; aS=9-o[:,1]; aM=9-o[:,0]
    k=S>0.3
    print(f"\n=== {name}: per-mode amplitude code for ageS (|z|/S, ages 2..9; and conditional on S after M vs before)")
    for j in np.argsort(-abs(ev)):
        if ev[j].imag<-1e-9: continue
        amp=np.abs(Z[:,j])/S
        m=np.array([np.median(amp[k&(aS==a)]) for a in range(2,10)])
        sl=np.polyfit(np.arange(2,10),np.log(m+1e-9),1)[0]
        # how much does amp separate ages: ratio of between-age variance of log amp to total
        la=np.log(amp[k]+1e-9); aa=aS[k]
        between=np.var([la[aa==a].mean() for a in range(2,10)]); tot=la[aa>=2].var()
        print(f"  mode {ev[j]:.2f} |{abs(ev[j]):.2f}|: median |z|/S by ageS {m.round(2)}  log-slope/step {sl:+.2f} (log|λ|={np.log(abs(ev[j])):+.2f}) frac var by ageS {between/tot:.2f}")
