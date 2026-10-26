import numpy as np
from seeds_clocks import load, run, T
from f7 import jac
x=np.random.default_rng(71).standard_normal((150000,T))
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]
EM=np.eye(10)[aM]*M[:,None]; ES=np.eye(10)[aS]*S[:,None]
for s in [2,3,0]:
    print(f"\nseed {s}: per checkpoint — best 2nd-max-hand gain on any rotating mode (and that mode's angle); max-slot own gradient; # usually-active neurons")
    for e in [23,24,25,26,27,30]:
        Wih,Whh,Wo=load(f"ckpt/s{s}_e{e}.npz"); hs=run(x,Wih,Whh); h=hs[:,-1]
        act=np.where((hs[:,5:]>0).mean((0,1))>0.7)[0]; ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Z=h[:,act]@np.linalg.inv(R).T
        best=(0,0)
        for k in range(len(ev)):
            if ev[k].imag<=1e-9: continue
            z=Z[:,k]; V=np.var(z); r2=lambda F: 1-np.mean(np.abs(F@np.linalg.lstsq(F,z,rcond=None)[0]-z)**2)/V
            gain=r2(np.c_[EM,ES])-r2(EM)
            # weight gain by mode's share of readout? keep simple
            if gain>best[0]: best=(gain,np.degrees(np.angle(ev[k])))
        J,_=jac(x[:20000],Wih,Whh,Wo); Jc=J-J.mean(1,keepdims=True); n=np.arange(20000); p1=o[:20000,0]
        print(f"  2^{e}: S-hand gain {best[0]:.2f} @ {best[1]:5.1f}° | d logit_max/d M {Jc[n,p1,p1].mean():7.1f} | active {len(act)}")
