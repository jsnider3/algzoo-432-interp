import numpy as np
from seeds_clocks import load, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
Wih,Whh,Wo=load("weights_seed2.npz"); v=3.24
x=np.random.default_rng(84).standard_normal((100000,T)); Mx=np.maximum(x.max(1),0.1)
def traj(x,t0=None):
    h=np.zeros((len(x),16)); Hs=[]; Ms=[]
    for t in range(T):
        pre=x[:,t:t+1]*Wih+h@Whh.T; h=np.maximum(pre,0)
        if t==t0: h=h.copy(); h[:,10]=np.maximum(h[:,10],v*np.maximum(x[:,:t+1].max(1),0.1))
        Hs.append(h); Ms.append(pre>0)
    return np.stack(Hs,1), np.stack(Ms,1)
H0,M0=traj(x)
cos=lambda a,b: (a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b)
print("Pulse at t0; compare measured mean final change with weight-derived predictions")
for t0 in [1,3,5,7]:
    H1,M1=traj(x,t0); meas=((H1[:,-1]-H0[:,-1])/Mx[:,None]).mean(0)
    # (a) per-sample linearization along the PULSED run's own masks (exact if masks don't change)
    d=np.zeros((len(x),16)); d[:,10]=(H1[:,t0,10]-H0[:,t0,10])/Mx
    for t in range(t0+1,T): d=(d@Whh.T)*M1[:,t]
    lin_own=d.mean(0)
    # (b) a single deterministic product using the AVERAGE mask at each step (pulsed run)
    D=np.zeros(16); D[10]=np.mean((H1[:,t0,10]-H0[:,t0,10])/Mx)
    for t in range(t0+1,T): D=(Whh@D)*M1[:,t].mean(0)
    # (c) fully linear (all neurons on)
    L=np.zeros(16); L[10]=D[10] if False else np.mean((H1[:,t0,10]-H0[:,t0,10])/Mx)
    for t in range(t0+1,T): L=Whh@L
    print(f" t0={t0} (age {9-t0}): cos(meas, per-sample-linearized)={cos(meas,lin_own):.3f}  cos(meas, avg-mask product)={cos(meas,D):.3f}  cos(meas, all-on linear)={cos(meas,L):.3f} | readout argmax: meas {np.argmax(meas@Wo.T)}, avg-mask {np.argmax(D@Wo.T)}, linear {np.argmax(L@Wo.T)}")
    # which neurons are gated (avg mask far from 1) along the way
    if t0==5: print("   avg on-fraction per step after pulse (rows t0+1..9):\n", M1[:,t0+1:].mean(0).round(2))
