import numpy as np
from seeds_clocks import load, T
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def final(x,Wih,Whh,hook):
    h=np.zeros((len(x),16))
    for t in range(T): h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0); h=hook(t,h)
    return h
for name,path,n,v in [("seed 2","weights_seed2.npz",10,3.24)]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(84).standard_normal((100000,T)); Mx=np.maximum(x.max(1),0.1)
    h0=final(x,Wih,Whh,lambda t,h:h); D=[]
    for t in range(1,T):
        def hook(tt,h,t=t):
            if tt==t: h=h.copy(); h[:,n]=np.maximum(h[:,n],v*np.maximum(x[:,:t+1].max(1),0.1))
            return h
        D.append(((final(x,Wih,Whh,hook)-h0)/Mx[:,None]).mean(0))
    D=np.array(D)[::-1]   # rows: pulse age 0..8
    print(f"{name}: mean final-state change per unit M from an n{n} pulse, by pulse age (rows 0..8):"); print(D)
    U,s,Vt=np.linalg.svd(D-D.mean(0)); print(" PCA var:", (s**2/(s**2).sum()).round(2)[:5])
    P=(D-D.mean(0))@Vt[:2].T; print(" PC1-2 angle steps:", np.diff(np.degrees(np.unwrap(np.arctan2(P[:,1],P[:,0])))).round(0), " radius", np.hypot(P[:,0],P[:,1]).round(1))
    print(" readout of the change (logits), argmax slot by pulse age:", (D@Wo.T).argmax(1), "(expected 9..1)")
