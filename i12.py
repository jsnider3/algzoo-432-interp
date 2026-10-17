import numpy as np
from seeds_clocks import load, run, T
for name,path,src,delay in [("mystery (n5)","weights.npz",5,1),("seed 2 (n10)","weights_seed2.npz",10,0)]:
    Wih,Whh,Wo=load(path)
    R=np.array([np.linalg.matrix_power(Whh,a)@np.eye(16)[src] for a in range(10-delay)]).T
    Q,_=np.linalg.qr(R)                                   # span of impulse responses
    x=np.random.default_rng(142).standard_normal((300000,T)); h=run(x,Wih,Whh)[:,-1]; tgt=np.argsort(x,1)[:,-2]
    C=np.cov(h.T); ev,U=np.linalg.eigh(C); top=U[:,::-1][:,:3]   # 3 largest-variance state directions
    Wc=Wo-Wo.mean(0)                                       # common offset irrelevant to argmax
    inspan=np.linalg.norm(Wc@Q)**2/np.linalg.norm(Wc)**2
    # how much the trained readout cancels the top-variance directions vs a random readout of same norm
    gain_top=np.linalg.norm(Wc@top)/np.linalg.norm(Wc)
    rnd=np.random.default_rng(0).standard_normal(Wc.shape); rnd-=rnd.mean(0)
    print(f"{name}: share of (centered) readout in impulse-response span {inspan:.2f} (span dim {Q.shape[1]}/16) | readout gain on top-3 variance directions {gain_top:.2f} vs random {np.linalg.norm(rnd@top)/np.linalg.norm(rnd):.2f}")
    # project readout onto impulse span only: accuracy
    Wp=(Wc@Q)@Q.T; print(f"   accuracy with readout projected onto impulse span: {((h@Wp.T).argmax(1)==tgt).mean():.3f}; onto complement: {((h@(Wc-Wp).T).argmax(1)==tgt).mean():.3f}")
