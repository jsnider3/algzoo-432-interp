import sys, numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H,T=16,10
import numpy as _np
def load(path):
    w=np.load(path); return w['rnn.weight_ih_l0'][:,0].astype(float), w['rnn.weight_hh_l0'].astype(float), w['linear.weight'].astype(float)
def run(x,Wih,Whh):
    h=np.zeros((len(x),len(Wih))); hs=[]
    for t in range(T): h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0); hs.append(h)
    return np.stack(hs,1)
def circ(z):  # circular mean direction and concentration
    u=np.mean(z/np.abs(z)); return np.degrees(np.angle(u)), np.abs(u)
def analyze(name, path):
    Wih,Whh,Wo=load(path)
    x=np.random.default_rng(0).standard_normal((300000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
    tgt=np.argsort(x,1)[:,-2]; acc=((h@Wo.T).argmax(1)==tgt).mean()
    # age-code test: shifted readout at t=8
    l8=hs[:,8]@Wo.T; p2_8=np.argsort(x[:,:9],1)[:,-2]; shift=(l8.argmax(1)==p2_8+1).mean()
    # active subsystem: neurons on >70% of the time over t>=5
    frac=(hs[:,5:]>0).mean((0,1)); act=np.where(frac>0.7)[0]
    ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Z=h[:,act]@np.linalg.inv(R).T
    o=np.argsort(-x,1); aM=9-o[:,0]; aS=9-o[:,1]; p1,p2=o[:,0],o[:,1]
    rows=[]
    for k in range(len(ev)):
        if ev[k].imag<=1e-6: continue
        z=Z[:,k]
        res={}
        for nm,ag in [('M',aM),('S',aS)]:
            ph=[];cc=[]
            for a in range(2,10):   # ignore ages 0-1 (handled by fast/sparse neurons)
                p,c=circ(z[ag==a]); ph.append(p); cc.append(c)
            inc=np.diff(np.unwrap(np.radians(ph))); res[nm]=(np.mean(cc), np.degrees(np.median(inc)), np.degrees(np.std(inc)))
        rows.append((k,ev[k],res))
    print(f"\n=== {name}: acc {acc:.4f}; shifted-readout(t=8) acc {shift:.3f}; active neurons {len(act)}: {act.tolist()}")
    for k,e,res in sorted(rows,key=lambda r:-abs(r[1])):
        flag=""
        for nm in 'MS':
            if res[nm][0]>0.6 and res[nm][2]<15: flag+=f" <-- {nm}-CLOCK"
        print(f"  mode {e.real:+.2f}{e.imag:+.2f}i |{abs(e):.3f}| angle {np.degrees(np.angle(e)):6.1f}° | "
              f"vs ageM: conc {res['M'][0]:.2f} step {res['M'][1]:6.1f}°±{res['M'][2]:4.1f} | vs ageS: conc {res['S'][0]:.2f} step {res['S'][1]:6.1f}°±{res['S'][2]:4.1f}{flag}")
    # phase inheritance on best S-clock
    best=max(rows,key=lambda r:r[2]['S'][0]-r[2]['S'][2]/100); k=best[0]; z=Z[:,k]
    d=[]
    for a in range(3,9):
        pa,_=circ(z[(aS==a)&(p2<p1)]); pb,_=circ(z[(aS==a)&(p2>p1)]); d.append(((pa-pb+180)%360)-180)
    print(f"  S-clock {best[1]:.2f}: phase(inherited from old max) - phase(set directly), ageS 3..8: {np.round(d,0)}")
if __name__=="__main__":
    analyze("mystery (example)", "weights.npz")
    for s in range(5): analyze(f"seed {s}", f"weights_seed{s}.npz")
