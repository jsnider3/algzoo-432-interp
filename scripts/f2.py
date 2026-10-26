import numpy as np
from seeds_clocks import load, run, T
from f1 import pairs
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); x,x2=pairs(100000)
    hs=run(x,Wih,Whh); h=hs[:,-1]; h2=run(x2,Wih,Whh)[:,-1]; p=(h@Wo.T).argmax(1); q=(h2@Wo.T).argmax(1); d=p!=q
    act=np.where((hs[:,5:]>0).mean((0,1))>0.7)[0]; ina=[n for n in range(16) if n not in act]
    ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Li=np.linalg.inv(R)
    Z=h[:,act]@Li.T; Z2=h2[:,act]@Li.T
    # group conjugate pairs
    groups=[]; used=set()
    for k in range(len(ev)):
        if k in used: continue
        g=[k]+[j for j in range(len(ev)) if j!=k and j not in used and abs(ev[j]-np.conj(ev[k]))<1e-8 and abs(ev[k].imag)>1e-9]
        used.update(g); groups.append(g)
    def transfer(gs, inact=False):
        Zp=Z.copy()
        for g in gs:
            for k in g: Zp[:,k]=Z2[:,k]
        hp=h.copy(); hp[:,act]=np.real(Zp@R.T)
        if inact: hp[:,ina]=h2[:,ina]
        return ((hp@Wo.T).argmax(1)==q)[d].mean()
    print(f"\n{name}: transfer when patching one mode (pair) from the swapped run")
    for g in sorted(groups,key=lambda g:-abs(ev[g[0]])):
        e=ev[g[0]]; print(f"  mode {e.real:+.2f}{abs(e.imag):+.2f}i |{abs(e):.2f}| {np.degrees(abs(np.angle(e))):5.1f}°: {transfer([g]):.2f}")
    print(f"  sparse/inactive neurons {ina} alone: {transfer([],True):.2f}")
    S=[]; hist=[]
    cand=list(range(len(groups)))
    for _ in range(4):
        b=max([c for c in cand if c not in S], key=lambda c: transfer([groups[i] for i in S+[c]])); S.append(b); hist.append(round(transfer([groups[i] for i in S]),2))
    print("  greedy modes:", [f"{np.degrees(abs(np.angle(ev[groups[i][0]]))):.0f}°|{abs(ev[groups[i][0]]):.2f}|" for i in S], hist)
