import numpy as np
from seeds_clocks import load
def best_pointer(path):
    Wih,Whh,Wo=load(path); best=(0,None,None)
    for n in range(16):
        for sgn in [1,-1]:
            for delay in [0,1]:
                v=np.zeros(16); v[n]=sgn; ok=0; marg=[]
                for a in range(10-delay):
                    L=Wo@v; L=L-L.mean(); s=9-a-delay; ok+=int(np.argmax(L)==s)
                    marg.append(L[s]-np.max(np.delete(L,s))); v=Whh@v
                sc=ok/(10-delay)
                if sc>best[0] or (sc==best[0] and best[1] is not None and np.median(marg)>best[2][1]):
                    best=(sc,(n,sgn,delay),(ok,np.median(marg)))
    return best
for s in [2,3,0]:
    row=[]
    for e in [22,23,24,25,26,27,28,30]:
        sc,(n,sg,dl),(ok,mg)=best_pointer(f"ckpt/s{s}_e{e}.npz"); row.append(f"2^{e}: {sc:.1f} ({'+' if sg>0 else '-'}n{n}{',d1' if dl else ''})")
    print(f"seed {s}: best weights-only pointer score (fraction of ages decoded) -> " + " | ".join(row))
