import numpy as np
from seeds_clocks import load
# weights-only pulse test over all 64 candidates (16 neurons x sign x 0/1-step delay): fraction of ages a for which
# argmax(Wo·Whh^a·(±e_n)) is the slot 9-a-delay. Which final models have a candidate that decodes every age?
def scores(Wo,Whh):
    out=[]
    for n in range(16):
        for sgn in [1,-1]:
            for delay in [0,1]:
                v=np.zeros(16); v[n]=sgn; ok=0
                for a in range(10-delay):
                    ok+=int(np.argmax(Wo@v)==9-a-delay); v=Whh@v
                out.append((ok/(10-delay),n,sgn,delay))
    return out
for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
    Wih,Whh,Wo=load(path); sc=scores(Wo,Whh)
    full=[(("+" if sg>0 else "-")+f"n{n}"+(",d1" if d else "")) for v,n,sg,d in sc if v==1.0]
    print(f"{name:8s}: candidates decoding every age {full}  best score {max(s[0] for s in sc):.2f}")
