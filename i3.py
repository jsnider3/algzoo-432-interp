import numpy as np
from seeds_clocks import load
Wih,Whh,Wo=load("weights.npz")
print("mystery: weights-only impulse readout Wo·Whh^a·(±e_n): slot picked by age a (want 9-a)")
for n,sign in [(4,+1),(5,+1),(9,+1),(6,-1),(8,-1),(1,-1),(7,-1),(14,-1)]:
    v=np.zeros(16); v[n]=sign; picks=[]; ok=0
    for a in range(10):
        L=Wo@v; L=L-L.mean(); picks.append(int(np.argmax(L))); ok+=picks[-1]==9-a; v=Whh@v
    print(f"  {'+' if sign>0 else '-'}n{n:<2d}: {picks}  correct {ok}/10")
# best linear combination of event neurons: find direction u (16-d) maximizing correct decoding? use lsq: Wo Whh^a u ≈ onehot(9-a)
A=np.concatenate([Wo@np.linalg.matrix_power(Whh,a) for a in range(10)])   # (100,16)
Y=np.concatenate([np.eye(10)[9-a] for a in range(10)])
u,*_=np.linalg.lstsq(A,Y,rcond=None)
picks=[int(np.argmax(Wo@np.linalg.matrix_power(Whh,a)@u)) for a in range(10)]
print("  best single write direction u (lsq): picks", picks, " u≈", np.round(u/np.abs(u).max(),2))
