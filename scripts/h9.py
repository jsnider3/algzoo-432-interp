import numpy as np
from seeds_clocks import load
Wih,Whh,Wo=load("weights_h32.npz")
chain=[("age0",[7,16,21,23]),("age1",[14]),("age2",[22]),("age3",[19]),("age6",[2]),("age8",[15])]
print("Strongest incoming recurrent weights for each time cell (W_hh[target, source]):")
for nm,cells in chain[1:]:
    for c in cells:
        top=np.argsort(-Whh[c])[:4]
        print(f"  {nm} cell n{c}: " + ", ".join(f"n{j}:{Whh[c,j]:+.2f}" for j in top) + f" | most negative: " + ", ".join(f"n{j}:{Whh[c,j]:+.2f}" for j in np.argsort(Whh[c])[:2]))
