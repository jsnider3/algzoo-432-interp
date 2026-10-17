import numpy as np
from seeds_clocks import load
np.set_printoptions(precision=1, suppress=True, linewidth=220)
for name,path,src in [("seed 2 (pulse in n10)","weights_seed2.npz",10),("seed 2 (pulse in n15, new-max detector)","weights_seed2.npz",15)]:
    Wih,Whh,Wo=load(path)
    print(f"\n{name}: weights-only impulse response, logits = Wo · Whh^a · e_src (all neurons linear)")
    v=np.zeros(16); v[src]=1.0; ok=0
    for a in range(10):
        L=Wo@v; L=L-L.mean(); ok+=int(np.argmax(L)==9-a)
        print(f"  age {a}: argmax slot {np.argmax(L)} (want {9-a}); margin over runner-up {np.sort(L)[-1]-np.sort(L)[-2]:.2f}; logits {L}")
        v=Whh@v
    print(f"  correct slot for {ok}/10 ages")
