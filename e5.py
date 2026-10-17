import numpy as np
from seeds_clocks import load
def runL(x,Wih,Whh):
    h=np.zeros((len(x),16))
    for t in range(x.shape[1]): h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0)
    return h
if __name__=="__main__":
    print("Length generalization: run on length-L sequences; readout slot i = age 9-i. Score only sequences whose 2nd max lies in the last 10 positions (and max anywhere).")
    print(f"{'model':8s} " + " ".join(f"L={L:2d}" for L in [10,11,12,14,17,20,30]))
    for name,path in [("mystery","weights.npz")]+[(f"seed {s}",f"weights_seed{s}.npz") for s in range(5)]:
        Wih,Whh,Wo=load(path); row=[]
        for L in [10,11,12,14,17,20,30]:
            x=np.random.default_rng(L).standard_normal((300000,L)); p=(runL(x,Wih,Whh)@Wo.T).argmax(1)
            p2=np.argsort(x,1)[:,-2]; s=p2>=L-10
            row.append(((p+L-10)==p2)[s].mean())
        print(f"{name:8s} "+" ".join(f"{r:.3f}" for r in row))
