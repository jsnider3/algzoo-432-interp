"""Gate-history proliferation (checks ChatGPT round 5's direction (d) diagnostic): for 8 development networks
(0, 16, ..., 112) and the zoo, 2^16 inputs split into a discovery half and a test half; the number of distinct ReLU
on/off histories up to step t and the test mass whose history never appears in the discovery half.  Usage: python scripts/z44.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
nets=[[P[j][i].double().numpy() for j in range(3)] for i in range(0,128,16)]+[[np.asarray(z[j],float) for j in range(3)] for z in Z]
x=th.randn(2**16,10,generator=th.Generator().manual_seed(5),dtype=th.float64).numpy(); half=2**15; T=(3,5,7,9,10); res=[]
for Wih,Whh,Wo in nets:
    h=np.zeros((len(x),16)); hist=[b'']*len(x); row=[]
    for t in range(10):
        h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0); hist=[a+b.tobytes() for a,b in zip(hist,np.packbits(h>0,axis=1))]
        if t+1 in T:
            seen=set(hist[:half]); row.append((len(set(hist)),np.mean([v not in seen for v in hist[half:]])))
    res.append(row)
res=np.array(res)
for k,t in enumerate(T):
    print(f"t={t:2d}: dev median distinct histories {np.median(res[:8,k,0]):.0f}, unseen test mass {100*np.median(res[:8,k,1]):.2f}%; zoo unseen % "+" ".join(f"{100*v:.1f}" for v in res[8:,k,1]))
