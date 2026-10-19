import numpy as np
from seeds_clocks import load, run, T
# REPORT 7.14 caveat: the "switch" criterion of k1.py (some neuron fires on >=80% of new-2nd events and on <0.4 of
# "neither" inputs) scored on untrained weights and on a bare sign detector. Same function and eval set as k1.py.
xe=np.random.default_rng(201).standard_normal((100000,T))
def best_pointer_sel(Wih,Whh):
    hs=run(xe,Wih,Whh); best=(9,None)
    for dl in [0,1]:
        rates=np.zeros((2,len(Wih)))
        for t in range(3,T):
            e=t-dl; P=xe[:,:e]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]; x_=xe[:,e]; f=hs[:,t]>0
            rates[0]+=f[(x_>S)&(x_<M)].mean(0)/(T-3); rates[1]+=f[x_<S].mean(0)/(T-3)
        for n in range(len(Wih)):
            if rates[0,n]>=0.8 and rates[1,n]<best[0]: best=(rates[1,n],(n,dl))
    return best
for s in range(5):
    print(f"seed {s}: " + " | ".join(f"2^{e}: {best_pointer_sel(*load(f'ckpt/s{s}_e{e}.npz')[:2])[0]:.2f}" for e in [0,24,30]))
for c in [-0.5,0,0.5]:
    a=b=0
    for t in range(3,T):
        P=xe[:,:t]; s_=-np.sort(-P,1); M,S=s_[:,0],s_[:,1]; x_=xe[:,t]; f=x_>c
        a+=f[(x_>S)&(x_<M)].mean()/(T-3); b+=f[x_<S].mean()/(T-3)
    print(f"bare detector x_t>{c:+.1f}: new-2nd rate {a:.2f}, neither rate {b:.2f} -> counts as a switch: {a>=0.8 and b<0.4}")
