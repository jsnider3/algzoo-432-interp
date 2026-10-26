import numpy as np
from seeds_clocks import load, T
Wih,Whh,Wo=load("weights_seed2.npz")
def run_hook(x,hook):
    h=np.zeros((len(x),16))
    for t in range(T):
        h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0); h=hook(t,h)
    return (h@Wo.T).argmax(1)
x=np.random.default_rng(83).standard_normal((300000,T)); base=run_hook(x,lambda t,h:h)
# typical n10 firing value when a genuine new 2nd max arrives
h=np.zeros((len(x),16)); vals=[]
for t in range(T):
    h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0)
    if t>=2:
        P=x[:,:t]; srt=-np.sort(-P,1); s=(x[:,t]>srt[:,1])&(x[:,t]<srt[:,0])&(h[:,10]>0); vals.append(h[s,10]/srt[s,0])
v=np.median(np.concatenate(vals)); print(f"median n10 activation / M on genuine new-2nd events: {v:.2f}")
M=x.max(1)
for t in [3,5,7,8]:
    P=x[:,:t+1]; srt=-np.sort(-P,1); sel=(x[:,t]<srt[:,1])  # x_t was NOT a new 2nd max
    def hook(tt,h,t=t):
        if tt==t: h=h.copy(); h[:,10]=np.maximum(h[:,10], v*np.maximum(x[:,:t+1].max(1),0.1))
        return h
    p=run_hook(x,hook)
    print(f"force n10 pulse at step {t}: P(answer = position {t}) {np.mean(p[sel]==t):.3f} (baseline {np.mean(base[sel]==t):.3f}); answer changed {np.mean(p[sel]!=base[sel]):.3f}")
print("--- restricted to runs where no input after step t beats the true 2nd max of x_0..x_t (no genuine later updates)")
for t in [2,3,5,7,8]:
    P=x[:,:t+1]; srt=-np.sort(-P,1); later=x[:,t+1:].max(1) if t<9 else -np.inf
    sel=(x[:,t]<srt[:,1])&(later<srt[:,1])
    def hook(tt,h,t=t):
        if tt==t: h=h.copy(); h[:,10]=np.maximum(h[:,10], v*np.maximum(x[:,:t+1].max(1),0.1))
        return h
    p=run_hook(x,hook)
    print(f"  pulse at {t}: n={sel.sum()}, P(answer = {t}) {np.mean(p[sel]==t):.3f}")
