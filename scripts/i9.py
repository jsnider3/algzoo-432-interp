from common import *
np.set_printoptions(precision=1, suppress=True, linewidth=220)
x=sample(400000,seed=131); lg,hs=run(x)
t=4  # event at step t; look at state at t+1 (after n5 fires)
P=x[:,:t]; srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; xt=x[:,t]
newM=xt>M; newS=(xt>S)&(xt<M)
Mt=np.maximum(x[:,:t+2].max(1),.1)
hM=(hs[newM,t+1]/Mt[newM,None]).mean(0); hS=(hs[newS,t+1]/Mt[newS,None]).mean(0)
d=hM-hS
print("mean state at t+1 (per unit M): new-max event vs new-2nd event; difference:")
print(" new-max :",hM); print(" new-2nd :",hS); print(" diff    :",d)
print("weights-only readout of the propagated pieces, slot index of the event = 9-a  (a = steps after t+1, +1)")
for a in range(0,8):
    A=np.linalg.matrix_power(Whh,a)
    LS=Wo@A@hS; LD=Wo@A@d; ev=9-(a+1)
    LS=LS-LS.mean(); LD=LD-LD.mean()
    print(f"  {a+1} steps after event: new-2nd state -> argmax {np.argmax(LS)} (event slot {ev}, logit {LS[ev]:+.1f}) | new-max minus new-2nd -> logit at event slot {LD[ev]:+.1f}, argmin slot {np.argmin(LD)}")
