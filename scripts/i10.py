from common import *
x=sample(400000,seed=132); lg,hs,ms,_=run_jac(x[:1]) if False else (None,None,None,None)
lg,hs=run(x)
t=4
P=x[:,:t]; o=np.argsort(-P,1); srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; xt=x[:,t]
later_ok=x[:,t+1:].max(1)<S   # no later updates: final answer should be old max position
sel=(xt>M)&later_ok; xs=x[sel]; oldmax=o[sel,0]; ev=t
# matched counterfactual: same sequence but x_t lowered to sit between S and M (a new-2nd event)
xc=xs.copy(); xc[:,t]=(S[sel]+M[sel])/2
def states(z):
    h=np.zeros((len(z),H)); out=[]; masks=[]
    for u in range(T):
        pre=z[:,u:u+1]*Wih+h@Whh.T; h=np.maximum(pre,0); out.append(h); masks.append(pre>0)
    return np.stack(out,1), np.stack(masks,1)
HA,MA=states(xs); HC,MC=states(xc)
d=HA[:,t+1]-HC[:,t+1]     # difference right after the pointer fires
groups={'n5 pointer':[5],'n4,n9 max markers':[4,9],'clock nbrs 0,10-15':[0,10,11,12,13,14,15],'comparators 1,6,7,8':[1,6,7,8],'n2,n3':[2,3]}
n=np.arange(len(xs))
def propagate(dd):
    for u in range(t+2,T): dd=(dd@Whh.T)*MA[:,u]
    return dd@Wo.T
Lc=HC[:,-1]@Wo.T; La=HA[:,-1]@Wo.T
print(f"new-max events at step {t}, no later updates (n={len(xs)}). Actual run answers old max: {(La.argmax(1)==oldmax).mean():.3f}; counterfactual (new-2nd) answers event slot: {(Lc.argmax(1)==ev).mean():.3f}")
tot=propagate(d)
print(f"gated propagation of full difference: mean Δlogit at event slot {tot[:,ev].mean():+.1f}, at old-max slot {tot[n,oldmax].mean():+.1f}; reconstruction check counterfactual+Δ answers old max {((Lc+tot).argmax(1)==oldmax).mean():.3f}")
for nm,g in groups.items():
    dd=np.zeros_like(d); dd[:,g]=d[:,g]; L=propagate(dd)
    print(f"  {nm:22s}: Δlogit at event slot {L[:,ev].mean():+7.1f} | at old-max slot {L[n,oldmax].mean():+7.1f}")
