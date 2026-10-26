from common import *
x=sample(300000,seed=93); lg,hs=run(x); base=lg.argmax(1)
# firing of n5 at step t by what happened at step t-1
for t in [4,6,8]:
    P=x[:,:t-1]; srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; xp=x[:,t-1]; f=hs[:,t,5]>0
    print(f"n5 fires at step {t} when x_(t-1) was: new max {f[xp>M].mean():.2f}, new 2nd {f[(xp>S)&(xp<M)].mean():.2f}, neither {f[xp<S].mean():.2f}")
v=np.median(np.concatenate([hs[:,t,5][(hs[:,t,5]>0)]/np.maximum(x[:,:t].max(1),.1)[(hs[:,t,5]>0)] for t in range(3,10)]))
print("median n5 activation / M when firing:", round(v,2))
def run_hook(hook):
    h=np.zeros((len(x),H))
    for t in range(T): h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0); h=hook(t,h)
    return (h@Wo.T).argmax(1)
print("force n5 pulse at step t (marks an event at t-1); runs with no later genuine update:")
for t in [3,5,7,9]:
    e=t-1; P=x[:,:e+1]; srt=-np.sort(-P,1); later=x[:,e+1:].max(1)
    sel=(x[:,e]<srt[:,1])&(later<srt[:,1])
    def hook(tt,h,t=t):
        if tt==t: h=h.copy(); h[:,5]=np.maximum(h[:,5],v*np.maximum(x[:,:t].max(1),.1))
        return h
    p=run_hook(hook)
    print(f"  pulse at {t}: n={sel.sum()}, P(answer = {e}) {np.mean(p[sel]==e):.3f} (baseline {np.mean(base[sel]==e):.3f})")
