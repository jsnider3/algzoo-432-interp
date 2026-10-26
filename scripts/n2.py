from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x=sample(300000,seed=901); lg,hs=run(x)
n=5
print("n5 incoming weights (Whh[5,:]):", Whh[n]); print("Wih[5]:", round(Wih[n],2))
# contribution of each source neuron to n5's preactivation at step t, grouped by what happened at step t-1
t=6; P=x[:,:t-1]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]; xp=x[:,t-1]
cases={'prev new max':xp>M,'prev new 2nd':(xp>S)&(xp<M),'prev neither':xp<S}
contrib=hs[:,t-1]*Whh[n][None,:]          # (N,16)
Mt=np.maximum(x[:,:t].max(1),.1)[:,None]
for k,sel in cases.items():
    c=(contrib[sel]/Mt[sel]).mean(0); print(f"  {k:13s}: preact/M {c.sum()+ (Wih[n]*x[sel,t]/Mt[sel,0]).mean():+.2f} | top contributions:", ", ".join(f"n{j}:{c[j]:+.2f}" for j in np.argsort(-np.abs(c))[:6]))
