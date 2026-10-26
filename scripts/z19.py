"""Which neurons carry the double-counted gate flips, and when?  (z16: gates flipped by two or more one-input runs carry
79% of the first-order margin model's bias.)  On z15's saved draws, for every gate (time step x neuron) flipped by >= 2
one-input runs, record the neuron's role (z3's rule-based buckets) and the phase of the sequence at that step:
  prefix   before the first of the top-4 values has arrived (the running max and 2nd max are lower inputs)
  middle   after the first top-4 value, before the 2nd max's position
  late     at or after the 2nd max's position
Shares among the first-order sign errors are compared with ordinary draws and with all single-input gate flips.
Population networks only (every 16th).  Usage: python scripts/z19.py [anchor, default mean]
"""
import sys, numpy as np, torch as th
from z2 import load
from z3 import ROLES
dev='cuda'; anchor=sys.argv[1] if len(sys.argv)>1 else 'mean'
z=np.load(f'pop1k/seed0_z15_{anchor}.npz')
P,_=load('pop1k/seed0.npz'); W=[w[::16].to(dev) for w in P]; N=W[0].shape[0]
roles=np.load('pop1k/seed0.buckets.npz')['roles'][::16]                               # (64, 16)
def run(w, X):
    h=th.zeros(X.shape[0],16,device=dev); G=[]
    for t in range(10):
        pre=X[:,t:t+1]*w[0]+h@w[1].T; G.append(pre>0); h=th.relu(pre)
    return th.stack(G,1)
m1=z['m0']+z['d'].sum(1); err=(m1>0)!=z['It']; sub=z['sub']
R=len(ROLES); cnt={k:np.zeros((R,3)) for k in ('shared_err','shared_ord','single_ord')}
for n in range(N):
    k=np.nonzero(z['net']==n)[0]
    if len(k)==0: continue
    lp=th.tensor(z['lp'][k],device=dev); top=th.tensor(z['top'][k],device=dev); v=th.tensor(z['v'][k],device=dev,dtype=th.float32)
    a=th.tensor(z['a'][k],device=dev,dtype=th.float32); zz=th.tensor(z['z'][k],device=dev,dtype=th.float32); B=len(k)
    base=a[:,None].expand(B,10).clone(); base.scatter_(-1,top,v)
    singles=base[:,None].repeat(1,6,1); singles.scatter_(-1,lp[:,:,None],zz[:,:,None])
    w=[q[n] for q in W]; G0=run(w,base); GS=run(w,singles.reshape(-1,10)).view(B,6,10,16)
    c=(GS!=G0[:,None]).sum(1)                                                          # (B, 10, 16) one-input runs flipping each gate
    t=th.arange(10,device=dev)[None]; first=top.min(1).values[:,None]; Sp=top[:,1,None]
    phase=th.where(t<first,0,th.where(t<Sp,1,2))                                       # (B, 10)
    role=th.tensor(roles[n],device=dev)
    for key,mask,sel in (('shared_err',c>=2,err[k]),('shared_ord',c>=2,sub[k]),('single_ord',c>=1,sub[k])):
        s=th.tensor(sel,device=dev); mm=mask&s[:,None,None]
        idx=(role[None,None,:]*3+phase[:,:,None]).expand(B,10,16)[mm]
        cnt[key]+=th.bincount(idx,minlength=R*3).view(R,3).double().cpu().numpy()
share={k:v/v.sum() for k,v in cnt.items()}
base=np.bincount(roles.ravel(),minlength=R)/roles.size
print(f"{anchor} anchor; 64 population networks; gate flips counted per (draw, step, neuron)")
print(f"  {'role':12s} {'neurons':>8s} {'single flips':>13s} {'shared flips':>13s} {'shared flips':>13s}")
print(f"  {'':12s} {'':>8s} {'(ordinary)':>13s} {'(ordinary)':>13s} {'(sign errors)':>13s}")
for r in range(R):
    print(f"  {ROLES[r]:12s} {base[r]:8.3f} {share['single_ord'][r].sum():13.3f} {share['shared_ord'][r].sum():13.3f} {share['shared_err'][r].sum():13.3f}")
print("  by phase (prefix / middle / late):")
for k in ('single_ord','shared_ord','shared_err'): print(f"    {k:11s} "+" / ".join(f"{x:.3f}" for x in share[k].sum(0)))
np.savez(f'pop1k/seed0_z19_{anchor}.npz',**cnt)
