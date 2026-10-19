"""Mechanism of the margin model's non-additivity, on the draws saved by z15.py.
Within one linear region of the network the margin is exactly additive in the lower inputs (it is affine there, and the
winning rival is fixed), so every residual r = m - m1 comes from ReLU gates or the winning rival differing between the
joint run and the six one-input runs.  For each saved draw this classifies the gates (time step x neuron) that differ from
the anchor run:
  shared   flipped by two or more one-input runs (the additive model counts the flip's effect more than once)
  new      flipped in the joint run but in no one-input run
  lost     flipped by a one-input run but not in the joint run
and records whether the winning rival differs.  Sign errors of the first-order model are compared with ordinary draws
(the every-32nd subsample).  Usage: python z16.py [anchor, default mean]
"""
import sys, numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
dev='cuda'; anchor=sys.argv[1] if len(sys.argv)>1 else 'mean'
z=np.load(f'pop1k/seed0_z15_{anchor}.npz'); names=list(z['names'])
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Zm=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([q[i] for q in Zm]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]

def run(w, X):
    """X (n, 10) -> gates (n, 10, 16) bool, logits (n, 10)"""
    h=th.zeros(X.shape[0],16,device=dev); G=[]
    for t in range(10):
        pre=X[:,t:t+1]*w[0]+h@w[1].T; G.append(pre>0); h=th.relu(pre)
    return th.stack(G,1), h@w[2].T

m=z['m']; d=z['d']; m0=z['m0']; m1=m0+d.sum(1); It=z['It']; err=(m1>0)!=It; sub=z['sub']
out={k:[] for k in ('net','err','sub','shared','new','lost','rival','tshared','nshared','r','sd','fn')}
for n in range(N):
    k=np.nonzero(z['net']==n)[0]
    if len(k)==0: continue
    lp=th.tensor(z['lp'][k],device=dev); top=th.tensor(z['top'][k],device=dev); v=th.tensor(z['v'][k],device=dev,dtype=th.float32)
    a=th.tensor(z['a'][k],device=dev,dtype=th.float32); zz=th.tensor(z['z'][k],device=dev,dtype=th.float32); B=len(k)
    base=a[:,None].expand(B,10).clone(); base.scatter_(-1,top,v)
    joint=base.clone(); joint.scatter_(-1,lp,zz)
    singles=base[:,None].repeat(1,6,1); singles.scatter_(-1,lp[:,:,None],zz[:,:,None])
    w=[q[n] for q in W]; y=top[:,1]
    G0,L0=run(w,base); GJ,LJ=run(w,joint); GS,LS=run(w,singles.reshape(-1,10)); GS=GS.view(B,6,10,16); LS=LS.view(B,6,10)
    fs=GS!=G0[:,None]; fj=GJ!=G0; cnt=fs.sum(1)
    def rival(L): return L.scatter(-1,y.view(-1,*([1]*(L.dim()-1))).expand(*L.shape[:-1],1),-1e30).argmax(-1)
    rv=(rival(LJ)!=rival(L0))|(rival(LS)!=rival(L0)[:,None]).any(1)|(rival(LS)!=rival(LJ)[:,None]).any(1)
    sh=(cnt>=2); tsh=th.where(sh.any(2),th.arange(10,device=dev)[None],99).min(1).values      # first step with a shared flip
    out['net'].append(np.full(B,n)); out['err'].append(err[k]); out['sub'].append(sub[k]); out['fn'].append(It[k]&~(m1[k]>0))
    out['shared'].append(sh.sum((1,2)).cpu().numpy()); out['new'].append((fj&(cnt==0)).sum((1,2)).cpu().numpy())
    out['lost'].append((~fj&(cnt>=1)).sum((1,2)).cpu().numpy()); out['rival'].append(rv.cpu().numpy())
    out['tshared'].append(tsh.cpu().numpy()); out['nshared'].append(sh.any(1).sum(1).cpu().numpy())
    out['r'].append(m[k]-m1[k]); out['sd'].append(d[k].sum(1))
o={k:np.concatenate(v) for k,v in out.items()}; pop=o['net']<N-6
def frac(mask,lab):
    e=mask&o['err']; s=mask&o['sub']
    print(f"  {lab:46s} sign errors {e.sum()/max(1,(o['err']&pop).sum()):.3f}   ordinary draws {s.sum()/max(1,(o['sub']&pop).sum()):.3f}")
print(f"{anchor} anchor; population networks; first-order sign errors {(o['err']&pop).sum()}, ordinary draws {(o['sub']&pop).sum()}")
print(" share of draws in which ...")
P_=pop
frac(P_&(o['shared']==0)&(o['new']==0)&(o['lost']==0)&~o['rival'],"no gate or rival differs (exactly additive)")
frac(P_&(o['shared']>0),"some gate is flipped by >= 2 one-input runs")
frac(P_&(o['new']>0),"some gate flips only jointly")
frac(P_&(o['lost']>0),"some one-input flip is undone jointly")
frac(P_&o['rival'],"the winning rival differs")
frac(P_&(o['shared']>0)&(o['new']==0)&~o['rival'],"shared flips only (no new gate, same rival)")
e=P_&o['err']
print(" among sign errors: truth right / model wrong share", f"{o['fn'][e].mean():.3f}")
print(" first step with a shared flip (sign errors):", np.bincount(np.minimum(o['tshared'][e],10),minlength=11))
print(" first step with a shared flip (ordinary)   :", np.bincount(np.minimum(o['tshared'][P_&o['sub']],10),minlength=11))
print(" number of neurons with a shared flip (sign errors):", np.bincount(o['nshared'][e],minlength=8)[:8])
np.savez(f'pop1k/seed0_z16_{anchor}.npz',**o)
