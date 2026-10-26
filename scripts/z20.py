"""Resolve the clamped indicator model's total bias per network.  (z18 on all 1,024: its per-network measurement noise was
0.20 points at 2,048 configurations, so its RMS across networks was only bounded, 0-0.05.)  Scored paired against each
configuration's own conditional accuracy p(c) (nin lower-input draws), whose noise is far below that of the model's local
error, on 2^18 configurations: per-network noise about 0.02 points.
Models: P1 = clip(I0 + sum_j (E_j - I0)) with E_j at q = 16 equal-probability midpoints (z11), or at q = 8 Gauss-Legendre
nodes in the CDF variable (z18), and the unclamped sum.  Every 4th population network + the 6 zoo models.
Usage: python scripts/z20.py [log2 configurations, default 18] [nin, default 16] [--all: every population network]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; every='--all' in sys.argv; sys.argv=[a for a in sys.argv if a!='--all']
m=int(sys.argv[1]) if len(sys.argv)>1 else 18; nin=int(sys.argv[2]) if len(sys.argv)>2 else 16
P,_=load('pop1k/seed0.npz'); P=[w if every else w[::4] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]
x=th.randn(2**m,10,generator=th.Generator().manual_seed(61)).to(dev); gi=th.Generator(device=dev).manual_seed(67)
ug,wg=np.polynomial.legendre.leggauss(8); UG=th.tensor((ug+1)/2,dtype=th.float32,device=dev); WG=th.tensor(wg/2,dtype=th.float32,device=dev)
UM=(th.arange(16,device=dev)+.5)/16
keys=['mid16','gl8','raw_mid16']; S1={k:th.zeros(N,dtype=th.float64,device=dev) for k in keys}; S2={k:th.zeros(N,dtype=th.float64,device=dev) for k in keys}
with th.no_grad():
    B=48
    for i in range(0,len(x),B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        base=mu_below(v4).expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        rows=[base[:,None]]
        for U in (UM,UG):
            nodes=th.special.ndtri(th.special.ndtr(v4)*U[None]); q=len(U)
            for j in range(6):
                r=base[:,None].repeat(1,q,1); r.scatter_(-1,lp[:,None,j:j+1].expand(b,q,1),nodes[...,None]); rows.append(r)
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zz); rows.append(full)
        X=th.cat(rows,1); I=(forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,-1,10).argmax(-1)==y[None,:,None]).float()
        I0=I[:,:,0]; Em=I[:,:,1:97].view(N,b,6,16).mean(-1); Eg=(I[:,:,97:145].view(N,b,6,8)*WG).sum(-1); p=I[:,:,145:].mean(-1)
        raw=I0+(Em-I0[...,None]).sum(-1)
        for k,v in (('mid16',raw.clamp(0,1)),('gl8',(I0+(Eg-I0[...,None]).sum(-1)).clamp(0,1)),('raw_mid16',raw)):
            e=(v-p).double(); S1[k]+=e.sum(1); S2[k]+=(e*e).sum(1)
n=len(x); pop=slice(0,N-6)
print(f"{N-6} population networks ({'all' if every else 'every 4th'}) + 6 zoo; 2^{m} configurations x {nin} lower-input draws; points")
for k in keys:
    b=(S1[k]/n).cpu().numpy(); se2=((S2[k]/n).cpu().numpy()-b**2)/n; bp=b[pop]
    print(f"  {k:10s} bias mean {100*bp.mean():+.3f}  median {100*np.median(bp):+.3f}  SD {100*bp.std():.3f}  RMS {100*np.sqrt((bp**2).mean()):.3f}"
          f"  noise {100*np.sqrt(se2[pop].mean()):.3f}  noise-subtracted RMS {100*np.sqrt(max(0,(bp**2).mean()-se2[pop].mean())):.3f}"
          f"  zoo "+" ".join(f"{100*v:+.3f}" for v in b[-6:]))
np.savez(f"pop1k/seed0_z20{'_all' if every else ''}.npz",**{k:(S1[k]/n).cpu().numpy() for k in keys},**{k+'_m2':(S2[k]/n).cpu().numpy() for k in keys},n=n)
