"""Joint (non-additive) lower-input models: a product Gauss-Legendre rule over all six lower inputs at once.
Additive models (z18, z32) cannot reach about 0.01 points (z34 bounds the whole family by its ideal).  The direct joint
alternative evaluates the network on the full q^6 product grid of the six lower inputs (nodes in each input's CDF,
Gauss-Legendre weights) and sums the weighted indicators: P(correct | c) without any additivity assumption, at q^6
forwards per configuration (64 at q = 2, 729 at q = 3).  Scored like z18 (against p(c) from nin draws per configuration,
per-network noise removed).  Every 16th population network + the 6 zoo models.
Usage: python scripts/z36.py [log2 configurations, default 12] [nin, default 64]
"""
import sys, itertools, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; lc=int(sys.argv[1]) if len(sys.argv)>1 else 12; nin=int(sys.argv[2]) if len(sys.argv)>2 else 64
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
x=th.randn(2**lc,10,generator=th.Generator().manual_seed(139)).to(dev); gi=th.Generator(device=dev).manual_seed(149)
QS=(2,3); rules={}
for q in QS:
    ug,wg=np.polynomial.legendre.leggauss(q); U=(ug+1)/2; wq=wg/2
    grid=np.array(list(itertools.product(range(q),repeat=6))); rules[q]=(th.tensor(U[grid],dtype=th.float32,device=dev),th.tensor(np.prod(wq[grid],1),dtype=th.float32,device=dev))
S1={q:th.zeros(N,dtype=th.float64,device=dev) for q in QS}; S2={q:th.zeros(N,dtype=th.float64,device=dev) for q in QS}
with th.no_grad():
    B=4
    for i in range(0,len(x),B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        base=mu_below(v4).expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        rows=[]; sizes=[]
        for q in QS:
            Ug,_=rules[q]; zg=th.special.ndtri(th.special.ndtr(v4)[:,None,:]*Ug[None])         # (b, q^6, 6)
            r=base[:,None].repeat(1,len(Ug),1); r.scatter_(-1,lp[:,None].expand(b,len(Ug),6),zg); rows.append(r); sizes.append(len(Ug))
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zz); rows.append(full)
        X=th.cat(rows,1); I=(forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,-1,10).argmax(-1)==y[None,:,None]).float()
        c=0; Pq={}
        for q,sz in zip(QS,sizes):
            Pq[q]=(I[:,:,c:c+sz]*rules[q][1]).sum(-1); c+=sz
        p=I[:,:,c:].mean(-1)
        for q in QS:
            e=(Pq[q]-p).double(); S1[q]+=e.sum(1); S2[q]+=(e*e).sum(1)
n=len(x)
print(f"{N-6} population networks (every 16th) + 6 zoo; joint product rule over the six lower inputs; 2^{lc} configurations x {nin} draws; points")
for q in QS:
    bb=(S1[q]/n).cpu().numpy(); se2=((S2[q]/n).cpu().numpy()-bb**2)/n; bp=100*bb[pop]
    print(f"  q = {q} ({q**6} forwards per configuration): bias mean {bp.mean():+.3f}  RMS {np.sqrt((bp**2).mean()):.3f}  noise {100*np.sqrt(se2[pop].mean()):.3f}"
          f"  noise-subtracted RMS {np.sqrt(max(0,(bp**2).mean()-1e4*se2[pop].mean())):.3f}  zoo "+" ".join(f"{100*v:+.3f}" for v in bb[-6:]))
