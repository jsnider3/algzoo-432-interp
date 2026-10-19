"""A bound for every additive lower-input model.  The nearby-pair and triple models (z18, z32) are truncated
approximations of one ideal: the exact first-order ANOVA projection of the margin,
  mA(z) = E[m] + sum_j (E[m | z_j] - E[m]),
which is the best (L2) sum of one-input effects.  Whatever pairs, triples or spans are kept, a model of the form
P(m0' + sum_j h_j(z_j) > 0) approximates P(mA > 0) at best.  So the accuracy error of the ideal itself, across networks,
bounds the whole family.  Here E[m | z_j] is computed with K background draws of the other five inputs shared with E[m]
(common random numbers; z17 showed K = 32 and 64 agree), on 2^lc configurations x nin draws, per network, with the
measurement noise removed.  Every 16th population network + the 6 zoo models.
Usage: python z34.py [log2 configurations, default 13] [nin, default 32] [K, default 32]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; lc=int(sys.argv[1]) if len(sys.argv)>1 else 13; nin=int(sys.argv[2]) if len(sys.argv)>2 else 32; K=int(sys.argv[3]) if len(sys.argv)>3 else 32
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
x=th.randn(2**lc,10,generator=th.Generator().manual_seed(131)).to(dev); gi=th.Generator(device=dev).manual_seed(137)
def margin(L,y):
    yy=y[None,:,None,None].expand(*L.shape[:-1],1)
    return (L.gather(-1,yy)-L.scatter(-1,yy,-1e30).max(-1,keepdim=True).values)[...,0]
nb=K; S1=th.zeros(N,dtype=th.float64,device=dev); S2=th.zeros(N,dtype=th.float64,device=dev); A1=th.zeros(N,dtype=th.float64,device=dev)
with th.no_grad():
    B=2
    for i in range(0,len(x),B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        base=mu_below(v4).expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        u=th.rand(b,nin+nb,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        zd,zb=zz[:,:nin],zz[:,nin:]
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zd)
        bg=zb[:,None,None].expand(b,nin,6,nb,6).clone()                               # (b, draw, j, background, input)
        eye=th.eye(6,device=dev,dtype=th.bool)[None,None,:,None,:]
        bg=th.where(eye,zd[:,:,None,None,:].expand_as(bg),bg)
        rows=base[:,None,None,None].repeat(1,nin,6,nb,1); rows.scatter_(-1,lp[:,None,None,None].expand(b,nin,6,nb,6),bg)
        bgr=base[:,None].repeat(1,nb,1); bgr.scatter_(-1,lp[:,None].expand(b,nb,6),zb)
        X=th.cat([full,bgr,rows.reshape(b,-1,10)],1)
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,-1,10); M=margin(L,y)
        It=(L[:,:,:nin].argmax(-1)==y[None,:,None]).float(); Eb=M[:,:,nin:nin+nb].mean(-1)
        g=M[:,:,nin+nb:].view(N,b,nin,6,nb).mean(-1)
        mA=Eb[...,None]+(g-Eb[...,None,None]).sum(-1); IA=(mA>0).float()
        e=(IA-It).mean(-1).double()                                                     # per configuration
        S1+=e.sum(1); S2+=(e*e).sum(1); A1+=(IA!=It).float().mean(-1).double().sum(1)
n=len(x); bb=(S1/n).cpu().numpy(); se2=((S2/n).cpu().numpy()-bb**2)/n; bp=100*bb[pop]
print(f"{N-6} population networks (every 16th) + 6 zoo; exact first-order ANOVA (additive) margin model, K = {K} shared backgrounds;"
      f" 2^{lc} configurations x {nin} draws; points")
print(f"  bias mean {bp.mean():+.4f}  median {np.median(bp):+.4f}  SD {bp.std():.4f}  RMS {np.sqrt((bp**2).mean()):.4f}  noise {100*np.sqrt(se2[pop].mean()):.4f}"
      f"  noise-subtracted RMS {np.sqrt(max(0,(bp**2).mean()-1e4*se2[pop].mean())):.4f}   disagreement {100*np.median((A1/n).cpu().numpy()[pop]):.2f}%"
      f"   zoo "+" ".join(f"{100*v:+.3f}" for v in bb[-6:]))
np.savez('pop1k/seed0_z34.npz',bias=bb,se2=se2)
