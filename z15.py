"""Where does the margin-additive model's bias come from?  (z13: locally 12x better than the indicator model, but about
-0.15 points low, growing with q.)  Here the model is checked draw by draw, with no node grid.
For each top-4 configuration c (sampled) and each of nin draws z of the 6 lower inputs (iid N(0,1) below the 4th largest):
  m      the true margin L_y - max_{o != y} L_o at (c, z)
  m0     the margin at the anchor (all 6 lower inputs at `anchor`: their conditional mean, or median)
  d_j    m(anchor with input j at z_j) - m0, exactly (one forward each)
  m1     m0 + sum_j d_j                          (first order in the margin; its P(m1 > 0) is z13's Pmarg as q -> inf)
  m2     m1 + sum_{j<l} (m(z_j, z_l) - m(z_j) - m(z_l) + m0)   (second order, 15 more forwards per draw)
Reported per network: bias of 1[m1 > 0] and 1[m2 > 0] against the truth on the same draws (paired, so nearly noise-free),
the two kinds of sign disagreement, and the residual r = m - m1.  The disagreeing draws (plus every 32nd draw) are saved
for the mechanism analysis in z16.py.  Every 16th population network + the 6 zoo models.
Usage: python z15.py [configurations, default 2048] [nin, default 64] [anchor: mean | median]
"""
import sys, itertools, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; nc=int(sys.argv[1]) if len(sys.argv)>1 else 2048; nin=int(sys.argv[2]) if len(sys.argv)>2 else 64
anchor=sys.argv[3] if len(sys.argv)>3 else 'mean'
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; names=[f"pop{16*i}" for i in range(N-6)]+["mystery"]+[f"seed{i}" for i in range(5)]
x=th.randn(nc,10,generator=th.Generator().manual_seed(31)).to(dev); gi=th.Generator(device=dev).manual_seed(41)
PAIRS=list(itertools.combinations(range(6),2))

def margin(L,y):
    """L (N, b, R, 10), y (b,) -> L_y - max over the other outputs, (N, b, R)"""
    yy=y[None,:,None,None].expand(*L.shape[:-1],1)
    return (L.gather(-1,yy)-L.scatter(-1,yy,-1e30).max(-1,keepdim=True).values)[...,0]

acc=th.zeros(N,12,device=dev,dtype=th.float64); rec=[]
with th.no_grad():
    B=8
    for i in range(0,nc,B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        a=mu_below(v4) if anchor=='mean' else th.special.ndtri(th.special.ndtr(v4)/2)
        base=a.expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        rows=[base[:,None]]                                                          # anchor
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zz); rows.append(full)   # truth
        for j in range(6):
            r=base[:,None].repeat(1,nin,1); r.scatter_(-1,lp[:,None,j:j+1].expand(b,nin,1),zz[...,j:j+1]); rows.append(r)
        for j,l in PAIRS:
            r=base[:,None].repeat(1,nin,1); r.scatter_(-1,lp[:,None,[j,l]].expand(b,nin,2),zz[...,[j,l]]); rows.append(r)
        X=th.cat(rows,1); R=X.shape[1]
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,R,10)
        M=margin(L,y); m0=M[:,:,0]; mt=M[:,:,1:1+nin]
        It=(L[:,:,1:1+nin].argmax(-1)==y[None,:,None])                               # truth by argmax (ties as argmax resolves them)
        ms=M[:,:,1+nin:1+7*nin].view(N,b,6,nin).transpose(2,3)                       # (N,b,nin,6)
        mp=M[:,:,1+7*nin:].view(N,b,15,nin).transpose(2,3)                           # (N,b,nin,15)
        d=ms-m0[...,None,None]; m1=m0[...,None]+d.sum(-1)
        jj=th.tensor([p[0] for p in PAIRS],device=dev); ll=th.tensor([p[1] for p in PAIRS],device=dev)
        inter=mp-ms[...,jj]-ms[...,ll]+m0[...,None,None]; m2=m1+inter.sum(-1)
        I1=m1>0; I2=m2>0; res=mt-m1
        acc+=th.stack([It.sum((1,2)),I1.sum((1,2)),I2.sum((1,2)),(It&~I1).sum((1,2)),(~It&I1).sum((1,2)),
                       (It&~I2).sum((1,2)),(~It&I2).sum((1,2)),res.sum((1,2)),(res*res).sum((1,2)),(mt-m2).abs().sum((1,2)),
                       ((I1!=It)&(I2==It)).sum((1,2)),((I1==It)&(I2!=It)).sum((1,2))],1).double()
        keep=(I1!=It)|(I2!=It); keep[:,:,::32]=True                                  # disagreements + every 32nd draw
        n_,c_,k_=keep.nonzero(as_tuple=True)
        rec.append(dict(net=n_.cpu(),cfg=(c_+i).cpu(),draw=k_.cpu(),It=It[n_,c_,k_].cpu(),m=mt[n_,c_,k_].cpu(),m0=m0[n_,c_].cpu(),
                        d=d[n_,c_,k_].cpu(),inter=inter[n_,c_,k_].cpu(),z=zz[c_,k_].cpu(),lp=lp[c_].cpu(),top=o[c_,:4].cpu(),
                        v=xc.gather(-1,o[:,:4])[c_].cpu(),a=a[c_,0].cpu(),sub=(k_%32==0).cpu()))
n=nc*nin; r=(acc/n).cpu().numpy(); pop=slice(0,N-6)
def show(lab,v,fmt):
    print(f"  {lab:56s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
print(f"{N-6} population networks (every 16th) + 6 zoo; {nc} top-4 configurations x {nin} lower-input draws; anchor at the {anchor}")
show("accuracy on these draws",r[:,0],"{:.4f}")
show("first order in the margin, exact: bias (points)",100*(r[:,1]-r[:,0]),"{:+.3f}")
show("  truth correct, model wrong (% of draws)",100*r[:,3],"{:.3f}")
show("  truth wrong, model correct (% of draws)",100*r[:,4],"{:.3f}")
show("second order in the margin: bias (points)",100*(r[:,2]-r[:,0]),"{:+.3f}")
show("  truth correct, model wrong (% of draws)",100*r[:,5],"{:.3f}")
show("  truth wrong, model correct (% of draws)",100*r[:,6],"{:.3f}")
show("  share of first-order sign errors fixed by second order",r[:,10]/(r[:,3]+r[:,4]),"{:.3f}")
show("  new sign errors made by second order (% of draws)",100*r[:,11],"{:.3f}")
show("residual m - m1: mean",r[:,7],"{:+.4f}")
show("residual m - m1: RMS",np.sqrt(r[:,8]),"{:.4f}")
show("residual m - m2: mean |.|",r[:,9],"{:.4f}")
cat={k:th.cat([q[k] for q in rec]) for k in rec[0]}
np.savez(f'pop1k/seed0_z15_{anchor}.npz',**{k:v.numpy() for k,v in cat.items()},names=np.array(names),nc=nc,nin=nin,summary=r)
