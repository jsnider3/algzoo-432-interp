"""Does averaging each input's effect over the others' spread (first-order ANOVA) remove the margin model's bias?
z15: first-order cut-HDMR in the margin (each lower input moved alone, the others at their mean) is -0.18 points low,
and the cause is double counting (z16: gates flipped by two inputs, rival take-overs).  The anchor choice does not
matter (mean -0.183, median -0.179).  The ANOVA first-order term of input j is g_j(z_j) = E[m | z_j], averaged over the
other five inputs' law, so an effect that another input usually pre-empts is counted at its average size:
  mA = Ebar + sum_j (g_j(z_j) - Ebar),  Ebar = E[m]
Here g_j at each draw's own z_j is averaged over K background draws of the other five (the next K draws of the same
configuration, so independent of it): an idealized, sampled test of whether the ANOVA form fixes the sign bias.
K finite adds noise to mA; K = 16 and 32 are compared.  Every 16th population network + the 6 zoo models.
Usage: python scripts/z17.py [configurations, default 2048] [nin, default 32] [K, default 32]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; nc=int(sys.argv[1]) if len(sys.argv)>1 else 2048; nin=int(sys.argv[2]) if len(sys.argv)>2 else 32; K=int(sys.argv[3]) if len(sys.argv)>3 else 32
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]
x=th.randn(nc,10,generator=th.Generator().manual_seed(31)).to(dev); gi=th.Generator(device=dev).manual_seed(43)
def margin(L,y):
    yy=y[None,:,None,None].expand(*L.shape[:-1],1)
    return (L.gather(-1,yy)-L.scatter(-1,yy,-1e30).max(-1,keepdim=True).values)[...,0]
nb=max(K,nin)                                                                        # background pool per configuration
acc=th.zeros(N,8,device=dev,dtype=th.float64)
with th.no_grad():
    B=2
    for i in range(0,nc,B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        base=mu_below(v4).expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        u=th.rand(b,nin+nb,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        zd,zb=zz[:,:nin],zz[:,nin:]                                                  # draws and an independent background pool
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zd)
        ring=(th.arange(nin,device=dev)[:,None]+th.arange(K,device=dev)[None])%nb       # draw k uses backgrounds k..k+K-1
        bg=zb[:,ring]                                                                # (b, nin, K, 6)
        anova=bg[:,:,None].repeat(1,1,6,1,1)                                          # (b, nin, j, K, 6): others from backgrounds
        eye=th.eye(6,device=dev,dtype=th.bool)[None,None,:,None,:]
        anova=th.where(eye,zd[:,:,None,None,:].expand_as(anova),anova)              # input j at the draw's own value
        rows=base[:,None,None,None].repeat(1,nin,6,K,1); rows.scatter_(-1,lp[:,None,None,None].expand(b,nin,6,K,6),anova)
        bgr=base[:,None].repeat(1,nb,1); bgr.scatter_(-1,lp[:,None].expand(b,nb,6),zb)
        single=base[:,None,None].repeat(1,nin,6,1); single.scatter_(-1,lp[:,None,:,None].expand(b,nin,6,1),zd[...,None])
        X=th.cat([base[:,None],full,bgr,single.reshape(b,-1,10),rows.reshape(b,-1,10)],1)
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,-1,10); M=margin(L,y); c=0
        m0=M[:,:,0]; c=1; mt=M[:,:,c:c+nin]; It=(L[:,:,c:c+nin].argmax(-1)==y[None,:,None]); c+=nin
        Eb=M[:,:,c:c+nb].mean(-1); c+=nb
        ms=M[:,:,c:c+nin*6].view(N,b,nin,6); c+=nin*6
        g=M[:,:,c:].view(N,b,nin,6,K).mean(-1)
        m1=m0[...,None]+(ms-m0[...,None,None]).sum(-1); mA=Eb[...,None]+(g-Eb[...,None,None]).sum(-1)
        I1=m1>0; IA=mA>0
        acc+=th.stack([It.sum((1,2)),I1.sum((1,2)),IA.sum((1,2)),(It&~IA).sum((1,2)),(~It&IA).sum((1,2)),
                       ((mt-mA)**2).sum((1,2)),((mt-m1)**2).sum((1,2)),(mt-mA).sum((1,2))],1).double()
r=(acc/(nc*nin)).cpu().numpy(); pop=slice(0,N-6)
def show(lab,v,fmt):
    print(f"  {lab:50s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
print(f"{N-6} population networks (every 16th) + 6 zoo; {nc} configurations x {nin} draws; ANOVA terms from K = {K} backgrounds")
show("cut-HDMR first order (anchor at the mean): bias",100*(r[:,1]-r[:,0]),"{:+.3f}")
show("ANOVA first order: bias",100*(r[:,2]-r[:,0]),"{:+.3f}")
show("  truth right, model wrong (% of draws)",100*r[:,3],"{:.3f}")
show("  truth wrong, model right (% of draws)",100*r[:,4],"{:.3f}")
show("margin residual RMS: cut-HDMR",np.sqrt(r[:,6]),"{:.3f}")
show("margin residual RMS: ANOVA",np.sqrt(r[:,5]),"{:.3f}")
show("margin residual mean: ANOVA",r[:,7],"{:+.4f}")
