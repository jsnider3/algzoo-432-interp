"""Why is z13's margin-additive model (Pmarg) biased low?  Test: additivity per rival instead of in the min-over-rivals
margin.  Pvec: each rival difference D_o = L_y - L_o is additive across the 6 lower inputs, and the input is correct iff
all nine D_o > 0, evaluated exactly over the q^6 node combinations by brute force (q = 8).  If Pmarg were low because two
lower inputs push different rivals (the scalar margin then double-counts), Pvec would remove the bias; it does not.
Every 16th population network + the 6 zoo models.  Usage: python z14.py [configurations, default 2048] [nin, default 256]
"""
import sys, numpy as np, torch as th
sys.path.insert(0,'.')
from batch_train import forward, target
from z2 import load
from z11 import expansion
from seeds_clocks import load as zload
dev='cuda'; nc=int(sys.argv[1]) if len(sys.argv)>1 else 2048; nin=int(sys.argv[2]) if len(sys.argv)>2 else 256; q=8; k=4
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]
x=th.randn(nc,10,generator=th.Generator().manual_seed(31)).to(dev); gi=th.Generator(device=dev).manual_seed(37)
keys=['p','I0','P1','Pmarg','Pvec']; acc={a:[] for a in keys}
def pairsum(d,i,j): return (d[:,:,i,:,None]+d[:,:,j,None,:]).flatten(2,3)            # (N,b,q*q,...) sums of two inputs' deltas
with th.no_grad():
    B=4
    for i in range(0,nc,B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4])
        X,_=expansion(xc,k,q); R=X.shape[1]
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,R,10); yi=y[None,:,None,None].expand(N,b,R,1)
        I=(L.argmax(-1)==y[None,:,None]).float(); I0=I[:,:,0]; E=I[:,:,1:].view(N,b,6,q).mean(-1); P1=(I0+(E-I0[...,None]).sum(-1)).clamp(0,1)
        Dv=L.gather(-1,yi)-L; Dv=Dv.scatter(-1,yi,1e30)                                  # (N,b,R,10) target minus each rival
        a=Dv[:,:,0]; d=(Dv[:,:,1:]-a[:,:,None]).view(N,b,6,q,10)
        AB=(pairsum(d,0,1)[:,:,:,None]+pairsum(d,2,3)[:,:,None]).flatten(2,3)            # (N,b,q^4,10)
        C=pairsum(d,4,5)                                                                 # (N,b,q^2,10)
        ok=th.ones(N,b,q**4,q**2,dtype=th.bool,device=dev)
        m0=Dv[:,:,0].min(-1).values; dm=(Dv.min(-1).values[:,:,1:]-m0[...,None]).view(N,b,6,q)
        okm=((pairsum(dm,0,1)[:,:,:,None]+pairsum(dm,2,3)[:,:,None]).flatten(2,3)[...,None]+pairsum(dm,4,5)[:,:,None]+m0[:,:,None,None])>0
        for oo in range(10):
            ok&=(a[:,:,None,None,oo]+AB[...,None,oo]+C[:,:,None,:,oo])>0
        Pv=ok.float().mean((-1,-2)); Pm=okm.float().mean((-1,-2))
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; low=th.special.ndtri(u.clamp(min=1e-30))
        Xi=xc[:,None].repeat(1,nin,1); Xi.scatter_(-1,o[:,None,4:].expand(b,nin,6),low)
        p=(forward(*W,Xi.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,nin,10).argmax(-1)==y[None,:,None]).float().mean(-1)
        for kk,v in zip(keys,(p,I0,P1,Pm,Pv)): acc[kk].append(v)
r={kk:th.cat(v,1) for kk,v in acc.items()}; p=r['p']; noise=p*(1-p)/(nin-1); pq=p.mean(1)*(1-p.mean(1)); pop=slice(0,N-6)
def show(lab,v,fmt):
    v=v.cpu().numpy(); print(f"  {lab:40s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
print(f"{N-6} population networks (every 16th) + 6 zoo; {nc} top-4 configurations, {nin} lower-input draws each; q = {q}")
for kk in keys[1:]: show(f"{kk:6s} local MSE / p(1-p)",(((r[kk]-p)**2-noise).mean(1))/pq,"{:.4f}")
for kk in keys[1:]: show(f"{kk:6s} bias, points",100*(r[kk]-p).mean(1),"{:+.3f}")
show("Pvec - Pmarg, points",100*(r['Pvec']-r['Pmarg']).mean(1),"{:+.3f}")
