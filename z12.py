"""What the top-4 integrand looks like: the MF4 + first-order leakage model (z11) as a function of the top-4 configuration.

z11 scores P(correct | top 4) = I0 + sum_j (E_j[I] - I0) against the truth; the open problem is integrating it over the
top-4 configurations (5,040 arrangements x 4 values) at a cost below sampling's. This describes that integrand, on a sampled
pool of configurations (sampling only to describe it): the variance of I0, of P and of the penalty P - I0 relative to
p(1-p); how often the penalty is nonzero; how its square and its mean concentrate near the MF4 decision boundary (the margin
between the target logit and the best other logit at the MF4 point); and how much of each variance lies between
arrangements (ANOVA over the 5,040 position arrangements of the top 4) rather than within them, over the values.
Every 16th population network + the 6 zoo models; q = 16 nodes (z11.expansion). Output: pop1k/z12_seed0.txt.
Usage: python z12.py [log2 pool]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z11 import expansion
from seeds_clocks import load as zload
dev='cuda'; m=int(sys.argv[1]) if len(sys.argv)>1 else 17; q=16; k=4
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; names=[f"pop{16*i}" for i in range(N-6)]+["mystery"]+[f"seed{i}" for i in range(5)]
x=th.randn(2**m,10,generator=th.Generator().manual_seed(23)).to(dev)
o=x.argsort(-1,descending=True)[:,:4]; arr=(o[:,0]*1000+o[:,1]*100+o[:,2]*10+o[:,3])        # arrangement id
_,gid=th.unique(arr,return_inverse=True); G=int(gid.max())+1
tr_all=[];I0_all=[];P_all=[];mg_all=[]
with th.no_grad():
    B=144
    for i in range(0,len(x),B):
        xc=x[i:i+B]; b=len(xc); y=target(xc)
        tr=(forward(*W,xc[None].expand(N,-1,-1)).argmax(-1)==y)
        X,_=expansion(xc,k,q); R=X.shape[1]
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,R,10)
        I=(L.argmax(-1)==y[None,:,None]).float()
        I0=I[:,:,0]; Ej=I[:,:,1:].view(N,b,10-k,q).mean(-1); Pm=(I0+(Ej-I0[...,None]).sum(-1)).clamp(0,1)
        L0=L[:,:,0]; Ly=L0.gather(-1,y[None,:,None].expand(N,b,1))[...,0]; Lo=L0.scatter(-1,y[None,:,None].expand(N,b,1),-1e9).max(-1).values
        tr_all.append(tr.float()); I0_all.append(I0); P_all.append(Pm); mg_all.append(Ly-Lo)
T=th.cat(tr_all,1); I0=th.cat(I0_all,1); Pm=th.cat(P_all,1); MG=th.cat(mg_all,1); pen=Pm-I0; n=T.shape[1]
def between(v):
    """ANOVA over arrangements: share of Var(v) between arrangement means (unbiased for the finite groups)"""
    cnt=th.zeros(G,device=dev).index_add_(0,gid,th.ones(n,device=dev))
    s1=th.zeros(N,G,device=dev).index_add_(1,gid,v); s2=th.zeros(N,G,device=dev).index_add_(1,gid,v*v)
    mu=s1/cnt.clamp(min=1); wv=(s2-s1*mu)/(cnt-1).clamp(min=1)                         # within-group sample variance
    tot=v.var(1); within=(wv*cnt).sum(1)/n                                              # E[Var(v | arrangement)]
    return (1-within/tot.clamp(min=1e-12))
res={}
res['p']=T.mean(1); res['pq']=res['p']*(1-res['p'])
res['I0']=I0.mean(1); res['P']=Pm.mean(1); res['pen']=pen.mean(1)
res['varI0']=I0.var(1); res['varP']=Pm.var(1); res['varpen']=pen.var(1)
res['nz']=(pen.abs()>1e-9).float().mean(1)
e2=(pen*pen).sum(1).clamp(min=1e-12); am=MG.abs()
for c in (0.25,.5,1,2,4): res[f'pen2<{c}']=((pen*pen)*(am<c).float()).sum(1)/e2; res[f'mass<{c}']=(am<c).float().mean(1)
for c in (0.25,.5,1,2,4): res[f'pen<{c}']=(pen*(am<c).float()).sum(1)/pen.sum(1)
res['btw_I0']=between(I0); res['btw_P']=between(Pm); res['btw_pen']=between(pen); res['btw_T']=between(T)
r={a:v.cpu().numpy() for a,v in res.items()}; pop=slice(0,N-6)
print(f"{N-6} population networks (every 16th) + 6 zoo; 2^{m} top-4 configurations (sampled, to describe the integrand); q = {q}; {G} arrangements seen")
def show(lab,f,fmt="{:.4f}"):
    v=f(r); print(f"  {lab:58s} pop median {fmt.format(np.median(v[pop]))}  [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]   zoo "+" ".join(fmt.format(a) for a in v[-6:]))
show("accuracy",lambda r:r['p'])
show("Var(I0) / p(1-p)",lambda r:r['varI0']/r['pq'],"{:.3f}")
show("Var(P) / p(1-p)   (P = I0 + first-order penalty)",lambda r:r['varP']/r['pq'],"{:.3f}")
show("Var(penalty) / p(1-p)",lambda r:r['varpen']/r['pq'],"{:.4f}")
show("mean penalty (points)",lambda r:100*r['pen'],"{:+.3f}")
show("share of configurations with a nonzero penalty",lambda r:r['nz'],"{:.3f}")
for c in (0.25,.5,1,2,4):
    show(f"|MF4 margin| < {c}: share of configurations",lambda r,c=c:r[f'mass<{c}'],"{:.3f}")
    show(f"     share of E[pen^2]",lambda r,c=c:r[f'pen2<{c}'],"{:.3f}"); show(f"     share of mean penalty",lambda r,c=c:r[f'pen<{c}'],"{:.3f}")
show("share of Var between arrangements: truth",lambda r:r['btw_T'],"{:.3f}")
show("share of Var between arrangements: I0",lambda r:r['btw_I0'],"{:.3f}")
show("share of Var between arrangements: P",lambda r:r['btw_P'],"{:.3f}")
show("share of Var between arrangements: penalty",lambda r:r['btw_pen'],"{:.3f}")
np.savez('pop1k/seed0_z12.npz',**r,names=np.array(names))
