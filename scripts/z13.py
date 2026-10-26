"""Local fidelity of lower-input models: at each top-4 configuration c (positions and values of the 4 largest inputs), how
close is each model to the true conditional accuracy p(c) = P(correct | c)?  p(c) is measured by sampling the 6 lower
inputs (iid N(0,1) truncated below the 4th largest), nin draws per configuration (measurement only, never an estimator).
Models, all from the same 1 + 6q rows of z11.expansion (q equal-probability nodes per lower input, the rest at the mean):
  I0     the MF4 anchor indicator
  P1     clip(I0 + sum_j (E_j - I0)): z11's first-order cut-HDMR in the indicator
  Praw   the same, unclipped
  Pprod  I0 ? prod_j E_j : 1 - prod_j (1 - E_j)  (independent break / rescue events)
  Pmarg  first-order cut-HDMR in the MARGIN m = L_y - max_o L_o: m = m0 + sum_j d_j(z_j), d_j = m(row j) - m0, and
         P(m > 0) over the q^6 node combinations, exactly (pair sums + sorted search)
Reported: local MSE E[(model - p(c))^2] (minus the draw noise) as a share of p(1-p), integrated bias, how often P1 reverses
the anchor (I0 = 0 but P1 = 1, or I0 = 1 but P1 = 0: at an anchor boundary sum_j E_j - 5 I0 jumps by 5), and where the
local error sits relative to the MF4 margin m0.  Every 16th population network + the 6 zoo models.
Usage: python scripts/z13.py [configurations, default 8192] [nin, default 256] [q, default 16]
"""
import sys, numpy as np, torch as th
sys.path.insert(0,'.')
from batch_train import forward, target
from z2 import load
from z11 import expansion
from seeds_clocks import load as zload
dev='cuda'; nc=int(sys.argv[1]) if len(sys.argv)>1 else 8192; nin=int(sys.argv[2]) if len(sys.argv)>2 else 256; q=int(sys.argv[3]) if len(sys.argv)>3 else 16; k=4
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]
x=th.randn(nc,10,generator=th.Generator().manual_seed(31)).to(dev)
gi=th.Generator(device=dev).manual_seed(37)
def margin(L,y):
    Ly=L.gather(-1,y.expand(*L.shape[:-1],1)); Lo=L.scatter(-1,y.expand(*L.shape[:-1],1),-1e30).max(-1,keepdim=True).values
    return (Ly-Lo)[...,0]
keys=['p','I0','P1','Praw','Pprod','Pmarg']; acc={a:[] for a in keys}; extra={'ties':[],'m0':[]}
with th.no_grad():
    B=32
    for i in range(0,nc,B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4])
        X,_=expansion(xc,k,q); R=X.shape[1]
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,R,10)
        yy=y[None,:,None,None]
        I=(L.argmax(-1)==y[None,:,None]).float(); I0=I[:,:,0]; E=I[:,:,1:].view(N,b,6,q).mean(-1)
        Praw=I0+(E-I0[...,None]).sum(-1); P1=Praw.clamp(0,1)
        Pprod=th.where(I0>0,E.prod(-1),1-(1-E).prod(-1))
        M=margin(L,yy.expand(N,b,R,1)); m0=M[:,:,0]; d=(M[:,:,1:]-m0[...,None]).view(N,b,6,q)
        Pm=th.empty(N,b,device=dev)
        for s in range(0,b,4):                                                       # exact convolution of six 16-atom laws
            dd=d[:,s:s+4]; A=(dd[:,:,0,:,None]+dd[:,:,1,None,:]).flatten(-2); Bq=(dd[:,:,2,:,None]+dd[:,:,3,None,:]).flatten(-2)
            C=(dd[:,:,4,:,None]+dd[:,:,5,None,:]).flatten(-2).sort(-1).values
            AB=(A[...,:,None]+Bq[...,None,:]).flatten(-2); thr=-m0[:,s:s+4,None]-AB
            cnt=C.shape[-1]-th.searchsorted(C.contiguous(),thr.contiguous(),right=True)
            Pm[:,s:s+4]=cnt.float().sum(-1)/q**6
        # truth: resample the 6 lower inputs, top 4 kept
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; low=th.special.ndtri(u.clamp(min=1e-30))
        Xi=xc[:,None].repeat(1,nin,1); Xi.scatter_(-1,o[:,None,4:].expand(b,nin,6),low)
        Li=forward(*W,Xi.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,nin,10)
        Ii=(Li.argmax(-1)==y[None,:,None]).float(); p=Ii.mean(-1)
        extra['ties'].append(((Li==Li.max(-1,keepdim=True).values).sum(-1)>1).float().mean(-1))
        for a,v in zip(keys,(p,I0,P1,Praw,Pprod,Pm)): acc[a].append(v)
        extra['m0'].append(m0)
r={a:th.cat(v,1) for a,v in acc.items()}; m0=th.cat(extra['m0'],1); ties=th.cat(extra['ties'],1)
p=r['p']; noise=(p*(1-p)/(nin-1))                                                    # unbiased Var(p_hat) per configuration
pq=p.mean(1)*(1-p.mean(1)); pop=slice(0,N-6); zoo=['mystery']+[f'seed{i}' for i in range(5)]
def show(lab,v,fmt):
    v=v.cpu().numpy(); print(f"  {lab:52s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(a) for a in v[-6:]))
print(f"{N-6} population networks (every 16th) + 6 zoo; {nc} top-4 configurations, {nin} lower-input draws each; q = {q}")
show("accuracy (configurations x draws)",p.mean(1),"{:.4f}")
show("E[p(c)(1-p(c))] / p(1-p)   (irreducible lower-input share)",(p*(1-p)*nin/(nin-1)).mean(1)/pq,"{:.3f}")
print(" local error: E[(model - p(c))^2] - noise, as a share of p(1-p); and integrated bias (points)")
for a in keys[1:]:
    show(f"{a:6s} local MSE / p(1-p)",(((r[a]-p)**2-noise).mean(1))/pq,"{:.4f}")
for a in keys[1:]:
    show(f"{a:6s} bias, points",100*(r[a]-p).mean(1),"{:+.3f}")
show("clipping alone (P1 - Praw), points",100*(r['P1']-r['Praw']).mean(1),"{:+.3f}")
rev=((r['I0']==0)&(r['P1']==1))|((r['I0']==1)&(r['P1']==0))
show("share of configurations where P1 reverses I0",rev.float().mean(1),"{:.4f}")
show("  among them, mean p(c) when I0 = 0 (P1 = 1)",(p*((r['I0']==0)&(r['P1']==1))).sum(1)/((r['I0']==0)&(r['P1']==1)).sum(1).clamp(min=1),"{:.3f}")
show("  among them, mean p(c) when I0 = 1 (P1 = 0)",(p*((r['I0']==1)&(r['P1']==0))).sum(1)/((r['I0']==1)&(r['P1']==0)).sum(1).clamp(min=1),"{:.3f}")
show("  share of P1's local squared error from reversals",(((r['P1']-p)**2-noise)*rev).sum(1)/(((r['P1']-p)**2-noise)).sum(1),"{:.3f}")
sh=m0.abs()<0.5
for a in keys[1:]:
    show(f"|m0| < 0.5 ({a:5s}): local MSE share of its total",(((r[a]-p)**2-noise)*sh).sum(1)/(((r[a]-p)**2-noise)).sum(1).clamp(min=1e-12),"{:.3f}")
show("share of draws with a tied top logit",ties.mean(1),"{:.2e}")
np.savez(f'pop1k/seed0_z13_q{q}.npz',**{a:v.cpu().numpy() for a,v in r.items()},m0=m0.cpu().numpy())
