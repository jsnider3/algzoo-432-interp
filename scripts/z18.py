"""A deterministic margin model without double counting: second-order cut-HDMR projected onto one-input terms.
z15-z17: the first-order margin model (each lower input moved alone from the anchor) is -0.18 points low because gate
flips and rival take-overs caused by several inputs are counted once per input; averaging each input's effect over the
others' spread (first-order ANOVA, z17, sampled) turns -0.18 into +0.06.  A deterministic version: from the second-order
cut model m2 = m0 + sum_j d_j(z_j) + sum_{j<l} D_jl(z_j, z_l) (all terms on the q-node grids), its exact ANOVA
first-order projection is
  mA(z) = m0 + sum_j h_j(z_j) - sum_{j<l} E[D_jl],    h_j(z) = d_j(z) + sum_{l != j} E_l[D_jl(z, Z_l)]
a sum of one-input functions, so P(mA > 0) is again an exact convolution over the q^6 node combinations.
Each input's effect is the effect it has on average given where the other inputs typically put the gates.
Variants keep only some pairs (both before the 2nd max's position; positions 1 or 2 apart), to cut the cost.
Scored like z13: against each configuration's true conditional accuracy p(c) (nin lower-input draws), local MSE and
integrated bias, with the first-order margin model (P1m) and z11's clamped indicator model (P1) on the same nodes.  Cost: 1 + 6q + 15q^2 rows per configuration.
Every 16th population network + the 6 zoo models (or `--holdout`: every 16th from 8, + zoo).
Nodes: equal-probability midpoints, or with --gl Gauss-Legendre in the CDF variable (weighted atoms), which reach the tails.
Usage: python scripts/z18.py [configurations, default 4096] [nin, default 128] [q, default 8] [--holdout] [--gl] [--all: every population network]
"""
import sys, itertools, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
args=[a for a in sys.argv[1:] if not a.startswith('--')]; hold='--holdout' in sys.argv; gl='--gl' in sys.argv; every='--all' in sys.argv
dev='cuda'; nc=int(args[0]) if len(args)>0 else 4096; nin=int(args[1]) if len(args)>1 else 128; q=int(args[2]) if len(args)>2 else 8
P,_=load('pop1k/seed0.npz'); P=[w if every else w[8 if hold else 0::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]
x=th.randn(nc,10,generator=th.Generator().manual_seed(53 if hold else 31)).to(dev); gi=th.Generator(device=dev).manual_seed(47)
PAIRS=list(itertools.combinations(range(6),2))
def margin(L,y):
    yy=y[None,:,None,None].expand(*L.shape[:-1],1)
    return (L.gather(-1,yy)-L.scatter(-1,yy,-1e30).max(-1,keepdim=True).values)[...,0]
if gl:                                                                                # Gauss-Legendre in the CDF variable: reaches both tails
    ug,wg=np.polynomial.legendre.leggauss(q); U=th.tensor((ug+1)/2,dtype=th.float32,device=dev); w=th.tensor(wg/2,dtype=th.float32,device=dev)
else:                                                                                 # equal-probability midpoints
    U=(th.arange(q,device=dev)+.5)/q; w=th.full((q,),1/q,device=dev)
w2=(w[:,None]*w[None,:]).flatten()
def conv_p(m0,h):
    """P(m0 + sum_j h_j > 0) with h (..., 6, q) atoms of weights w, exactly over the q^6 combinations (pair sums + sorted search)"""
    A=(h[...,0,:,None]+h[...,1,None,:]).flatten(-2); Bq=(h[...,2,:,None]+h[...,3,None,:]).flatten(-2)
    C,ix=(h[...,4,:,None]+h[...,5,None,:]).flatten(-2).sort(-1)
    tail=th.cat([w2[ix].flip(-1).cumsum(-1).flip(-1),th.zeros(*ix.shape[:-1],1,device=dev)],-1)   # weight of C atoms at or above each index
    thr=-m0[...,None]-(A[...,:,None]+Bq[...,None,:]).flatten(-2)
    pos=th.searchsorted(C.contiguous(),thr.contiguous(),right=True)
    return (tail.gather(-1,pos)*(w2[:,None]*w2[None,:]).flatten()).sum(-1)
keys=['p','I0','P1','P1m','PA','PA_beforeS','PA_gap1','PA_gap2','PA_beforeS_gap2']; acc={k:[] for k in keys}; npair={k:0.0 for k in keys}
with th.no_grad():
    B=4
    for i in range(0,nc,B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        base=mu_below(v4).expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        nodes=th.special.ndtri(th.special.ndtr(v4)*U[None])                                     # (b, q)
        rows=[base[:,None]]
        for j in range(6):
            r=base[:,None].repeat(1,q,1); r.scatter_(-1,lp[:,None,j:j+1].expand(b,q,1),nodes[...,None]); rows.append(r)
        for j,l in PAIRS:
            r=base[:,None].repeat(1,q*q,1)
            r.scatter_(-1,lp[:,None,j:j+1].expand(b,q*q,1),nodes[:,:,None].expand(b,q,q).reshape(b,q*q,1))
            r.scatter_(-1,lp[:,None,l:l+1].expand(b,q*q,1),nodes[:,None,:].expand(b,q,q).reshape(b,q*q,1)); rows.append(r)
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zz); rows.append(full)
        X=th.cat(rows,1); L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,-1,10); M=margin(L,y)
        m0=M[:,:,0]; ms=M[:,:,1:1+6*q].view(N,b,6,q); d=ms-m0[...,None,None]; c=1+6*q
        D=th.zeros(N,b,6,6,q,q,device=dev)                                             # D[j,l,a,c] = pair term at (node a of j, node c of l)
        for (j,l) in PAIRS:
            mp=M[:,:,c:c+q*q].view(N,b,q,q); c+=q*q
            Djl=mp-ms[:,:,j,:,None]-ms[:,:,l,None,:]+m0[...,None,None]; D[:,:,j,l]=Djl; D[:,:,l,j]=Djl.transpose(-1,-2)
        p=(L[:,:,c:c+nin].argmax(-1)==y[None,:,None]).float().mean(-1)
        I0=(L[:,:,0].argmax(-1)==y[None]).float()
        pj=lp[:,:,None]; pl=lp[:,None,:]; Sp=o[:,1,None,None]; gap=(pj-pl).abs()
        masks={'PA':th.ones(b,6,6,dtype=th.bool,device=dev),'PA_beforeS':(pj<Sp)&(pl<Sp),'PA_gap1':gap==1,'PA_gap2':gap<=2,
               'PA_beforeS_gap2':(pj<Sp)&(pl<Sp)&(gap<=2)}
        P1m=th.empty(N,b,device=dev)
        for s in range(b): P1m[:,s]=conv_p(m0[:,s],d[:,s])
        E=((ms>0).float()*w).sum(-1); P1=(I0+(E-I0[...,None]).sum(-1)).clamp(0,1)         # z11's clamped indicator model, same nodes
        out=[p,I0,P1,P1m]
        for k,mk in masks.items():
            Dm=D*mk[None,:,:,:,None,None]                                                # keep only the selected pairs
            h=d+(Dm*w).sum(-1).sum(3)                                                    # h_j(a) = d_j(a) + sum_l E_l D_jl(a, .)
            const=(Dm*w[:,None]*w[None,:]).sum((-1,-2)).sum((-1,-2))/2
            Pk=th.empty(N,b,device=dev)
            for s in range(b): Pk[:,s]=conv_p(m0[:,s]-const[:,s],h[:,s])
            out.append(Pk); npair[k]+=float((mk&~th.eye(6,dtype=th.bool,device=dev)).sum())/2
        for k,v in zip(keys,out): acc[k].append(v)
r={k:th.cat(v,1) for k,v in acc.items()}; p=r['p']; noise=p*(1-p)/(nin-1); pq=p.mean(1)*(1-p.mean(1)); pop=slice(0,N-6)
def show(lab,v,fmt):
    v=v.cpu().numpy(); print(f"  {lab:44s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  RMS {fmt.format(np.sqrt(np.mean(v[pop]**2)))}  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
print(f"{N-6} population networks ({'all' if every else 'every 16th'}{' from 8, held out' if hold and not every else ''}) + 6 zoo; {nc} configurations, {nin} draws each; q = {q} ({'Gauss-Legendre' if gl else 'midpoint'} nodes); {1+6*q+15*q*q} rows per configuration")
print(" pairs kept per configuration: "+", ".join(f"{k} {npair[k]/nc:.2f}" for k in keys[4:]))
for k in keys[1:]: show(f"{k:4s} local MSE / p(1-p)",(((r[k]-p)**2-noise).mean(1))/pq,"{:.4f}")
for k in keys[1:]: show(f"{k:4s} bias, points",100*(r[k]-p).mean(1),"{:+.3f}")
for k in keys[4:]: show(f"{k} - P1m, points (noise-free)",100*(r[k]-r['P1m']).mean(1),"{:+.3f}")
np.savez(f"pop1k/seed0_z18_q{q}{'_gl' if gl else ''}{'_holdout' if hold else ''}{'_all' if every else ''}.npz",**{k:v.cpu().numpy() for k,v in r.items()})
