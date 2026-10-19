"""Can the lower-input model reach about 0.01 points?  z18's best (margin + pairs at most 2 positions apart, projected onto
one-input terms) is +0.026 points on average, RMS 0.048, on all 1,024 networks; keeping only adjacent pairs gives -0.03.
Double counting among three nearby inputs (a gate flipped by three one-input runs) is what pairs cannot fix, so add the
third-order cut terms of every triple of lower inputs spanning at most 2 positions (three consecutive positions):
  T_jkl = m(a,b,c) - m(a,b) - m(a,c) - m(b,c) + m(a) + m(b) + m(c) - m0
and project the whole model onto one-input terms (ANOVA), which keeps P(m > 0) an exact convolution:
  mA = m0 + sum_j h_j(z_j) - sum_pairs E[D] - 2 sum_triples E[T],
  h_j(a) = d_j(a) + sum_{pairs (j,l)} E_l D_jl(a, .) + sum_{triples (j,k,l)} E_{k,l} T_jkl(a, ., .)
Scored against each configuration's conditional accuracy (nin draws) on 2^lc configurations, so that per-network noise
is about 0.007 points.  q = 6 Gauss-Legendre nodes in the CDF variable.  Every 16th population network + the zoo, or
--holdout (every 16th from 8).  Usage: python z32.py [log2 configurations, default 17] [nin, default 64] [span, default 2] [--holdout]
"""
import sys, itertools, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
args=[a for a in sys.argv[1:] if not a.startswith('--')]; hold='--holdout' in sys.argv
dev='cuda'; lc=int(args[0]) if len(args)>0 else 17; nin=int(args[1]) if len(args)>1 else 64; q=6; SPAN=int(args[2]) if len(args)>2 else 2
P,_=load('pop1k/seed0.npz'); P=[w[8 if hold else 0::16] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ug,wg=np.polynomial.legendre.leggauss(q); U=th.tensor((ug+1)/2,dtype=th.float32,device=dev); w=th.tensor(wg/2,dtype=th.float32,device=dev)
w2=(w[:,None]*w[None,:]).flatten(); w3=(w[:,None,None]*w[None,:,None]*w[None,None,:])
PAIRS=list(itertools.combinations(range(6),2)); TRI=list(itertools.combinations(range(6),3)); TS={2:4,3:10}.get(SPAN,20)      # triple slots per configuration
def margin(L,y):
    yy=y[None,:,None,None].expand(*L.shape[:-1],1)
    return (L.gather(-1,yy)-L.scatter(-1,yy,-1e30).max(-1,keepdim=True).values)[...,0]
def conv_p(m0,h):
    A=(h[...,0,:,None]+h[...,1,None,:]).flatten(-2); Bq=(h[...,2,:,None]+h[...,3,None,:]).flatten(-2)
    C,ix=(h[...,4,:,None]+h[...,5,None,:]).flatten(-2).sort(-1)
    tail=th.cat([w2[ix].flip(-1).cumsum(-1).flip(-1),th.zeros(*ix.shape[:-1],1,device=dev)],-1)
    thr=-m0[...,None]-(A[...,:,None]+Bq[...,None,:]).flatten(-2)
    pos=th.searchsorted(C.contiguous(),thr.contiguous(),right=True)
    return (tail.gather(-1,pos)*(w2[:,None]*w2[None,:]).flatten()).sum(-1)
x=th.randn(2**lc,10,generator=th.Generator().manual_seed(107 if hold else 109)).to(dev); gi=th.Generator(device=dev).manual_seed(113)
keys=['P1m','PA_gap1','PA_gap2','PT_gap1','PT_gap2']; S1={k:th.zeros(N,dtype=th.float64,device=dev) for k in keys}; S2={k:th.zeros(N,dtype=th.float64,device=dev) for k in keys}
ntri=0.
with th.no_grad():
    B=4
    for i in range(0,len(x),B):
        xc=x[i:i+B]; b=len(xc); y=target(xc); o=xc.argsort(-1,descending=True); v4=xc.gather(-1,o[:,3:4]); lp=o[:,4:]
        base=mu_below(v4).expand(b,10).clone(); base.scatter_(-1,o[:,:4],xc.gather(-1,o[:,:4]))
        nodes=th.special.ndtri(th.special.ndtr(v4)*U[None])
        rows=[base[:,None]]
        for j in range(6):
            r=base[:,None].repeat(1,q,1); r.scatter_(-1,lp[:,None,j:j+1].expand(b,q,1),nodes[...,None]); rows.append(r)
        na=nodes[:,:,None].expand(b,q,q).reshape(b,q*q,1); nb=nodes[:,None,:].expand(b,q,q).reshape(b,q*q,1)
        for j,l in PAIRS:
            r=base[:,None].repeat(1,q*q,1); r.scatter_(-1,lp[:,None,j:j+1].expand(b,q*q,1),na); r.scatter_(-1,lp[:,None,l:l+1].expand(b,q*q,1),nb); rows.append(r)
        # triple slots: triples whose positions span at most 2 (sorted positions consecutive), up to TS per configuration
        span=th.stack([lp[:,list(t)].max(1).values-lp[:,list(t)].min(1).values for t in TRI],1)          # (b, 20)
        okt=span<=SPAN; slot=th.full((b,TS),-1,dtype=th.long,device=dev)
        for s_ in range(b):
            idx=th.nonzero(okt[s_])[:,0][:TS]; slot[s_,:len(idx)]=idx
        ntri+=float((slot>=0).sum())
        n3=[nodes[:,:,None,None].expand(b,q,q,q).reshape(b,-1,1),nodes[:,None,:,None].expand(b,q,q,q).reshape(b,-1,1),nodes[:,None,None,:].expand(b,q,q,q).reshape(b,-1,1)]
        for t_ in range(TS):
            r=base[:,None].repeat(1,q**3,1); tri=th.tensor(TRI,device=dev)[slot[:,t_].clamp(min=0)]          # (b, 3)
            for c in range(3): r.scatter_(-1,lp.gather(1,tri[:,c:c+1])[:,None].expand(b,q**3,1),n3[c])
            rows.append(r)
        u=th.rand(b,nin,6,device=dev,generator=gi)*th.special.ndtr(v4)[:,None]; zz=th.special.ndtri(u.clamp(min=1e-30))
        full=base[:,None].repeat(1,nin,1); full.scatter_(-1,lp[:,None].expand(b,nin,6),zz); rows.append(full)
        X=th.cat(rows,1); L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,b,-1,10); M=margin(L,y)
        m0=M[:,:,0]; ms=M[:,:,1:1+6*q].view(N,b,6,q); d=ms-m0[...,None,None]; c=1+6*q
        mp=th.zeros(N,b,6,6,q,q,device=dev)                                            # pair margins m(a, b)
        for j,l in PAIRS:
            v=M[:,:,c:c+q*q].view(N,b,q,q); c+=q*q; mp[:,:,j,l]=v; mp[:,:,l,j]=v.transpose(-1,-2)
        D=mp-ms[:,:,:,None,:,None]-ms[:,:,None,:,None,:]+m0[...,None,None,None,None]
        eye=th.eye(6,dtype=th.bool,device=dev); D=D.masked_fill(eye[None,None,:,:,None,None],0.)
        T=th.zeros(N,b,TS,q,q,q,device=dev)
        for t_ in range(TS):
            v=M[:,:,c:c+q**3].view(N,b,q,q,q); c+=q**3
            tri=th.tensor(TRI,device=dev)[slot[:,t_].clamp(min=0)]; jj,kk,ll=tri[:,0],tri[:,1],tri[:,2]; bi=th.arange(b,device=dev)
            Djk=mp[:,bi,jj,kk][:,:,:,:,None]; Djl=mp[:,bi,jj,ll][:,:,:,None,:]; Dkl=mp[:,bi,kk,ll][:,:,None,:,:]
            sj=ms[:,bi,jj][:,:,:,None,None]; sk=ms[:,bi,kk][:,:,None,:,None]; sl=ms[:,bi,ll][:,:,None,None,:]
            T[:,:,t_]=(v-Djk-Djl-Dkl+sj+sk+sl-m0[...,None,None,None])*(slot[:,t_]>=0).float()[None,:,None,None,None]
        p=(L[:,:,c:c+nin].argmax(-1)==y[None,:,None]).float().mean(-1)
        gap=(lp[:,:,None]-lp[:,None,:]).abs()
        out={'P1m':th.stack([conv_p(m0[:,s_],d[:,s_]) for s_ in range(b)],1)}
        for gname,gmask in (('gap1',gap==1),('gap2',gap<=2)):
            Dm=D*gmask[None,:,:,:,None,None]; h=d+(Dm*w).sum(-1).sum(3); cst=(Dm*w[:,None]*w[None,:]).sum((-1,-2)).sum((-1,-2))/2
            out['PA_'+gname]=th.stack([conv_p(m0[:,s_]-cst[:,s_],h[:,s_]) for s_ in range(b)],1)
            # triples: E over the other two of T, added to each member's h; constant 2 E[T]
            h3=h.clone(); ET=(T*w3).sum((-1,-2,-3))                                           # (N, b, TS)
            for t_ in range(TS):
                tri=th.tensor(TRI,device=dev)[slot[:,t_].clamp(min=0)]; bi=th.arange(b,device=dev)
                for c_,axes in ((0,(-1,-2)),(1,(-1,-3)),(2,(-2,-3))):
                    marg=(T[:,:,t_]*w3).sum(axes)/w.view(*([1]*0),q)                           # E over the other two, at node a
                    h3[:,bi,tri[:,c_]]+=marg
            cst3=cst+2*ET.sum(-1)
            out['PT_'+gname]=th.stack([conv_p(m0[:,s_]-cst3[:,s_],h3[:,s_]) for s_ in range(b)],1)
        for k in keys:
            e=(out[k]-p).double(); S1[k]+=e.sum(1); S2[k]+=(e*e).sum(1)
n=len(x)
print(f"{N-6} population networks (every 16th{' from 8, held out' if hold else ''}) + 6 zoo; 2^{lc} configurations x {nin} draws; q = {q} Gauss-Legendre;"
      f" {ntri/n:.2f} triples spanning <= {SPAN} positions per configuration; points")
for k in keys:
    bb=(S1[k]/n).cpu().numpy(); se2=((S2[k]/n).cpu().numpy()-bb**2)/n; bp=100*bb[pop]
    print(f"  {k:8s} bias mean {bp.mean():+.4f}  median {np.median(bp):+.4f}  RMS {np.sqrt((bp**2).mean()):.4f}  noise {100*np.sqrt(se2[pop].mean()):.4f}"
          f"  noise-subtracted RMS {np.sqrt(max(0,(bp**2).mean()-1e4*se2[pop].mean())):.4f}  zoo "+" ".join(f"{100*v:+.3f}" for v in bb[-6:]))
np.savez(f"pop1k/seed0_z32{'_span'+str(SPAN) if SPAN!=2 else ''}{'_holdout' if hold else ''}.npz",**{k:(S1[k]/n).cpu().numpy() for k in keys},**{k+'_m2':(S2[k]/n).cpu().numpy() for k in keys},n=n)
