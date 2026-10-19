"""Is the lower-input correction local to the MF4 decision boundary?  Along an S-line (arrangement and M, R, Q fixed, S
varying over (R, M) with weight w(S) = phi(S) / (Phi(M) - Phi(R))), the anchor margin m0(S) crosses zero at points S*.
If the lower inputs shift the margin by Sigma (law fixed near S*), and m0 is linear there with slope s, then
  integral of [P(m0 + Sigma > 0) - 1(m0 > 0)] w dS  =  (w(S*) E[Sigma] - w'(S*) E[Sigma^2] / (2 s)) / |s|  + ...
a mean-shift term (the boundary moves by -E[Sigma] / s) and a smearing term (spread against the density gradient).
Tested on the first-order margin model (z13's Pmarg, q = 8 Gauss-Legendre nodes): its line penalty computed on a K-point
S grid (the reference) against (i) the two-term formula with moments at the crossings and (ii) the law of Sigma frozen
at each crossing and applied along the exact m0(S) of the grid.  Lines are sampled (a measurement of the approximation).
Every 64th population network + the 6 zoo models.  Usage: python z22.py [lines, default 2048] [K, default 64]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; nl=int(sys.argv[1]) if len(sys.argv)>1 else 2048; K=int(sys.argv[2]) if len(sys.argv)>2 else 64; q=8
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ug,wg=np.polynomial.legendre.leggauss(q); U=th.tensor((ug+1)/2,dtype=th.float32,device=dev); wq=th.tensor(wg/2,dtype=th.float32,device=dev)
w2=(wq[:,None]*wq[None,:]).flatten()
def margin(L,y):
    yy=y.expand(*L.shape[:-1],1)
    return (L.gather(-1,yy)-L.scatter(-1,yy,-1e30).max(-1,keepdim=True).values)[...,0]
def F(t,d):
    """P(t + sum_j d_j > 0) for atoms d (..., 6, q) with weights wq, at thresholds t (..., T)"""
    A=(d[...,0,:,None]+d[...,1,None,:]).flatten(-2); Bq=(d[...,2,:,None]+d[...,3,None,:]).flatten(-2)
    C,ix=(d[...,4,:,None]+d[...,5,None,:]).flatten(-2).sort(-1)
    tail=th.cat([w2[ix].flip(-1).cumsum(-1).flip(-1),th.zeros(*ix.shape[:-1],1,device=dev)],-1)
    AB=(A[...,:,None]+Bq[...,None,:]).flatten(-2); wAB=(w2[:,None]*w2[None,:]).flatten()
    thr=-t[...,:,None]-AB[...,None,:]                                                  # (..., T, q^4)
    pos=th.searchsorted(C[...,None,:].expand(*thr.shape[:-1],C.shape[-1]).contiguous(),thr.contiguous(),right=True)
    return (tail[...,None,:].expand(*thr.shape[:-1],tail.shape[-1]).gather(-1,pos)*wAB).sum(-1)
x=th.randn(nl,10,generator=th.Generator().manual_seed(73)).to(dev)
uS=(th.arange(K,device=dev)+.5)/K
out={k:th.zeros(N,device=dev,dtype=th.float64) for k in ('ref','ref_nocross','formula','mean_term','smear_term','frozen','ncross','I0','Pm')}
with th.no_grad():
    for i in range(nl):
        xi=x[i]; o=xi.argsort(descending=True); Mv,Rv,Qv=xi[o[0]],xi[o[2]],xi[o[3]]
        fR,fM=th.special.ndtr(Rv),th.special.ndtr(Mv); S=th.special.ndtri(fR+(fM-fR)*uS)            # (K,)
        mu=mu_below(Qv.view(1,1))[0,0]; lp=o[4:]
        base=mu.expand(K,10).clone(); base[:,o[0]]=Mv; base[:,o[2]]=Rv; base[:,o[3]]=Qv; base[:,o[1]]=S
        nodes=th.special.ndtri(th.special.ndtr(Qv)*U)                                   # (q,)
        rows=[base[:,None]]
        for j in range(6):
            r=base[:,None].repeat(1,q,1); r[:,:,lp[j]]=nodes[None]; rows.append(r)
        X=th.cat(rows,1)                                                                # (K, 1+6q, 10)
        L=forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).view(N,K,1+6*q,10)
        Mg=margin(L,o[1].view(1,1,1,1).expand(N,K,1+6*q,1)); m0=Mg[:,:,0]; d=(Mg[:,:,1:]-m0[...,None]).view(N,K,6,q)
        I0=(m0>0).float(); Pm=F(m0[...,None],d)[...,0]                                    # (N, K)
        # reference: smooth part by the midpoint rule, the indicator exactly from interpolated crossings
        sgn=m0>0; cross=(sgn[:,1:]!=sgn[:,:-1])                                         # (N, K-1)
        Sk,Sk1=S[:-1],S[1:]; t=m0[:,:-1]/(m0[:,:-1]-m0[:,1:]); Sx=Sk+(Sk1-Sk)*t          # crossing location
        ux=(th.special.ndtr(Sx)-fR)/(fM-fR)
        # exact integral of I0 over u in (0,1): piecewise between crossings, using sign at grid points
        edges=th.cat([th.zeros(N,1,device=dev),th.where(cross,ux,th.nan)],1)
        I0int=th.zeros(N,device=dev)
        for n_ in range(N):
            pts=th.cat([th.zeros(1,device=dev),ux[n_][cross[n_]],th.ones(1,device=dev)]); s0=sgn[n_,0].float()
            seg=pts[1:]-pts[:-1]; signs=(s0+th.arange(len(seg),device=dev))%2 if True else None
            I0int[n_]=(seg*((sgn[n_,0].float()+th.arange(len(seg),device=dev))%2)).sum()
        ref=Pm.mean(1)-I0int
        # formula: moments of Sigma at the crossing (interpolated), slope from the two grid points
        Es=(d*wq).sum(-1).sum(-1); Vs=((d*d*wq).sum(-1)-((d*wq).sum(-1))**2).sum(-1); E2=Vs+Es**2
        Esx=Es[:,:-1]+(Es[:,1:]-Es[:,:-1])*t; E2x=E2[:,:-1]+(E2[:,1:]-E2[:,:-1])*t
        sl=(m0[:,1:]-m0[:,:-1])/(Sk1-Sk); ww=th.exp(-Sx*Sx/2)/np.sqrt(2*np.pi)/(fM-fR); wp=-Sx*ww
        mean_t=th.where(cross,ww*Esx/sl.abs(),0.).sum(1); smear_t=th.where(cross,-wp*E2x/(2*sl*sl.abs()),0.).sum(1)
        # frozen law: Sigma's atoms at the grid point nearest each crossing, applied along m0(S) on the grid
        near=th.where(t<.5,th.arange(K-1,device=dev)[None],th.arange(1,K,device=dev)[None])   # (N, K-1)
        fz=th.zeros(N,device=dev)
        for n_ in range(N):
            ks=th.nonzero(cross[n_])[:,0]
            if len(ks)==0: continue
            # each grid point is assigned to its nearest crossing
            cpos=Sx[n_,ks]; asg=(S[:,None]-cpos[None]).abs().argmin(1)
            dd=d[n_,near[n_,ks]]                                                         # (C, 6, q)
            Fk=F(m0[n_][None,:].expand(len(ks),K),dd)                                    # (C, K)
            fz[n_]=(Fk.gather(0,asg[None])[0]).mean()-I0int[n_]
        for k,v in (('ref',ref),('ref_nocross',th.where(cross.any(1),0.,ref)),('formula',mean_t+smear_t),('mean_term',mean_t),('smear_term',smear_t),('frozen',fz),
                    ('ncross',cross.float().sum(1)),('I0',I0int),('Pm',Pm.mean(1))):
            out[k]+=v.double()
r={k:(v/nl).cpu().numpy() for k,v in out.items()}
def show(lab,v,fmt="{:+.3f}"):
    print(f"  {lab:46s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
print(f"{N-6} population networks (every 64th) + 6 zoo; {nl} sampled S-lines, K = {K} S points; first-order margin model, q = {q} GL")
show("crossings of the MF4 boundary per line",r['ncross'],"{:.3f}")
show("line penalty, reference (points)",100*r['ref'])
show("  of which on lines with no crossing",100*r['ref_nocross'])
show("  two-term boundary formula",100*r['formula'])
show("    mean-shift term",100*r['mean_term'])
show("    smearing term",100*r['smear_term'])
show("  law frozen at each crossing",100*r['frozen'])
show("  formula - reference",100*(r['formula']-r['ref']))
show("  frozen - reference",100*(r['frozen']-r['ref']))
np.savez('pop1k/seed0_z22.npz',**r)
