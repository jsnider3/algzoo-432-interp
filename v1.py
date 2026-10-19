"""Exact integration along the 2nd-max value: an unbiased sampler that integrates out the coordinate crossing both ties.

Given everything but the 2nd-largest input's value (positions, M, R, the lower inputs), S is N(0,1) truncated to (R, M),
and the target stays at S's position. Along that segment the network is piecewise affine in s: from step pS on, each
piece splits where a neuron's pre-activation changes sign, and at the end the correct set is an interval per piece.
So P(correct | rest) = sum of Phi-masses / (Phi(M) - Phi(R)) exactly (trace_S, numba). Averaging it over samples is
unbiased. Compared with plain sampling, ordered Sobol (w1.transport), and REPORT 7.15's exact last-input integration
(w1.cond_prob), each also paired with Sobol.

  ceil   variance-reduction factor p(1-p)/Var(conditional estimator) for the lines S in (R, M), M in (S, inf),
         R in (4th, S), and x10 over the full line (equal-probability grid of G nodes ~ exact), random inputs
  combo  f(x) = 1{pS >= k} P_S(x) + Q_k(x_1..9), with Q_k = exact x10 integral over {u: pS(x_1..9, u) < k}: the S line only
         where it is cheap (S late; the x10 prefix is shared), the last input elsewhere; each term unbiased. Needs a design with
         the full joint law (ordered Sobol): the S-free design pins S, which enters the x10 term (biased, 46-218 SE).
         k = 10 is the plain last-input baseline.
  score  matched-compute efficiency: p(1-p) / (n * cost * Var(estimate)) over scrambles (> 1 beats plain sampling);
         the S line is paired with random inputs, the ordered Sobol design (S's coordinate wasted), and an S-free
         10-dimensional ordered design (transport_S)
Cost in forward-equivalents from multiply-add counts: a step is 272 (16x16 + 16), the readout 160, a forward 2,880.
  full   every piece is recomputed: 528 per piece-step (affine intercept and slope), 320 per readout
  incr   sibling pieces (one gate apart) are updated from the previous piece: 32 per step, 20 per readout
For x10 the line has <= 19 pieces: full 3.05, incremental 1.18 (REPORT 7.15 measured 2.2-2.5 by timing).
Usage: python v1.py ceil | score [zoo|pop] | combo   (pop = 64 networks of pop1k/seed0.npz)
"""
import sys, math, numpy as np, numba as nb, torch as th
from scipy.stats import norm, qmc
from batch_train import forward, target
from itertools import permutations
from scipy.special import ndtri, ndtr
from w1 import transport, cond_prob, fwd
dev="cuda" if th.cuda.is_available() else "cpu"; SQ2=math.sqrt(2.0); FWD=2880.0
X10_COST={"full":(2448+256+16+19*320)/FWD,"incr":(2448+256+16+320+18*20)/FWD}

def models(which):
    if which=="zoo":
        from seeds_clocks import load
        names=["mystery"]+[f"seed{i}" for i in range(5)]; Ws=[load("weights.npz")]+[load(f"weights_seed{i}.npz") for i in range(5)]
        return names, Ws, None
    from z2 import load
    P,_=load("pop1k/seed0.npz"); idx=np.arange(0,1024,16); truth=np.load("pop1k/seed0.npz")["acc_2e20"][idx]
    return [f"pop{i}" for i in idx], [tuple(p[i].numpy().astype(np.float64) for p in P) for i in idx], truth

def transport_S(u, T=10):
    """S-free ordered design, (n,10) uniforms -> inputs: u0 -> M (max of T), u1 -> R given M (2nd largest of the T-1 below M,
    from F^9 + 9 F^8 (1-F) = u1 with F = Phi(R)/Phi(M)), u2 -> arrangement, u3.. -> lower inputs below R. S is set to the
    mid-quantile of (R, M) only to fix the order; trace_S integrates it out. Exact in law for uniform u."""
    n=len(u); M=-ndtri(-np.expm1(np.log(u[:,0])/T)); lo=np.zeros(n); hi=np.ones(n)
    for _ in range(60):
        F=(lo+hi)/2; g=F**9+9*F**8*(1-F); lo=np.where(g<u[:,1],F,lo); hi=np.where(g<u[:,1],hi,F)
    PM=ndtr(M); R=ndtri((lo+hi)/2*PM); S=ndtri(.5*(ndtr(R)+PM))
    arr=np.array(list(permutations(range(T),3))); a=arr[np.minimum((u[:,2]*len(arr)).astype(int),len(arr)-1)]
    x=np.empty((n,T)); rows=np.arange(n)[:,None]; mask=np.ones((n,T),bool); mask[rows,a]=False
    x[rows,a]=np.c_[M,S,R]; x[mask]=ndtri(ndtr(R)[:,None]*u[:,3:]).ravel(); return x

@nb.njit(cache=True)
def Phi(z): return 0.5*math.erfc(-z/SQ2)

@nb.njit(cache=True)
def trace_S(Wih, Whh, Wo, x, MAXP=512):
    """x (n,10) -> P(correct | all but S's value) (n,), multiply-adds full / incremental (n,), pieces (n,), overflow (n,)"""
    n=x.shape[0]; H=Whh.shape[0]; O=Wo.shape[0]; T=x.shape[1]
    P=np.zeros(n); mf=np.zeros(n); mi=np.zeros(n); npc=np.zeros(n,np.int64); ovf=np.zeros(n,np.int64)
    lo=np.empty(MAXP); hi=np.empty(MAXP); C=np.empty((MAXP,H)); D=np.empty((MAXP,H)); sib=np.zeros(MAXP,np.int64)
    lo2=np.empty(MAXP); hi2=np.empty(MAXP); C2=np.empty((MAXP,H)); D2=np.empty((MAXP,H)); sib2=np.zeros(MAXP,np.int64)
    Ac=np.empty(H); Ad=np.empty(H); bp=np.empty(H+2); h=np.empty(H); hn=np.empty(H)
    for b in range(n):
        o=np.argsort(-x[b]); pS=o[1]; M=x[b,o[0]]; R=x[b,o[2]]
        for i in range(H): h[i]=0.0
        for t in range(pS):
            for i in range(H):
                a=Wih[i]*x[b,t]
                for j in range(H): a+=Whh[i,j]*h[j]
                hn[i]=a if a>0 else 0.0
            for i in range(H): h[i]=hn[i]
        macs=pS*272.0; mincr=macs; npcs=1; lo[0]=R; hi[0]=M; sib[0]=0
        for i in range(H): C[0,i]=h[i]; D[0,i]=0.0
        for t in range(pS,T):
            nn=0
            for k in range(npcs):
                if t==pS:
                    for i in range(H):
                        a=0.0
                        for j in range(H): a+=Whh[i,j]*C[k,j]
                        Ac[i]=a; Ad[i]=Wih[i]
                    macs+=256.0; mincr+=256.0
                else:
                    for i in range(H):
                        a=Wih[i]*x[b,t]; d=0.0
                        for j in range(H): a+=Whh[i,j]*C[k,j]; d+=Whh[i,j]*D[k,j]
                        Ac[i]=a; Ad[i]=d
                    macs+=528.0; mincr+=(32.0 if sib[k]==1 else 528.0)
                nb_=0; bp[nb_]=lo[k]; nb_+=1
                for i in range(H):
                    if Ad[i]!=0.0:
                        s=-Ac[i]/Ad[i]
                        if s>lo[k] and s<hi[k]: bp[nb_]=s; nb_+=1
                bp[nb_]=hi[k]; nb_+=1; bs=np.sort(bp[:nb_]); macs+=16.0; mincr+=16.0
                for q in range(nb_-1):
                    l=bs[q]; u=bs[q+1]
                    if u<=l: continue
                    if nn>=MAXP: ovf[b]=1; break
                    mid=0.5*(l+u)
                    for i in range(H):
                        g=1.0 if Ac[i]+Ad[i]*mid>0 else 0.0
                        C2[nn,i]=g*Ac[i]; D2[nn,i]=g*Ad[i]
                    lo2[nn]=l; hi2[nn]=u; sib2[nn]=1 if q>0 else 0; nn+=1
            for k in range(nn):
                lo[k]=lo2[k]; hi[k]=hi2[k]; sib[k]=sib2[k]
                for i in range(H): C[k,i]=C2[k,i]; D[k,i]=D2[k,i]
            npcs=nn
        tot=0.0
        for k in range(npcs):
            Lt=0.0; Ldt=0.0
            for j in range(H): Lt+=Wo[pS,j]*C[k,j]; Ldt+=Wo[pS,j]*D[k,j]
            L=lo[k]; U=hi[k]; ok=True
            for oo in range(O):
                if oo==pS: continue
                lc=0.0; ld=0.0
                for j in range(H): lc+=Wo[oo,j]*C[k,j]; ld+=Wo[oo,j]*D[k,j]
                dc=Lt-lc; dd=Ldt-ld
                if dd>0:
                    r=-dc/dd
                    if r>L: L=r
                elif dd<0:
                    r=-dc/dd
                    if r<U: U=r
                elif dc<=0: ok=False
            if ok and U>L: tot+=Phi(U)-Phi(L)
            macs+=320.0; mincr+=(20.0 if sib[k]==1 else 320.0)
        den=Phi(M)-Phi(R); P[b]=tot/den if den>0 else 1.0; mf[b]=macs; mi[b]=mincr; npc[b]=npcs
    return P, mf, mi, npc, ovf

@th.no_grad()
def grid_line(Ws, x, kind, G, ch=64):
    """P(correct | all but one coordinate) on an equal-probability grid of G nodes, (N, n)"""
    P=[th.tensor(np.stack([w[k] for w in Ws]),dtype=th.float32,device=dev) for k in range(3)]; N=len(Ws); out=np.zeros((N,len(x)))
    for i in range(0,len(x),ch):
        xc=x[i:i+ch]; B=len(xc); o=np.argsort(-xc,1); v=np.take_along_axis(xc,o,1); u=(np.arange(G)+.5)/G
        lo,hi,pos={"x10":(np.zeros(B),np.ones(B),np.full(B,9)),"S":(norm.cdf(v[:,2]),norm.cdf(v[:,0]),o[:,1]),
                   "M":(norm.cdf(v[:,1]),np.ones(B),o[:,0]),"R":(norm.cdf(v[:,3]),norm.cdf(v[:,1]),o[:,2])}[kind]
        xx=np.repeat(xc[:,None],G,1); xx[np.arange(B),:,pos]=np.clip(norm.ppf(lo[:,None]+u[None]*(hi-lo)[:,None]),-12,12)
        xt=th.tensor(xx.reshape(-1,10),dtype=th.float32,device=dev)
        out[:,i:i+ch]=(forward(*P,xt[None].expand(N,-1,-1)).argmax(-1)==target(xt)[None]).float().reshape(N,B,G).mean(-1).cpu().numpy()
    return out

def q_restricted(y, W, k):
    """w1.cond_prob restricted to x10 in {pS < k}; returns P (n,), pieces (n,)"""
    Wih,Whh,Wo=W; n=len(y); a=Wih.reshape(-1); b=fwd(y,Wih,Whh)@Whh.T
    with np.errstate(divide='ignore'): hinge=np.where(a!=0,-b/a,np.inf)
    o=np.argsort(y,1); i1,i2=o[:,-1],o[:,-2]; m1=y[np.arange(n),i1]; m2=y[np.arange(n),i2]
    bp=np.sort(np.concatenate([hinge,m1[:,None],m2[:,None]],1),1)
    lo=np.concatenate([np.full((n,1),-np.inf),bp],1); hi=np.concatenate([bp,np.full((n,1),np.inf)],1)
    mid=np.where(np.isinf(lo),hi-1,np.where(np.isinf(hi),lo+1,(lo+hi)/2))
    tgt=np.where(mid<m2[:,None],i2[:,None],np.where(mid<m1[:,None],9,i1[:,None]))
    keep=np.where(mid<m2[:,None],(i2<k)[:,None],np.where(mid<m1[:,None],9<k,(i1<k)[:,None]))   # pS of the piece < k
    act=(mid[:,:,None]*a+b[:,None,:])>0; s=(act*a)@Wo.T; c=(act*b[:,None,:])@Wo.T
    st=np.take_along_axis(s,tgt[...,None],2); ct=np.take_along_axis(c,tgt[...,None],2); ds=st-s; dc=ct-c
    with np.errstate(divide='ignore',invalid='ignore'): r=-dc/ds
    L=np.where(ds>0,r,-np.inf).max(2); U=np.where(ds<0,r,np.inf).min(2)
    bad=((ds==0)&(dc<=0)); bad[np.arange(n)[:,None],np.arange(lo.shape[1]),tgt]=False
    L=np.maximum(L,lo); U=np.minimum(U,hi)
    p=np.where((U>L)&~bad.any(2)&keep,ndtr(U)-ndtr(L),0.0)
    return p.sum(1), keep.sum(1)

def combo(W, x, k):
    """returns estimate per sample, cost full / incremental per sample (forward-equivalents)"""
    q,npc=q_restricted(x[:,:9],W,k); pS=np.argsort(-x,1)[:,1]; use=pS>=k
    est=q.copy(); cf=(2448+256+16+npc*320)/FWD; ci=(2448+256+16+320+np.maximum(npc-1,0)*20)/FWD
    if use.any():
        P,mf,mi,_,_=trace_S(*W,np.ascontiguousarray(x[use])); est[use]+=P
        cf[use]+=(mf-pS[use]*272)/FWD; ci[use]+=(mi-pS[use]*272)/FWD      # prefix shared with the x10 pass
    return est, cf, ci

def plain(W, x):
    Wih,Whh,Wo=W; h=np.zeros((len(x),len(Wih)))
    for t in range(x.shape[1]): h=np.maximum(x[:,t:t+1]*Wih+h@Whh.T,0)
    return ((h@Wo.T).argmax(1)==np.argsort(x,1)[:,-2]).astype(float)

if __name__=="__main__":
    mode=sys.argv[1]; which=sys.argv[2] if len(sys.argv)>2 else "zoo"; names,Ws,truth=models(which)
    if mode=="combo":
        xb=np.random.default_rng(0).standard_normal((2**22,10)); p=np.array([plain(W,xb).mean() for W in Ws]); V=p*(1-p)
        print("zoo models; efficiency p(1-p)/(n cost Var) over 32 ordered-Sobol scrambles, full / incremental cost")
        for m in (12,14):
            n=2**m; R=32
            for k in (5,7,10):
                E=np.zeros((R,len(Ws))); CF=np.zeros(len(Ws)); CI=np.zeros(len(Ws))
                for r in range(R):
                    xs=transport(qmc.Sobol(11,scramble=True,seed=10*m+r).random_base2(m),10)
                    for i,W in enumerate(Ws):
                        e,cf,ci=combo(W,xs,k); E[r,i]=e.mean(); CF[i]+=cf.mean()/R; CI[i]+=ci.mean()/R
                var=E.var(0,ddof=1); ef=V/(n*CF*var); ei=V/(n*CI*var); bias=(E.mean(0)-p)/np.sqrt(var/R+V/2**22)
                print(f"  n=2^{m} k={k:2d}: cost {np.median(CF):.2f}/{np.median(CI):.2f} fwd; "+"  ".join(f"{names[i]} {ef[i]:.1f}/{ei[i]:.1f}" for i in range(len(Ws)))+f"   max |bias|/SE {np.abs(bias).max():.1f}", flush=True)
    elif mode=="ceil":
        n=2**15 if which=="zoo" else 2**13; x=np.random.default_rng(0).standard_normal((n,10))
        p=np.array([plain(W,x).mean() for W in Ws]) if truth is None else truth; V=p*(1-p)
        vrf={k:V/grid_line(Ws,x,k,256).var(1) for k in ("x10","S","M","R")}
        print(f"{len(Ws)} networks, {n} random inputs each; variance-reduction factor p(1-p)/Var(P_line), grid of 256 nodes\n  network   x10     S      M      R")
        rows=range(len(Ws)) if which=="zoo" else []
        for i in rows: print(f"  {names[i]:8s} "+" ".join(f"{vrf[k][i]:5.2f} " for k in vrf))
        print("  median   "+" ".join(f"{np.median(vrf[k]):5.2f} " for k in vrf))
        if which!="zoo": print("  p10-p90  "+" ".join(f"{np.percentile(vrf[k],10):.1f}-{np.percentile(vrf[k],90):.1f}" for k in vrf))
    else:
        R=32 if which=="zoo" else 16; ms=(10,12,14) if which=="zoo" else (12,)
        NT=2**20
        if truth is None:
            NT=2**22; xb=np.random.default_rng(0).standard_normal((NT,10)); truth=np.array([plain(W,xb).mean() for W in Ws])
        V=truth*(1-truth)
        for m in ms:
            n=2**m; E={k:np.zeros((R,len(Ws))) for k in ("plain","Sobol","Sobol+x10","S","Sobol+S","S-free Sobol+S")}; cf=np.zeros(len(Ws)); ci=np.zeros(len(Ws)); pc=np.zeros(len(Ws))
            for r in range(R):
                xr=np.random.default_rng(10*m+r).standard_normal((n,10)); xs=transport(qmc.Sobol(11,scramble=True,seed=10*m+r).random_base2(m),10)
                xf=transport_S(qmc.Sobol(10,scramble=True,seed=500+10*m+r).random_base2(m))
                for i,W in enumerate(Ws):
                    E["plain"][r,i]=plain(W,xr).mean(); E["Sobol"][r,i]=plain(W,xs).mean(); E["Sobol+x10"][r,i]=cond_prob(xs[:,:9],W).mean()
                    E["S"][r,i]=trace_S(*W,xr)[0].mean(); p_,mf,mi,npcs,_=trace_S(*W,xs); E["Sobol+S"][r,i]=p_.mean(); E["S-free Sobol+S"][r,i]=trace_S(*W,xf)[0].mean()
                    cf[i]+=mf.mean()/FWD/R; ci[i]+=mi.mean()/FWD/R; pc[i]+=npcs.mean()/R
            cost={"plain":(1,1),"Sobol":(1,1),"Sobol+x10":(X10_COST["full"],X10_COST["incr"]),"S":(cf,ci),"Sobol+S":(cf,ci),"S-free Sobol+S":(cf,ci)}
            print(f"\nn=2^{m}, {R} scrambles; efficiency p(1-p)/(n cost Var) under full / incremental cost (>1 beats plain sampling); "
                  f"S-line cost {np.median(cf):.2f} / {np.median(ci):.2f} fwd, {np.median(pc):.1f} pieces (median)", flush=True)
            for k,v in E.items():
                var=v.var(0,ddof=1); eff=[V/(n*np.asarray(c)*var) for c in cost[k]]; bias=(v.mean(0)-truth)/np.sqrt(var/R+V/NT)
                if which=="zoo": print(f"  {k:10s} "+"  ".join(f"{names[i]} {eff[0][i]:4.1f}/{eff[1][i]:4.1f}" for i in range(len(Ws)))+f"   max |bias|/SE {np.abs(bias).max():.1f}", flush=True)
                else: print(f"  {k:10s} median {np.median(eff[0]):4.2f}/{np.median(eff[1]):4.2f} (p10-p90 {np.percentile(eff[0],10):.2f}-{np.percentile(eff[0],90):.2f} full)   max |bias|/SE {np.abs(bias).max():.1f}", flush=True)
