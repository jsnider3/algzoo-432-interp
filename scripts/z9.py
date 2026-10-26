"""Exact mean-field accuracy by linear-region walking: a deterministic, sampling-free estimate from the weights.

Mean field (REPORT 7.13): given the top-3 values M > S > R and their positions (pM, pS, pR), the 7 lower inputs are replaced
by their conditional mean mu(R) = -phi(R)/Phi(R). For one arrangement and one R the network input is then affine in (M, S),
so the ReLU RNN is piecewise affine on the slice R < S < M < MMAX. Walk its linear regions (best-first across region edges,
starting from the domain centroid); in each region the logits are affine, the incorrect set {some logit >= the pS logit} is
a polygon, and phi(M) phi(S) is integrated over it exactly in S (Phi) and by Gauss-Legendre in M. Then
    error(MF) = sum over 720 arrangements of  int dR phi(R) Phi(R)^7 I(arr, R),
with the R integral by composite Gauss-Legendre on PANELS (the integrand bends sharply just below R = 0, hence the dense
panels there). Checked against Monte Carlo of the mean-field network on single slices (check mode).
Cost in forward-equivalents, one forward = 10 steps of a 16x16 matvec = 5,120 flops (v1.py's unit is 2,880 multiply-adds,
about 12% larger), counting every affine pass, gate check, constraint evaluation, clip and polygon integral. Steps before
min(pM, pS) are shared within a slice; a neighbouring region reuses the parent's steps before the crossed neuron's step.

  run    full estimate; K = Gauss-Legendre nodes per R panel (a scalar or one per panel)
  check  6 random slices at R = -0.7 and 0.4: exact incorrect mass against 400,000 Monte Carlo draws of (M, S)
Usage: python scripts/z9.py weights.npz run [K] [workers]  |  python scripts/z9.py weights.npz check
"""
import sys, math, heapq, time, numpy as np
from itertools import permutations
from multiprocessing import Pool
from scipy.stats import norm
from scipy.special import ndtr
from seeds_clocks import load
MMAX=5.0; FWD=5120.0; GX,GW=np.polynomial.legendre.leggauss(8); ARRS=list(permutations(range(10),3))
PANELS=np.array([-1,-.5,-.3,-.15,-.06,0,.4,.8,1.2,1.6,2.1,2.8,4])     # P(R < -1) ~ 1e-5

def poly_mass(P):
    """integral of phi(M) phi(S) over the convex polygon P (k, 2) in (M, S); returns (mass, flops)"""
    if len(P)<3: return 0.0,0.0
    m0,m1=P[:,0].min(),P[:,0].max()
    if m1-m0<1e-14: return 0.0,0.0
    br=np.unique(np.concatenate([P[:,0],np.arange(math.floor(m0*2)/2,m1,0.5)])); br=br[(br>=m0)&(br<=m1)]
    a,b=br[:-1],br[1:]; keep=b-a>1e-14; a,b=a[keep],b[keep]
    if len(a)==0: return 0.0,0.0
    Mq=(a[:,None]+b[:,None])/2+(b[:,None]-a[:,None])/2*GX[None]; wq=(b[:,None]-a[:,None])/2*GW[None]
    E0=P; E1=np.roll(P,-1,0); dM=E1[:,0]-E0[:,0]; ok=np.abs(dM)>1e-14
    lo=np.minimum(E0[:,0],E1[:,0]); hi=np.maximum(E0[:,0],E1[:,0])
    t=(Mq[...,None]-E0[:,0])/np.where(ok,dM,1); S=E0[:,1]+t*(E1[:,1]-E0[:,1])
    inside=ok&(Mq[...,None]>=lo-1e-12)&(Mq[...,None]<=hi+1e-12)
    Su=np.where(inside,S,-np.inf).max(-1); Sl=np.where(inside,S,np.inf).min(-1)
    return float((wq*norm.pdf(Mq)*np.clip(ndtr(Su)-ndtr(Sl),0,None)).sum()), Mq.size*(len(P)*6+60)

def clip(P, lab, V, k, hid):
    """keep constraint k >= 0, with V (ncons, nverts) the constraint values at P's vertices; lab[i] labels edge P[i] -> P[i+1].
    Constraints are affine, so their values at new vertices are interpolated (exact), never re-evaluated"""
    v=V[k]; out=[];ol=[];cols=[]; n=len(P)
    for i in range(n):
        j=(i+1)%n; vi,vj=v[i],v[j]
        if vi>=0:
            out.append(P[i]); cols.append(('v',i))
            if vj>=0: ol.append(lab[i])
            else: t=vi/(vi-vj); out.append(P[i]+t*(P[j]-P[i])); cols.append(('x',i,j,t)); ol.append(lab[i]); ol.append(hid)
        elif vj>=0: t=vi/(vi-vj); out.append(P[i]+t*(P[j]-P[i])); cols.append(('x',i,j,t)); ol.append(lab[i])
    Vn=np.stack([V[:,c[1]] if c[0]=='v' else V[:,c[1]]+c[3]*(V[:,c[2]]-V[:,c[1]]) for c in cols],1) if cols else V[:,:0]
    return np.array(out),np.array(ol),Vn

class Slice:
    """one (arrangement, R) slice: run() returns (incorrect mass, regions, cost in forwards)"""
    def __init__(s, W, arr, R):
        s.Wih,s.Whh,s.Wo=W; s.pM,s.pS,s.pR=arr; s.R=R; mu=-norm.pdf(R)/norm.cdf(R)
        s.c=np.full(10,mu); s.c[[s.pM,s.pS]]=0; s.c[s.pR]=R; s.m=np.zeros(10); s.m[s.pM]=1; s.sv=np.zeros(10); s.sv[s.pS]=1
        s.D=np.array([[R,R],[MMAX,R],[MMAX,MMAX]]); s.Dlab=np.array([-1,-3,-2])          # domain R < S < M < MMAX
        s.t0=min(s.pM,s.pS); h=np.zeros(16)                                               # steps before t0 do not see (M, S)
        for t in range(s.t0): h=np.maximum(s.Wih*s.c[t]+s.Whh@h,0)
        s.h0=h; s.flops=s.t0*512.0
    def affine(s, M, S, parent=None, tc=0):
        """per-step pre-activations as affine maps A[t] (16, 3) of (1, M, S), gates G, logits L for the region at (M, S)"""
        z=np.array([1.0,M,S]); A=np.zeros((10,16,3)); G=np.zeros((10,16),bool)
        if parent is not None: A[:tc],G[:tc]=parent[0][:tc],parent[1][:tc]
        if parent is None or tc<=s.t0: tc=s.t0; H=np.zeros((16,3)); H[:,0]=s.h0
        else: H=A[tc-1]*G[tc-1][:,None]
        for t in range(tc,10):
            A[t]=np.outer(s.Wih,[s.c[t],s.m[t],s.sv[t]])+s.Whh@H; G[t]=A[t]@z>0; H=A[t]*G[t][:,None]
            s.flops+=512*(1+(t>=s.pM)+(t>=s.pS))
        s.flops+=320*3; return A,G,s.Wo@H
    def gates_key(s, M, S, parent, tc):
        z=np.array([1.0,M,S]); G=parent[1].copy()
        if tc<=s.t0: tc=s.t0; H=s.h0.copy()
        else: H=(parent[0][tc-1]*G[tc-1][:,None])@z
        for t in range(tc,10):
            a=s.Wih*(s.c[t]+s.m[t]*M+s.sv[t]*S)+s.Whh@H; G[t]=a>0; H=np.maximum(a,0); s.flops+=512
        return G.tobytes()
    def run(s):
        seen=set(); inc=0.0; nreg=0; cnt=0; p0=s.D.mean(0); reg=s.affine(*p0); seen.add(reg[1].tobytes()); heap=[(-1.0,0,reg)]
        while heap:
            _,_,reg=heapq.heappop(heap); A,G,L=reg; nreg+=1
            Cs=(A*np.where(G,1.0,-1.0)[...,None]).reshape(-1,3)[16*s.t0:]               # region = all gate signs held
            act=np.where(np.abs(Cs[:,1])+np.abs(Cs[:,2])>1e-12)[0]; C=Cs[act]
            P,lab=s.D.copy(),s.Dlab.copy(); V=C[:,:1]+C[:,1:2]*P[None,:,0]+C[:,2:3]*P[None,:,1]; s.flops+=C.shape[0]*12
            while len(P):
                mn=V.min(1)
                if mn.min()>=-1e-10: break
                k=int(mn.argmin()); P,lab,V=clip(P,lab,V,k,k); s.flops+=C.shape[0]*8
            if len(P)<3: continue
            d=L[s.pS][None]-L; dv=d[:,:1]+d[:,1:2]*P[:,0]+d[:,2:3]*P[:,1]; dv[s.pS]=1; s.flops+=40*len(P)
            if (dv<=0).any():                                                             # touches the incorrect set
                mreg,f1=poly_mass(P); Q,ql=P.copy(),lab.copy()
                for j in range(10):
                    if j!=s.pS and len(Q):
                        vq=d[j,0]+d[j,1]*Q[:,0]+d[j,2]*Q[:,1]
                        if (vq<0).any(): Q,ql,_=clip(Q,ql,vq[None],0,1000+j)
                mc,f2=poly_mass(Q) if len(Q)>=3 else (0.0,0.0); inc+=mreg-mc; s.flops+=f1+f2
            for i in range(len(P)):                                                       # neighbours across neuron edges
                if lab[i]<0: continue
                a_,b_=P[i],P[(i+1)%len(P)]
                if np.hypot(*(b_-a_))<1e-9: continue
                tc=s.t0+act[lab[i]]//16; n=-C[lab[i],1:]; n/=np.hypot(*n); mid=(a_+b_)/2+1e-7*n
                if not (mid[1]>s.R and mid[0]>mid[1] and mid[0]<MMAX): continue
                key=s.gates_key(mid[0],mid[1],reg,tc)
                if key in seen: continue
                seen.add(key); cnt+=1; heapq.heappush(heap,(-norm.pdf(mid[0])*norm.pdf(mid[1]),cnt,s.affine(mid[0],mid[1],reg,tc)))
        return inc, nreg, s.flops/FWD

def rnodes(K):
    """composite Gauss-Legendre in R on PANELS, K nodes per panel (scalar or per panel), weight phi(R) Phi(R)^7 folded in"""
    Ks=np.broadcast_to(np.atleast_1d(K),(len(PANELS)-1,)); R=[];Wt=[]
    for a,b,k in zip(PANELS[:-1],PANELS[1:],Ks):
        x,w=np.polynomial.legendre.leggauss(int(k)); r=(a+b)/2+(b-a)/2*x; R.append(r); Wt.append((b-a)/2*w*norm.pdf(r)*norm.cdf(r)**7)
    return np.concatenate(R),np.concatenate(Wt)

def work(args):
    name,ai,Rs=args; W=load(name); out=[]
    for R in Rs: inc,_,fw=Slice(W,ARRS[ai],R).run(); out.append((inc,fw))
    return out

if __name__=="__main__":
    name,mode=sys.argv[1],sys.argv[2] if len(sys.argv)>2 else "run"
    if mode=="run":
        K=[int(k) for k in (sys.argv[3] if len(sys.argv)>3 else "2").split(',')]; K=K[0] if len(K)==1 else np.array(K)
        Rs,Wt=rnodes(K); t0=time.time()
        with Pool(int(sys.argv[4]) if len(sys.argv)>4 else 6) as p: res=np.array(p.map(work,[(name,ai,Rs) for ai in range(720)],chunksize=8))
        e=res[:,:,0]*Wt[None]; fw=res[:,:,1].sum(); neg=Rs<0                                # res (720, nodes, 2)
        print(f'{name} K={sys.argv[3] if len(sys.argv)>3 else 2} ({len(Rs)} R nodes): mean-field accuracy {1-e.sum():.6f}; '
              f'cost {fw:,.0f} forwards ({res[:,neg,1].sum()/fw*100:.0f}% on R < 0, which holds {e[:,neg].sum()/e.sum()*100:.0f}% of the error); {time.time()-t0:.0f}s')
    else:
        W=load(name); rng=np.random.default_rng(1)
        for ai in rng.choice(720,6,replace=False):
            for R in (-0.7,0.4):
                inc,nreg,fw=Slice(W,ARRS[ai],R).run()
                n=400000; u=rng.random((n,2)); S=norm.ppf(ndtr(R)+u[:,0]*(ndtr(MMAX)-ndtr(R))); M=norm.ppf(ndtr(S)+u[:,1]*(ndtr(MMAX)-ndtr(S)))
                wgt=(ndtr(MMAX)-ndtr(R))*(ndtr(MMAX)-ndtr(S)); s=Slice(W,ARRS[ai],R); x=np.tile(s.c,(n,1)); x[:,s.pM]=M; x[:,s.pS]=S; h=np.zeros((n,16))
                for t in range(10): h=np.maximum(x[:,t:t+1]*W[0]+h@W[1].T,0)
                wrong=(h@W[2].T).argmax(1)!=s.pS
                print(f'arr {ARRS[ai]} R={R:+.1f}: {nreg:3d} regions, {fw:5.0f} forwards; incorrect mass {inc:.6f}, '
                      f'Monte Carlo {np.mean(wgt*wrong):.6f} +- {np.std(wgt*wrong)/np.sqrt(n):.6f}')
