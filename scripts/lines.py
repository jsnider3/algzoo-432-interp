"""Exact integration of the network's correctness along one input coordinate (generalizes v1.trace_S).

For a base input x, a position p and an interval (lo, hi), the network is piecewise affine in s = x_p: from step p on,
each piece splits where a neuron's pre-activation changes sign; at the end the correct set (target logit above all
others) is an interval per piece. Returns P(correct | x_p ~ N(0,1) truncated to (lo, hi)), exactly, and the cost in
forward-equivalents (v1.py's multiply-add count, a forward = 2,880; every piece fully recomputed).
"""
import math, numpy as np, numba as nb
SQ2=math.sqrt(2.0)

@nb.njit(cache=True)
def _Phi(z): return 0.5*math.erfc(-z/SQ2)

@nb.njit(cache=True, parallel=True)
def trace_line(Wih, Whh, Wo, x, p, lo_, hi_, tgt, MAXP=512):
    n=x.shape[0]; H=Whh.shape[0]; O=Wo.shape[0]; T=x.shape[1]; P=np.zeros(n); cost=np.zeros(n); ovf=np.zeros(n,np.int64)
    for b in nb.prange(n):
        lo=np.empty(MAXP); hi=np.empty(MAXP); C=np.empty((MAXP,H)); D=np.empty((MAXP,H))
        lo2=np.empty(MAXP); hi2=np.empty(MAXP); C2=np.empty((MAXP,H)); D2=np.empty((MAXP,H))
        Ac=np.empty(H); Ad=np.empty(H); bp=np.empty(H+2); h=np.zeros(H); hn=np.empty(H); pb=p[b]; tg=tgt[b]
        for t in range(pb):
            for i in range(H):
                a=Wih[i]*x[b,t]
                for j in range(H): a+=Whh[i,j]*h[j]
                hn[i]=a if a>0 else 0.0
            for i in range(H): h[i]=hn[i]
        macs=pb*272.0; npcs=1; lo[0]=lo_[b]; hi[0]=hi_[b]
        for i in range(H): C[0,i]=h[i]; D[0,i]=0.0
        for t in range(pb,T):
            nn=0
            for k in range(npcs):
                if t==pb:
                    for i in range(H):
                        a=0.0
                        for j in range(H): a+=Whh[i,j]*C[k,j]
                        Ac[i]=a; Ad[i]=Wih[i]
                    macs+=256.0
                else:
                    for i in range(H):
                        a=Wih[i]*x[b,t]; d=0.0
                        for j in range(H): a+=Whh[i,j]*C[k,j]; d+=Whh[i,j]*D[k,j]
                        Ac[i]=a; Ad[i]=d
                    macs+=528.0
                nb_=0; bp[nb_]=lo[k]; nb_+=1
                for i in range(H):
                    if Ad[i]!=0.0:
                        s=-Ac[i]/Ad[i]
                        if s>lo[k] and s<hi[k]: bp[nb_]=s; nb_+=1
                bp[nb_]=hi[k]; nb_+=1; bs=np.sort(bp[:nb_])
                for q in range(nb_-1):
                    l=bs[q]; u=bs[q+1]
                    if u<=l: continue
                    if nn>=MAXP: ovf[b]=1; break
                    mid=0.5*(l+u)
                    for i in range(H):
                        g=1.0 if Ac[i]+Ad[i]*mid>0 else 0.0
                        C2[nn,i]=g*Ac[i]; D2[nn,i]=g*Ad[i]
                    lo2[nn]=l; hi2[nn]=u; nn+=1
            for k in range(nn):
                lo[k]=lo2[k]; hi[k]=hi2[k]
                for i in range(H): C[k,i]=C2[k,i]; D[k,i]=D2[k,i]
            npcs=nn
        tot=0.0
        for k in range(npcs):
            Lt=0.0; Ldt=0.0
            for j in range(H): Lt+=Wo[tg,j]*C[k,j]; Ldt+=Wo[tg,j]*D[k,j]
            L=lo[k]; U=hi[k]; ok=True
            for oo in range(O):
                if oo==tg: continue
                lc=0.0; ld=0.0
                for j in range(H): lc+=Wo[oo,j]*C[k,j]; ld+=Wo[oo,j]*D[k,j]
                dc=Lt-lc; dd=Ldt-ld
                if dd>0:
                    r=-dc/dd
                    if r>L: L=r
                elif dd<0:
                    r=-dc/dd
                    if r<U: U=r
                elif dc<0 or (dc==0 and oo<tg): ok=False         # exact tie on the piece: argmax takes the lowest index
            if ok and U>L: tot+=_Phi(U)-_Phi(L)
            macs+=320.0
        den=_Phi(hi_[b])-_Phi(lo_[b]); P[b]=tot/den if den>0 else 1.0; cost[b]=macs/2880.0
    return P, cost, ovf
