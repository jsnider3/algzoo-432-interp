"""Mechanistic model of the lower inputs: given the top-3 values and positions, the 7 lower inputs are iid N(0,1)
truncated below R. How well can P(correct | top-3) be computed from the weights, and does integrating it over the
top-3 configurations match sampling at equal compute?

  closures  hidden mean/covariance propagated through the ReLUs, then the target - runner-up margin, against 4,096
            redraws per configuration (48 networks x 48 configurations):
              resid  soft gates, exact ReLU variance, independent residuals
              stein  soft gates, Stein (linear) part only
              biv    exact bivariate-Gaussian ReLU covariance
              jac    hard gates at the mean-field point (delta method)
  floor     on all networks: P(correct | top-3) from the first-order ANOVA (additive) model, i.e. each lower input's own
            effect on the two margins (target - max position, target - R position), computed by exact network passes at
            q equal-probability nodes with the other lower inputs at their mean, summed; P = P(both margins > 0) under
            the sum (and under a Gaussian with its moments). Compared with mean field and with the truth on the same
            configurations (lower inputs redrawn 8 times).
  score     estimate = mean of P_additive over an ordered-Sobol top-3 design (w1.transport), against the 2^20-input
            accuracy; cost per configuration 1 + 7 q 0.55 forward-equivalents (a node at position p reruns steps p..9)
Usage: python z8.py pop1k/seed0.npz {closures|floor|score}
"""
import sys, math, numpy as np, torch as th
from scipy.stats import norm, qmc
from batch_train import forward, target
from z2 import load
from z4 import input_vars
from w1 import transport
dev="cuda" if th.cuda.is_available() else "cpu"; SQ2PI=math.sqrt(2*math.pi)

def bvn(h, k, r):
    """P(X<h, Y<k), standard bivariate normal with corr r (20-point Gauss-Legendre on the integral over the correlation)"""
    xs,ws=np.polynomial.legendre.leggauss(20); out=th.special.ndtr(h)*th.special.ndtr(k)
    for xi,wi in zip(xs,ws):
        t=r*(xi+1)/2; out=out+r/2*wi*th.exp(-(h*h-2*t*h*k+k*k)/(2*(1-t*t)))/th.sqrt(1-t*t)/(2*math.pi)
    return out

def low_moments(x):
    """per configuration: order o, R, lower-input mask, and the truncated-normal mean / variance of each lower input"""
    o=np.argsort(-x,1); R=np.take_along_axis(x,o[:,2:3],1)[:,0]; low=np.ones(x.shape,bool); np.put_along_axis(low,o[:,:3],False,1)
    lam=norm.pdf(R)/norm.cdf(R); return o, R, low, -lam, 1-R*lam-lam**2

def propagate(Wih, Whh, xm, xv, kind):
    """hidden mean (N,B,H) and covariance (N,B,H,H) for inputs with means xm and variances xv (B,T)"""
    N,H=Wih.shape; B,T=xm.shape; WT=Whh.transpose(1,2)[:,None]; Wh=Whh[:,None]; I=th.eye(H,device=dev)
    m=th.zeros(N,B,H,device=dev); C=th.zeros(N,B,H,H,device=dev); ww=Wih[:,None,:,None]*Wih[:,None,None,:]
    for t in range(T):
        am=xm[None,:,t,None]*Wih[:,None,:]+th.einsum("nbj,nij->nbi",m,Whh); AC=Wh@C@WT+xv[None,:,t,None,None]*ww
        if kind=="jac":
            g=(am>0).float(); C=g[...,:,None]*AC*g[...,None,:]; m=am.clamp_min(0); continue
        v=th.diagonal(AC,0,-2,-1).clamp_min(0); s=v.sqrt(); det=s<1e-6; z=am/s.clamp_min(1e-6)
        Phi=th.where(det,(am>0).float(),th.special.ndtr(z)); phi=th.where(det,th.zeros_like(z),th.exp(-.5*z*z)/SQ2PI)
        Er=th.where(det,am.clamp_min(0),am*Phi+s*phi); Vr=th.where(det,th.zeros_like(v),((am*am+v)*Phi+am*s*phi-Er*Er).clamp_min(0))
        Cs=Phi[...,:,None]*AC*Phi[...,None,:]
        if kind=="biv":
            A=AC+1e-9*I; sd=th.diagonal(A,0,-2,-1).sqrt(); zz=am/sd; rho=(A/(sd[...,:,None]*sd[...,None,:])).clamp(-1+1e-6,1-1e-6)
            zi,zj,si,sj=zz[...,:,None],zz[...,None,:],sd[...,:,None],sd[...,None,:]; q=th.sqrt(1-rho*rho); pdf=lambda u: th.exp(-.5*u*u)/SQ2PI
            E2=si*sj*((zi*zj+rho)*bvn(zi,zj,rho)+zi*pdf(zj)*th.special.ndtr((zi-rho*zj)/q)+zj*pdf(zi)*th.special.ndtr((zj-rho*zi)/q)
                      +q/SQ2PI*pdf(th.sqrt((zi*zi-2*rho*zi*zj+zj*zj)/(q*q))))
            Eb=am*th.special.ndtr(zz)+sd*pdf(zz); Cb=E2-Eb[...,:,None]*Eb[...,None,:]
            C=th.where(det[...,:,None]|det[...,None,:],Cs,Cb); C=C-th.diag_embed(th.diagonal(C,0,-2,-1))+th.diag_embed(Vr)
        else:
            C=Cs+(th.diag_embed((Vr-Phi*Phi*v).clamp_min(0)) if kind=="resid" else 0)
        m=Er
    return m, C

def margins(Wih, Whh, Wo, xx, o):
    """inputs (B,T) and their order o (B,T) -> (N,B,2) margins target - max position, target - R position"""
    N=Wih.shape[0]; xx=xx[None].expand(N,-1,-1); h=th.zeros(N,xx.shape[1],Whh.shape[1],device=dev)
    for t in range(xx.shape[2]): h=th.relu(th.baddbmm(xx[:,:,t,None]*Wih[:,None,:],h,Whh.transpose(1,2)))
    L=h@Wo.transpose(1,2); g=lambda k: L.gather(-1,o[None,:,k:k+1].expand(N,-1,1)).squeeze(-1)
    return th.stack([g(1)-g(0),g(1)-g(2)],-1)

def additive(Wih, Whh, Wo, x, q, S=1024, gen=None):
    """top-3 configurations x (B,10) -> P(correct) under the additive model (N,B), under its Gaussian (N,B), mean field (N,B)"""
    B=len(x); o,R,low,mu,_=low_moments(x); xm=np.where(low,mu[:,None],x).astype(np.float32); ot=th.tensor(o,device=dev)
    f0=margins(Wih,Whh,Wo,th.tensor(xm,device=dev),ot); N=f0.shape[0]
    nodes=norm.ppf(((np.arange(q)+.5)/q)[None,:]*norm.cdf(R)[:,None]).astype(np.float32); lowpos=np.where(low)[1].reshape(B,7); tab=[]
    for j in range(7):
        xj=np.repeat(xm[:,None],q,1); xj[np.arange(B),:,lowpos[:,j]]=nodes
        tab.append(margins(Wih,Whh,Wo,th.tensor(xj.reshape(B*q,10),device=dev),ot.repeat_interleave(q,0)).reshape(N,B,q,2)-f0[:,:,None])
    tab=th.stack(tab,2); idx=th.randint(0,q,(S,7),device=dev,generator=gen); tot=f0[:,:,None,:].expand(N,B,S,2).clone()
    for j in range(7): tot+=tab[:,:,j][:,:,idx[:,j]]
    mean=f0+tab.mean(3).sum(2); d=tab-tab.mean(3,keepdim=True); cov=th.einsum("nbjqk,nbjql->nbkl",d,d)/q
    s=th.diagonal(cov,0,-2,-1).clamp_min(1e-12).sqrt(); z=mean/s; rho=(cov[...,0,1]/(s[...,0]*s[...,1])).clamp(-.999,.999)
    return (tot>0).all(-1).float().mean(-1), bvn(z[...,0],z[...,1],rho), (f0>0).all(-1).float()

def redraw(x, low, R, rng):
    xr=x.copy(); u=rng.random(x.shape); xr[low]=norm.ppf(u*norm.cdf(R)[:,None])[low]; return xr.astype(np.float32)

def correct(P, xx, ch=4096):
    N=P[0].shape[0]; out=np.zeros((N,len(xx)),np.float32); xt=th.tensor(xx,device=dev)
    for i in range(0,len(xx),ch):
        xc=xt[i:i+ch]; out[:,i:i+ch]=(forward(*P,xc[None].expand(N,-1,-1)).argmax(-1)==target(xc)[None]).cpu().numpy()
    return out

if __name__=="__main__":
    path,mode=sys.argv[1],sys.argv[2]; P,_=load(path); P=[p.to(dev) for p in P]; N=P[0].shape[0]; gen=th.Generator(device=dev).manual_seed(0)
    if mode=="closures":
        nets=np.arange(0,N,16)[:48]; Wih,Whh,Wo=[p[nets] for p in P]; n=len(nets); rng=np.random.default_rng(5)
        x=rng.standard_normal((48,10)).astype(np.float32); o,R,low,mu,var=low_moments(x)
        xm=th.tensor(np.where(low,mu[:,None],x).astype(np.float32),device=dev); xv=th.tensor(np.where(low,var[:,None],0).astype(np.float32),device=dev)
        xs=np.concatenate([redraw(np.repeat(x[b:b+1],4096,0),np.repeat(low[b:b+1],4096,0),np.repeat(R[b:b+1],4096),rng) for b in range(48)])
        h=th.zeros(n,len(xs),16,device=dev); xs=th.tensor(xs,device=dev)
        for t in range(10): h=th.relu(th.baddbmm(xs[None,:,t,None]*Wih[:,None,:],h,Whh.transpose(1,2)))
        h=h.reshape(n,48,4096,16); L=th.einsum("nbkh,noh->nbko",h,Wo); tg=th.tensor(o[:,1],device=dev)
        Lm=L.mean(2); Lm.scatter_(-1,tg[None,:,None].expand(n,48,1),-1e30); ru=Lm.argmax(-1); ar=th.arange(n)[:,None]
        w=Wo[ar,tg[None].expand(n,48)]-Wo[ar,ru]; d=(h*w[:,:,None]).sum(-1); mE,sE=d.mean(-1),d.std(-1)
        print(f"{n} networks x 48 configurations; true margin (target - runner-up): median mean {mE.median():.2f}, median SD {sE.median():.2f}")
        for kind in ("resid","stein","biv","jac"):
            m,C=propagate(Wih,Whh,xm,xv,kind); mM=(m*w).sum(-1); sM=th.einsum("nbh,nbhk,nbk->nb",w,C,w).clamp_min(0).sqrt()
            ok=th.isfinite(mM)&th.isfinite(sM)&(mM.abs()<1e4); r=(sM/sE)[ok]
            print(f"  {kind:5s}: margin mean |error| median {(mM-mE)[ok].abs().median():.2f};  SD ratio median {r.median():.2f} (IQR {th.quantile(r,.25):.2f}-{th.quantile(r,.75):.2f});"
                  f"  {(~ok).any(1).sum().item()} networks with diverging moments")
    elif mode=="floor":
        n=2048; m=8; rng=np.random.default_rng(21); x=rng.standard_normal((n,10)).astype(np.float32); o,R,low,_,_=low_moments(x)
        res=[np.concatenate([a.cpu().numpy() for a in r],1) for r in zip(*[additive(*P,x[i:i+32],8,gen=gen) for i in range(0,n,32)])]
        tr=sum(correct(P,redraw(x,low,R,rng)) for _ in range(m))/m; acc=tr.mean(1)
        print(f"{N} networks, {n} configurations x {m} redraws; truth {acc.mean():.4f}; noise of the truth ~{np.sqrt((tr*(1-tr)).mean(1)/(n*m)).mean()*100:.3f} points")
        for nm,e in zip(("additive model (q=8)","its Gaussian","mean field (max, R competitors)"),res):
            dd=(e.mean(1)-acc)*100; print(f"  {nm:32s} error (points): mean {dd.mean():+.3f}  SD {dd.std():.3f}  RMS {np.sqrt((dd**2).mean()):.3f};  per-configuration MAE {np.abs(e-tr).mean():.4f}")
        tot=acc*(1-acc); within=(tr*(1-tr)).mean(1)*m/(m-1)
        print(f"  variance within top-3 configurations {np.median(within/tot)*100:.0f}% (median); exact P(correct | top-3) would gain at most {np.median(tot/(tot-within)):.2f}x over sampling")
        V=input_vars(x)
        for k,nm in enumerate(["S<0","M-S tie","S-R tie","middle","clean"]):
            s=V["reg"]==k; dd=(res[0][:,s].mean(1)-tr[:,s].mean(1))*s.mean()*100
            print(f"    {nm:8s} {s.mean()*100:5.1f}%  error rate truth {1-tr[:,s].mean():.4f}, additive {1-res[0][:,s].mean():.4f};  contribution {dd.mean():+.3f} (SD {dd.std():.3f})")
    else:
        truth=np.load(path)["acc_2e20"]; p=truth.mean()
        print(f"{N} networks; truth = 2^20-input accuracy; p(1-p) = {p*(1-p):.4f}\n  q  configs  cost (fwd)  RMSE   mean   SD    | sampling at same cost   worth (samples)")
        for q in (2,4):
            for me in (6,8,10):
                est=[]
                for rep in range(2):
                    x=transport(qmc.Sobol(11,scramble=True,seed=100*q+10*me+rep).random_base2(me),10).astype(np.float32)
                    est.append(np.concatenate([additive(*P,x[i:i+32],q,gen=gen)[0].cpu().numpy() for i in range(0,len(x),32)],1).mean(1))
                e=(np.array(est)-truth[None])*100; rm=np.sqrt((e**2).mean()); C=2**me*(1+7*q*.55)
                print(f"  {q}  {2**me:6d}  {C:9.0f}   {rm:.3f}  {e.mean():+.3f}  {e.std():.3f}  | {np.sqrt(p*(1-p)/C)*100:.3f}   {p*(1-p)/(rm/100)**2:8.0f}")
