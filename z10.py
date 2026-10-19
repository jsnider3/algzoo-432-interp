"""Anatomy of the mean-field leakage: which lower inputs, at which positions, carry accuracy(mean field) - accuracy(true)?

Mean field (MF3) keeps the top-3 values and positions and sets the 7 lower inputs to mu(R) = -phi(R)/Phi(R). Each variant
below keeps some lower inputs at their true values (or uses another stand-in) and is scored paired with the true inputs on
one shared pool, so leakage(variant) = accuracy(variant) - accuracy(true) has only the disagreement noise.
  MFk         top-k kept, the rest at their mean given they are below the k-th: mu(x_(k)), k = 3..7
  keep t      MF3, plus the lower input at absolute position t kept (one per position): which positions leak
  zone        MF3, plus the lower inputs before the first top-3 event / between events / after the last event kept
  stand-in    the 7 lower inputs at the truncated-normal median, or at 0, instead of the mean
  ranks       the lower inputs keep their order (positions of the 7 order statistics) but take fixed values: the i-th
              smallest at Phi^-1(Phi(R) (i - 3/8) / 7.25), so only their arrangement varies, not their values
  events      MF3, plus the lower inputs that were a running top-k event when they arrived (above the k-th largest of the
              prefix, k = 1, 2, 3); a top-2-event lower input is one the network had to treat as a candidate 2nd max
  near R      MF3, plus the lower inputs within d of R (d = 0.5, 1); and plus those below R - d (the far tail)
  flips       MF3 against the truth per input: P(MF right, true wrong) - P(MF wrong, true right), by |MF margin| bin, and
              by what the wrong answer was (the max's position, R's position, or a lower input's position)
Usage: python z10.py pop1k/seed0.npz [log2 pool size]
"""
import sys, numpy as np, torch as th
from scipy.stats import norm
from batch_train import forward, target
from z2 import load
dev="cuda" if th.cuda.is_available() else "cpu"

def mu_below(v): return -th.exp(-v*v/2)/np.sqrt(2*np.pi)/th.special.ndtr(v)

def topk_mf(x, k, keep=None):
    """keep the top-k (and positions in the boolean mask keep), the rest at E[x | x < x_(k)]"""
    o=x.argsort(-1,descending=True); xk=x.gather(-1,o[:,k-1:k]); out=mu_below(xk).expand_as(x).clone()
    m=th.zeros_like(x,dtype=th.bool).scatter(-1,o[:,:k],True)
    if keep is not None: m|=keep
    out[m]=x[m]; return out

@th.no_grad()
def run(W, x, variants, chunk=4096):
    """per network: accuracy of the true input and of each variant; also MF3 flips by margin bin"""
    N=W[0].shape[0]; bins=th.tensor([0,.25,.5,1,2,4,1e9],device=dev); acc=th.zeros(N,len(variants)+1,device=dev)
    fl=th.zeros(N,2,len(bins)-1,device=dev); cat=th.zeros(N,2,3,device=dev)
    for i in range(0,len(x),chunk):
        xc=x[i:i+chunk]; y=target(xc); xs=xc[None].expand(N,-1,-1)
        at=forward(*W,xs).argmax(-1); rt=at==y; acc[:,0]+=rt.sum(1); o=xc.argsort(-1,descending=True)
        def kind(a): return th.where(a==o[None,:,0],0,th.where(a==o[None,:,2],1,2))      # max / R / lower position
        for j,f in enumerate(variants.values()):
            L=forward(*W,f(xc)[None].expand(N,-1,-1)); rv=L.argmax(-1)==y; acc[:,j+1]+=rv.sum(1)
            if j==0:                                                                    # MF3: flips by MF margin
                Ly=L.gather(-1,y[None,:,None].expand(N,-1,1))[...,0]; Lo=L.scatter(-1,y[None,:,None].expand(N,-1,1),-1e9).max(-1).values
                b=th.bucketize((Ly-Lo).abs(),bins)-1
                for k in range(len(bins)-1):
                    fl[:,0,k]+=((b==k)&rv&~rt).sum(1); fl[:,1,k]+=((b==k)&~rv&rt).sum(1)
                kt,kv=kind(at),kind(L.argmax(-1))
                for c in range(3): cat[:,0,c]+=((kt==c)&rv&~rt).sum(1); cat[:,1,c]+=((kv==c)&~rv&rt).sum(1)
    return (acc/len(x)).cpu().numpy(), (fl/len(x)).cpu().numpy(), (cat/len(x)).cpu().numpy()

if __name__=="__main__":
    P,s=load(sys.argv[1]); m=int(sys.argv[2]) if len(sys.argv)>2 else 20
    from seeds_clocks import load as zload
    Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
    Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
    W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; names=[f"pop{i}" for i in range(N-6)]+["mystery"]+[f"seed{i}" for i in range(5)]
    x=th.randn(2**m,10,generator=th.Generator().manual_seed(5)).to(dev)
    def pos_mask(xc, which):
        o=xc.argsort(-1,descending=True)[:,:3]; t=th.arange(10,device=dev)[None]; lo,hi=o.min(1,keepdim=True).values,o.max(1,keepdim=True).values
        return {"before":t<lo,"between":(t>lo)&(t<hi),"after":t>hi}[which]
    def stand(xc, how):
        o=xc.argsort(-1,descending=True); R=xc.gather(-1,o[:,2:3])
        v={"median":th.special.ndtri(th.special.ndtr(R)/2),"zero":th.zeros_like(R)}[how]
        out=v.expand_as(xc).clone(); m3=th.zeros_like(xc,dtype=th.bool).scatter(-1,o[:,:3],True); out[m3]=xc[m3]; return out
    def ranks(xc):
        o=xc.argsort(-1,descending=True); R=xc.gather(-1,o[:,2:3]); u=(th.arange(7,0,-1,device=dev)-.375)/7.25
        v=th.special.ndtri(th.special.ndtr(R)*u[None]); out=xc.clone(); out.scatter_(-1,o[:,3:],v); return out
    def events(xc, k):
        m=th.zeros_like(xc,dtype=th.bool)
        for t in range(10):
            m[:,t]=True if t<k else xc[:,t]>xc[:,:t].topk(k,-1).values[:,-1]
        return m
    def near(xc, d, far=False):
        R=xc.gather(-1,xc.argsort(-1,descending=True)[:,2:3]); return (xc<R-d) if far else (xc>=R-d)
    V={"MF3":lambda xc: topk_mf(xc,3)}
    for k in (4,5,6,7): V[f"MF{k}"]=(lambda k: lambda xc: topk_mf(xc,k))(k)
    for t in range(10): V[f"keep t={t}"]=(lambda t: lambda xc: topk_mf(xc,3,th.arange(10,device=dev)[None].expand_as(xc)==t))(t)
    for z in ("before","between","after"): V[f"zone {z}"]=(lambda z: lambda xc: topk_mf(xc,3,pos_mask(xc,z)))(z)
    for h in ("median","zero"): V[f"stand-in {h}"]=(lambda h: lambda xc: stand(xc,h))(h)
    V["ranks"]=ranks
    for k in (1,2,3): V[f"events top-{k}"]=(lambda k: lambda xc: topk_mf(xc,3,events(xc,k)))(k)
    for d in (.5,1.): V[f"near R, d={d}"]=(lambda d: lambda xc: topk_mf(xc,3,near(xc,d)))(d); V[f"far, d={d}"]=(lambda d: lambda xc: topk_mf(xc,3,near(xc,d,True)))(d)
    acc,fl,cat=run(W,x,V); L=(acc[:,1:]-acc[:,:1])*100; pop=slice(0,N-6); se=np.sqrt(np.median(fl[pop].sum((1,2)))/2**m)*100
    print(f"{N-6} population networks (s{s}) + 6 zoo, pool 2^{m}; leakage = accuracy(variant) - accuracy(true), points "
          f"(MF3 pairing SE about {se:.3f} per network)")
    print(f"{'variant':18s} {'pop mean':>9s} {'pop SD':>7s} {'pop RMS':>8s} {'RMS kept':>9s}  "+" ".join(f"{n:>8s}" for n in names[-6:]))
    rms0=np.sqrt(np.mean(L[pop,0]**2))
    for j,n in enumerate(V):
        l=L[pop,j]; r=np.sqrt(np.mean(l**2))
        print(f"{n:18s} {l.mean():+9.3f} {l.std():7.3f} {r:8.3f} {100*(1-r/rms0):8.0f}%  "+" ".join(f"{v:+8.3f}" for v in L[-6:,j]))
    print("\nMF3 flips by |MF margin| bin (points; + = MF right and true wrong):")
    edges=["0-.25",".25-.5",".5-1","1-2","2-4",">4"]
    print(f"{'':18s} "+" ".join(f"{e:>14s}" for e in edges))
    for lab,sl in (("pop mean",pop),("mystery",slice(N-6,N-5)),("seed0",slice(N-5,N-4)),("seed2",slice(N-3,N-2))):
        f=fl[sl].mean(0)*100; print(f"{lab:18s} "+" ".join(f"{a:6.3f}/{-b:7.3f}" for a,b in zip(f[0],f[1])))
    print("\nMF3 flips by the wrong answer (points; MF right & true wrong: true picked max / R / lower;  MF wrong & true right: MF picked ...)")
    for lab,sl in (("pop mean",pop),("mystery",slice(N-6,N-5)),("seed0",slice(N-5,N-4)),("seed2",slice(N-3,N-2))):
        c=cat[sl].mean(0)*100; print(f"{lab:18s} true wrong: max {c[0,0]:.3f}  R {c[0,1]:.3f}  lower {c[0,2]:.3f}   MF wrong: max {c[1,0]:.3f}  R {c[1,1]:.3f}  lower {c[1,2]:.3f}")
    np.savez(sys.argv[1].replace(".npz","_z10.npz"),acc=acc,flips=fl,cat=cat,names=np.array(list(V)))
