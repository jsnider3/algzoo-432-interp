"""Leakage as structure + small noise: mean field with the 4th-largest exact (MF4), plus a first-order (cut-HDMR) penalty
for the remaining 6 lower inputs, against the same expansion around MF3.

z10: MF3's leakage in R > 0 is two large mechanisms that nearly cancel, the 4th-largest winning (+0.45 points, which MF3
cannot produce) and S-vs-R swaps (3% of inputs flip, net -0.35). Keeping the 4th exact (MF4) removes both; what is left
is a fairly uniform optimism (+0.30, SD 0.29 across networks), from the 6 smallest inputs. If that residual is a sum of
small effects, a one-at-a-time expansion around MF4 should be additive, where the same expansion around MF3 was not
(REPORT 7.13: first order around MF3 leaves 0.1-0.35 points; second order overshoots).
  model  P(correct | kept) = I0 + sum_j (E_j[I] - I0): input j alone at q equal-probability nodes of its law (truncated
         below the k-th largest), the others at their mean; I0 = the mean-field indicator
  pairs  (order 2) + sum_{i<j} (E_ij[I] - E_i[I] - E_j[I] + I0), each pair on its q x q node grid
  exact  order 1 with each E_j[I] integrated exactly along its line (lines.trace_line, about 3.4 forwards per line), on
         every 8th population network and the zoo
Per network, on one pool of sampled inputs (the top-k design here is sampling, to score the leakage model alone):
leakage(true) = mean(I0) - accuracy(true) and leakage(model) = mean(I0) - mean(P), paired on the same inputs.
Cost per configuration: 1 + (10 - k) q forwards.
Usage: python scripts/z11.py pop1k/seed0.npz [log2 pool] [q] [k list, default 3,4 | pairs<k> | exact<k>]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
dev="cuda" if th.cuda.is_available() else "cpu"

def expansion(x, k, q):
    """inputs (B, 1 + (10-k) q, 10): row 0 the MFk point, then each lower input alone at q nodes; also the target"""
    B=x.shape[0]; o=x.argsort(-1,descending=True); xk=x.gather(-1,o[:,k-1:k]); mu=mu_below(xk)
    base=mu.expand(B,10).clone(); base.scatter_(-1,o[:,:k],x.gather(-1,o[:,:k]))
    u=(th.arange(q,device=x.device)+.5)/q; nodes=th.special.ndtri(th.special.ndtr(xk)*u[None])      # (B, q)
    rows=[base[:,None]]
    for j in range(k,10):
        r=base[:,None].repeat(1,q,1); pj=o[:,j]
        r.scatter_(-1,pj[:,None,None].expand(B,q,1),nodes[...,None]); rows.append(r)
    return th.cat(rows,1), target(x)

def pair_rows(x, k, q):
    """rows (B, C(10-k, 2) q^2, 10): each pair of lower inputs on the q x q node grid, the rest at the MFk point"""
    B=x.shape[0]; o=x.argsort(-1,descending=True); xk=x.gather(-1,o[:,k-1:k]); mu=mu_below(xk)
    base=mu.expand(B,10).clone(); base.scatter_(-1,o[:,:k],x.gather(-1,o[:,:k]))
    u=(th.arange(q,device=x.device)+.5)/q; nodes=th.special.ndtri(th.special.ndtr(xk)*u[None])
    a=nodes[:,:,None].expand(B,q,q).reshape(B,q*q); b=nodes[:,None,:].expand(B,q,q).reshape(B,q*q); rows=[]
    for i in range(k,10):
        for j in range(i+1,10):
            r=base[:,None].repeat(1,q*q,1); r.scatter_(-1,o[:,i,None,None].expand(B,q*q,1),a[...,None]); r.scatter_(-1,o[:,j,None,None].expand(B,q*q,1),b[...,None]); rows.append(r)
    return th.cat(rows,1)

@th.no_grad()
def run2(W, x, k, q, rows=14000):
    """order 1 and order 2 on the same q nodes: per network acc(true), mean I0, mean P1, mean P2, and SE of (P2 - true)"""
    N=W[0].shape[0]; n=10-k; npair=n*(n-1)//2; R=1+n*q+npair*q*q; chunk=max(8,rows//R); out=th.zeros(N,5,device=dev)
    for i in range(0,len(x),chunk):
        xc=x[i:i+chunk]; B=len(xc); y=target(xc)
        tr=(forward(*W,xc[None].expand(N,-1,-1)).argmax(-1)==y).float()
        X1,_=expansion(xc,k,q); X=th.cat([X1,pair_rows(xc,k,q)],1)
        I=(forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).argmax(-1).view(N,B,R)==y[None,:,None]).float()
        I0=I[:,:,0]; Es=I[:,:,1:1+n*q].view(N,B,n,q); E1=Es.mean(-1); P1=I0+(E1-I0[...,None]).sum(-1)
        Ep=I[:,:,1+n*q:].view(N,B,npair,q,q); P2=P1.clone(); c=0
        for a in range(n):
            for b in range(a+1,n):
                # E_ab - E_a - E_b + I0 with E_a, E_b on the same nodes (the singles rows)
                P2+=Ep[:,:,c].mean((-1,-2))-E1[:,:,a]-E1[:,:,b]+I0; c+=1
        d=P2.clamp(0,1)-tr
        out+=th.stack([tr.sum(1),I0.sum(1),P1.clamp(0,1).sum(1),P2.clamp(0,1).sum(1),(d*d).sum(1)],1)
    o=out/len(x); return o.cpu().numpy()

@th.no_grad()
def run(W, x, ks, q, rows=14000):
    N=W[0].shape[0]; out={k:th.zeros(N,4,device=dev) for k in ks}                   # acc(true), mean I0, mean P, mean (P - true)^2
    chunk=max(16,rows//(1+(10-min(ks))*q))
    for i in range(0,len(x),chunk):
        xc=x[i:i+chunk]; B=len(xc); y=target(xc)
        tr=(forward(*W,xc[None].expand(N,-1,-1)).argmax(-1)==y).float()
        for k in ks:
            X,_=expansion(xc,k,q); R=X.shape[1]
            I=(forward(*W,X.reshape(1,-1,10).expand(N,-1,-1)).argmax(-1).view(N,B,R)==y[None,:,None]).float()
            I0=I[:,:,0]; Ej=I[:,:,1:].view(N,B,10-k,q).mean(-1); P=(I0+(Ej-I0[...,None]).sum(-1)).clamp(0,1)
            out[k]+=th.stack([tr.sum(1),I0.sum(1),P.sum(1),((P-tr)**2).sum(1)],1)
    return {k:(v/len(x)).cpu().numpy() for k,v in out.items()}

if __name__=="__main__":
    P_,s=load(sys.argv[1]); m=int(sys.argv[2]) if len(sys.argv)>2 else 18; q=int(sys.argv[3]) if len(sys.argv)>3 else 4
    from seeds_clocks import load as zload
    Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
    Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
    W=[th.cat([a,b]).to(dev) for a,b in zip(P_,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6); zoo=["mystery"]+[f"seed{i}" for i in range(5)]
    x=th.randn(2**m,10,generator=th.Generator().manual_seed(13)).to(dev)
    if len(sys.argv)>4 and sys.argv[4].startswith("pairs"):
        k=int(sys.argv[4][5:] or 4); r=run2(W,x,k,q); se=np.sqrt(np.median(r[pop,4])/2**m)*100
        print(f"{N-6} population networks + 6 zoo; pool 2^{m}; MF{k}; q = {q} for both orders; points (pool noise about {se:.3f} per network)")
        for lab,e in ((f"MF{k} alone",r[:,1]),(f"+ 1st order ({1+(10-k)*q} fwd)",r[:,2]),(f"+ 2nd order ({1+(10-k)*q+(10-k)*(9-k)//2*q*q} fwd)",r[:,3])):
            e=(e-r[:,0])*100; l=e[pop]; print(f"  {lab:28s} pop mean {l.mean():+.3f}  SD {l.std():.3f}  RMS {np.sqrt((l**2).mean()):.3f}   "+" ".join(f"{n} {v:+.3f}" for n,v in zip(zoo,e[-6:])))
        np.save(sys.argv[1].replace(".npz",f"_z11_pairs{k}_q{q}.npy"),r); sys.exit()
    if len(sys.argv)>4 and sys.argv[4].startswith("exact"):
        from lines import trace_line
        k=int(sys.argv[4][5:] or 4); sel=list(range(0,N-6,8))+list(range(N-6,N)); xs=x.cpu().numpy().astype(np.float64); B=len(xs)
        o=np.argsort(-xs,1); xk=xs[np.arange(B),o[:,k-1]]; base=np.repeat(mu_below(th.tensor(xk))[:,None].numpy(),10,1)
        np.put_along_axis(base,o[:,:k],np.take_along_axis(xs,o[:,:k],1),1); tgt=o[:,1].astype(np.int64); out=[]
        def fwd(Wn,X):
            h=np.zeros((len(X),16))
            for t in range(10): h=np.maximum(X[:,t:t+1]*Wn[0]+h@Wn[1].T,0)
            return (h@Wn[2].T).argmax(1)
        for nidx in sel:
            Wn=[w[nidx].cpu().numpy().astype(np.float64) for w in W]; T_=(fwd(Wn,xs)==tgt).astype(float); I0=(fwd(Wn,base)==tgt).astype(float); Pm=I0.copy(); cst=1.0
            for j in range(k,10):
                Pj,c,ov=trace_line(Wn[0],Wn[1],Wn[2],base,o[:,j].astype(np.int64),np.full(B,-8.0),xk,tgt); Pm+=Pj-I0; cst+=c.mean()
                if ov.any(): raise RuntimeError(f"line-tracing overflow (MAXP) on {int(ov.sum())} lines; result not exact")
            Pm=np.clip(Pm,0,1); out.append((T_.mean(),I0.mean(),Pm.mean(),np.mean((Pm-T_)**2),cst))
        r=np.array(out); e0=(r[:,1]-r[:,0])*100; e1=(r[:,2]-r[:,0])*100; npop=len(sel)-6
        print(f"{npop} population networks (every 8th) + 6 zoo; pool 2^{m}; MF{k}; exact lines; points (pool noise about {np.sqrt(np.median(r[:npop,3])/B)*100:.3f} per network);"
              f" cost {r[:,4].mean():.1f} forwards per configuration")
        for lab,e in ((f"MF{k} alone",e0),(f"MF{k} + exact 1st order",e1)):
            l=e[:npop]; print(f"  {lab:26s} pop mean {l.mean():+.3f}  SD {l.std():.3f}  RMS {np.sqrt((l**2).mean()):.3f}   "+" ".join(f"{n} {v:+.3f}" for n,v in zip(zoo,e[-6:])))
        np.save(sys.argv[1].replace(".npz",f"_z11_exact{k}.npy"),r); sys.exit()
    ks=tuple(int(k) for k in (sys.argv[4] if len(sys.argv)>4 else "3,4").split(",")); res=run(W,x,ks,q)
    print(f"{N-6} population networks (s{s}) + 6 zoo; pool 2^{m}; q = {q}; points; residual = estimate - true accuracy")
    for k,r in res.items():
        lt=(r[:,1]-r[:,0])*100; lm=(r[:,1]-r[:,2])*100; e0=lt; e1=lt-lm                # MFk alone, MFk + first-order penalty
        for lab,e in ((f"MF{k} alone",e0),(f"MF{k} + 1st-order penalty ({1+(10-k)*q} fwd/config)",e1)):
            l=e[pop]; print(f"  {lab:44s} pop mean {l.mean():+.3f}  SD {l.std():.3f}  RMS {np.sqrt((l**2).mean()):.3f}   "+" ".join(f"{n} {v:+.3f}" for n,v in zip(zoo,e[-6:])))
        print(f"  predicted vs true MF{k} leakage across the population: corr {np.corrcoef(lm[pop],lt[pop])[0,1]:.3f}, slope {np.polyfit(lm[pop],lt[pop],1)[0]:.2f};"
              f" pool noise in the model residual about {np.sqrt(np.median(r[pop,3])/2**m)*100:.3f} per network")
    np.savez(sys.argv[1].replace(".npz",f"_z11_q{q}_k{'-'.join(map(str,ks))}.npz"),**{f"k{k}":v for k,v in res.items()})
