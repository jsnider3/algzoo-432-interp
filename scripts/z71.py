"""Learned rows from a mixed pool: does the selection prefer other kinds of probe once it may choose them?
Candidates (rows with a target output; each gives a margin and a bit):
  MF4 rows of the one-box K = 8 design (5,760),
  z66's families: the compact lower-slice rows with the lower inputs high / low (180 + 180), alternating (360) and
  distractor trains (180),
  z64's double pulses (180),
  the 20 single pulses (+-1 at one position; target: the pulse's position).
Selection: z68's greedy OMP inside each training fold; ridge readout with the pulses' 10 quantities and the 20
weight statistics; k = 16, 32, 64, 128.  Cost: the selected rows' prefix trie + 30 operations per row + 5,000-operation
allowance + 6 FE for the pulse quantities.  Fitted on the 1,024, cross-fitted, applied unchanged to the 512 new
networks; the kinds of row chosen are counted.
Usage: python scripts/z71.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60, z64, z66
from z62 import ridge
from z68 import omp, feats
KS=(16,32,64,128)
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; base=np.c_[z60.pulses(*Wnp),gfeat.stats(*Wnp)]
    _,_,_,c=gfeat.summary(W,1,8,cells=True); Ms=[c[2].astype(float)]; Bs=[c[0].astype(float)]
    rows=[gfeat.rows(1,8).reshape(720,8,2,10)[:,:,0].reshape(-1,10).astype(float)]; kind=['MF4']*len(rows[0])
    pul=np.concatenate([np.eye(10),-np.eye(10)]); fams=dict(z66.FAMS); fams['double pulses']=(z64.X.astype(float),z64.TGT,None); fams['single pulses']=(pul,np.r_[np.arange(10),np.arange(10)],None)
    for nm,(x,tg,_) in fams.items():
        b,m,_=z66.run(Wnp,x,np.asarray(tg)); Ms.append(m); Bs.append(b.astype(float)); rows.append(x); kind+= [nm]*len(x)
    M=np.concatenate(Ms,1); B=np.concatenate(Bs,1); ROWS=np.concatenate(rows); kind=np.array(kind)
    print(f"{M.shape[1]} candidate rows: "+", ".join(f"{k} {int((kind==k).sum())}" for k in dict.fromkeys(kind)),flush=True)
    def cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel].astype(np.float32)); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6
    cf={k:np.zeros(1024) for k in KS}; costs={k:[] for k in KS}
    for f in range(8):
        A,Bf=tr[fold!=f],tr[fold==f]; sel=omp(M[A],y[A],base[A],max(KS))
        for k in KS: F=feats(M,B,sel[:k],base); cf[k][Bf]=ridge(F[A],y[A],F[Bf])[0]; costs[k].append(cost(sel[:k]))
    sel=omp(M[tr],y[tr],base[tr],max(KS))
    for k in KS:
        F=feats(M,B,sel[:k],base); pr=ridge(F[tr],y[tr],F[te])[0]; tot=max(max(costs[k]),cost(sel[:k]))
        r=np.sqrt(((cf[k]-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean()); ks=kind[sel[:k]]
        print(f"  {k:3d} rows {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.3f}x) | new {100*rn:.3f} ({rn**2/(pqn/tot):.3f}x) | mystery {100*(pr[-1]-y[mys]):+.2f} | kinds: "
              +", ".join(f"{q} {int((ks==q).sum())}" for q in dict.fromkeys(kind) if (ks==q).any()),flush=True)
