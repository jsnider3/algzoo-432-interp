"""The learned-row frontier with 2,048 training networks, to larger sizes: pool = z74's 8-box K = 4 MF4 rows (23,040)
+ z71's other families (single and double pulses, lower inputs high / low / alternating, distractor trains: 1,100),
with pulses, 20 weight statistics and predicted roles (z67's logistic classifier, out-of-fold on the training set);
k = 32 ... 512 rows chosen by z68's OMP on the 2,048 training networks (seeds 4 and 7); ridge readout; scored on the
512 seed-6 networks (seed-61 labels).
Usage: python z78.py
"""
import numpy as np, torch as th
from z2 import load
import gfeat, z60, z64, z66
from z62 import ridge
from z67 import unit_desc, fit_predict_units, net_counts
from z68 import omp, feats
KS=(32,64,128,256,384,512)
if __name__=='__main__':
    Wa,_=load('pop_unif/b128_n1024_seed4.npz'); Wb,_=load('pop_unif/b128_n1024_seed7.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz')
    W=[th.cat([a.float(),b.float(),c.float()]) for a,b,c in zip(Wa,Wb,Wn)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n1024_seed7.acc22s72.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy')]
    trn,te=np.arange(2048),np.arange(2048,2560); pqn=(y[te]*(1-y[te])).mean()
    R=np.r_[np.load('pop_unif/roles_b128_n1024_n512.npz')['roles'][:1024],np.load('pop_unif/roles_b128_n1024_seed7.npz')['roles'],np.zeros((512,16),int)]
    P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp); D=unit_desc(*Wnp)
    _,_,_,c=gfeat.summary(W,2,4,cells=True); Ms=[c[2].astype(float)]; Bs=[c[0].astype(float)]; rows=[gfeat.rows(2,4).reshape(-1,2,10)[:,0].astype(float)]; del c
    pul=np.concatenate([np.eye(10),-np.eye(10)]); fams=dict(z66.FAMS); fams['double pulses']=(z64.X.astype(float),z64.TGT,None); fams['single pulses']=(pul,np.r_[np.arange(10),np.arange(10)],None)
    for nm,(x,tg,_) in fams.items(): b,m,_=z66.run(Wnp,x,np.asarray(tg)); Ms.append(m); Bs.append(b.astype(float)); rows.append(x)
    M=np.concatenate(Ms,1); B=np.concatenate(Bs,1); ROWS=np.concatenate(rows); del Ms,Bs
    def cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel].astype(np.float32)); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6+1.8
    fold=np.random.default_rng(0).permutation(len(trn))%8; oof=np.zeros((len(trn),16,8))
    for f in range(8): oof[fold==f]=fit_predict_units(D,R,trn[fold!=f],trn[fold==f],'logistic')
    roles=np.zeros((len(y),10)); roles[trn]=net_counts(oof); roles[te]=net_counts(fit_predict_units(D,R,trn,te,'logistic'))
    base=np.c_[P,ST,roles]; sel=omp(M[trn],y[trn],base[trn],max(KS)); kind=np.array(['MF4']*23040+sum([[k]*len(v[0]) for k,v in fams.items()],[]))
    print(f"{M.shape[1]} candidate rows; 2,048 training networks",flush=True)
    for k in KS:
        F=feats(M,B,sel[:k],base); e=ridge(F[trn],y[trn],F[te])[0]-y[te]; r=np.sqrt((e**2).mean()); tot=cost(sel[:k]); ks=kind[sel[:k]]
        print(f"  {k:3d} rows {tot:6.1f} FE | RMS on the 512 {100*r:.3f}, mean {100*e.mean():+.3f} ({r*r/(pqn/tot):.3f}x) | non-MF4 rows: "
              +", ".join(f"{q} {int((ks==q).sum())}" for q in fams if (ks==q).any()),flush=True)
