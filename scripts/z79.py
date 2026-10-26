"""Learning curve to 4,096 training networks: z78's estimator (combined pool of 24,140 rows; pulses, 20 weight
statistics, predicted roles; OMP; ridge) trained on 1,024 (seed 4), 2,048 (+ seed 7) and 4,096 (+ the 2,048 of seed 8,
pop_unif/b128_n2048_seed8.npz, labelled here on an independent bank of 2^22 inputs, seed 79, with roles computed here
offline on 4,096 probes and cached) networks; k = 32 ... 512; scored on the 512 seed-6 networks (seed-61 labels).
Usage: python scripts/z79.py
"""
import os, numpy as np, torch as th
from z2 import load, neuron_features
from z3 import roles as role_rules
from z72 import labels
import gfeat, z60, z64, z66
from z62 import ridge
from z67 import unit_desc, fit_predict_units, net_counts
from z68 import omp, feats
KS=(32,64,128,256,384,512)
if __name__=='__main__':
    Wa,_=load('pop_unif/b128_n1024_seed4.npz'); Wb,_=load('pop_unif/b128_n1024_seed7.npz'); Wc,_=load('pop_unif/b128_n2048_seed8.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz')
    yc=labels(Wc,'pop_unif/b128_n2048_seed8.acc22s79.npy',seed=79)
    cache='pop_unif/roles_b128_n2048_seed8.npz'
    if os.path.exists(cache): R8=np.load(cache)['roles']
    else:
        nm=list(np.load('pop1k/seed0.roles.npz')['names']); XP=th.randn(4096,10,generator=th.Generator().manual_seed(11))
        R8=role_rules(np.concatenate([neuron_features([w[i:i+128] for w in Wc],XP) for i in range(0,2048,128)]),nm); np.savez(cache,roles=R8)
    W=[th.cat([a.float(),b.float(),c.float(),d.float()]) for a,b,c,d in zip(Wa,Wb,Wc,Wn)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n1024_seed7.acc22s72.npy'),yc,np.load('pop_unif/b128_n512_seed6.acc22s61.npy')]
    te=np.arange(4096,4608); pqn=(y[te]*(1-y[te])).mean()
    R=np.r_[np.load('pop_unif/roles_b128_n1024_n512.npz')['roles'][:1024],np.load('pop_unif/roles_b128_n1024_seed7.npz')['roles'],R8,np.zeros((512,16),int)]
    print(f"seed-8 networks: accuracy {100*yc.mean():.2f}% SD {100*yc.std():.2f}",flush=True)
    P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp); D=unit_desc(*Wnp)
    _,_,_,c=gfeat.summary(W,2,4,cells=True); Ms=[c[2].astype(np.float32)]; Bs=[c[0].astype(np.float32)]; rows=[gfeat.rows(2,4).reshape(-1,2,10)[:,0].astype(float)]; del c
    pul=np.concatenate([np.eye(10),-np.eye(10)]); fams=dict(z66.FAMS); fams['double pulses']=(z64.X.astype(float),z64.TGT,None); fams['single pulses']=(pul,np.r_[np.arange(10),np.arange(10)],None)
    for nm,(x,tg,_) in fams.items(): b,m,_=z66.run(Wnp,x,np.asarray(tg)); Ms.append(m.astype(np.float32)); Bs.append(b.astype(np.float32)); rows.append(x)
    M=np.concatenate(Ms,1); B=np.concatenate(Bs,1); ROWS=np.concatenate(rows); del Ms,Bs
    def cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel].astype(np.float32)); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6+1.8
    for tag,trn in (('1,024',np.arange(1024)),('2,048',np.arange(2048)),('4,096',np.arange(4096))):
        fold=np.random.default_rng(0).permutation(len(trn))%8; oof=np.zeros((len(trn),16,8))
        for f in range(8): oof[fold==f]=fit_predict_units(D,R,trn[fold!=f],trn[fold==f],'logistic')
        roles=np.zeros((len(y),10)); roles[trn]=net_counts(oof); roles[te]=net_counts(fit_predict_units(D,R,trn,te,'logistic'))
        base=np.c_[P,ST,roles]; sel=omp(M[trn].astype(float),y[trn],base[trn],max(KS)); print(f"\n{tag} training networks",flush=True)
        for k in KS:
            F=feats(M,B,sel[:k],base).astype(float); e=ridge(F[trn],y[trn],F[te])[0]-y[te]; r=np.sqrt((e**2).mean()); tot=cost(sel[:k])
            print(f"  {k:3d} rows {tot:6.1f} FE | RMS on the 512 {100*r:.3f}, mean {100*e.mean():+.3f} ({r*r/(pqn/tot):.3f}x)",flush=True)
