"""Does doubling the training population help?  After z72's blind test, the 1,024 seed-7 networks
(pop_unif/b128_n1024_seed7.npz, labels on the seed-72 bank) join the 1,024 training networks.  Learned-row models
(z74's 8-box K = 4 MF4 pool + pulses + 20 weight statistics + role counts from z67's logistic unit classifier) and the
compact design's ridge (44 named quantities + weight statistics + role counts) are fitted on 1,024 or on 2,048
networks and scored on the 512 seed-6 networks (seed-61 labels), which neither training set contains.  Role counts of
training networks are out-of-fold (8 folds); the unit classifier needs the neuron roles of the seed-7 networks, which
are computed here (z2.neuron_features on 4,096 probes, z3.roles; offline) and cached.
Usage: python z75.py
"""
import os, numpy as np, torch as th
from z2 import load, neuron_features
from z3 import roles as role_rules
import gfeat, z60
from z62 import named, ridge
from z67 import unit_desc, fit_predict_units, net_counts
from z68 import omp, feats
KS=(16,32,64,128,256)
if __name__=='__main__':
    Wa,_=load('pop_unif/b128_n1024_seed4.npz'); Wb,_=load('pop_unif/b128_n1024_seed7.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz')
    W=[th.cat([a.float(),b.float(),c.float()]) for a,b,c in zip(Wa,Wb,Wn)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n1024_seed7.acc22s72.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy')]
    A,AB,te=np.arange(1024),np.arange(2048),np.arange(2048,2560); pqn=(y[te]*(1-y[te])).mean()
    cache='pop_unif/roles_b128_n1024_seed7.npz'
    if os.path.exists(cache): R7=np.load(cache)['roles']
    else:
        nm=list(np.load('pop1k/seed0.roles.npz')['names']); XP=th.randn(4096,10,generator=th.Generator().manual_seed(11))
        R7=role_rules(np.concatenate([neuron_features([w[i:i+128] for w in Wb],XP) for i in range(0,1024,128)]),nm); np.savez(cache,roles=R7)
    R=np.r_[np.load('pop_unif/roles_b128_n1024_n512.npz')['roles'][:1024],R7,np.zeros((512,16),int)]
    P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp); D=unit_desc(*Wnp)
    _,_,_,c=gfeat.summary(W,2,4,cells=True); M=c[2].astype(float); B=c[0].astype(float); ROWS=gfeat.rows(2,4).reshape(-1,2,10)[:,0]
    _,_,_,c2=gfeat.summary(W,1,2,cells=True); Xn,_=named(dict(zip(('I4','I3','m4','m3'),c2)),z60.ENDS,P)
    def cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel]); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6+1.8
    for tag,trn in (('1,024 training networks',A),('2,048 training networks',AB)):
        fold=np.random.default_rng(0).permutation(len(trn))%8; oof=np.zeros((len(trn),16,8))
        for f in range(8): oof[fold==f]=fit_predict_units(D,R,trn[fold!=f],trn[fold==f],'logistic')
        roles=np.zeros((len(y),10)); roles[trn]=net_counts(oof); roles[te]=net_counts(fit_predict_units(D,R,trn,te,'logistic'))
        base=np.c_[P,ST,roles]; sel=omp(M[trn],y[trn],base[trn],max(KS)); print(f"\n{tag}",flush=True)
        for k in KS:
            F=feats(M,B,sel[:k],base); e=ridge(F[trn],y[trn],F[te])[0]-y[te]; r=np.sqrt((e**2).mean()); tot=cost(sel[:k])
            print(f"  learned rows, {k:3d}: {tot:6.1f} FE | RMS on the 512 {100*r:.3f}, mean {100*e.mean():+.3f} ({r*r/(pqn/tot):.3f}x)",flush=True)
        F=np.c_[Xn,ST,roles]; e=ridge(F[trn],y[trn],F[te])[0]-y[te]; r=np.sqrt((e**2).mean()); tot=z60.total_cost(2,'4',z60.ENDS,1,1,1,0)+1.8
        print(f"  compact 44 + weight statistics + roles: {tot:6.1f} FE | RMS on the 512 {100*r:.3f}, mean {100*e.mean():+.3f} ({r*r/(pqn/tot):.3f}x)",flush=True)
