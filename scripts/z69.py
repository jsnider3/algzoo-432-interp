"""The learned-row frontier (z68) with everything cheap that helped: rows chosen from the MF4 rows of the one-box K = 8
design (5,760 candidates: 720 arrangements x 8 S values), plus the pulses, the 20 weight statistics and the neuron
roles predicted from the weights (z67's logistic unit classifier, about 2 FE; nested inside the folds as in z67).
Selection: z68's greedy OMP inside each training fold, k = 8 ... 256 rows; ridge readout on the selected rows'
normalized margins and bits, log s, and the base quantities.  Cost: the selected rows' prefix trie + 30 operations
per row + 5,000-operation allowance + pulses (6 FE) + role classifier (1.8 FE).  Fitted on the 1,024, cross-fitted
with 8 folds, applied unchanged to the 512 new networks (independent labels) and the mystery model; each budget is
also compared with sampling shrunk toward the population mean (z63, part D).
Usage: python scripts/z69.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60
from z62 import ridge
from z67 import unit_desc, fit_predict_units, net_counts
from z68 import omp, feats
KS=(8,16,32,64,128,256)
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    R=np.load('pop_unif/roles_b128_n1024_n512.npz')['roles']; D=unit_desc(*Wnp)
    _,_,_,c=gfeat.summary(W,1,8,cells=True); M=c[2].astype(float); B=c[0].astype(float)
    ROWS=gfeat.rows(1,8).reshape(720,8,2,10)[:,:,0].reshape(-1,10)
    def cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel]); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6+1.8
    oof=np.zeros((1024,16,8))
    for f in range(8): A,Bf=tr[fold!=f],tr[fold==f]; oof[Bf]=fit_predict_units(D,R,A,Bf,'logistic')
    CO=net_counts(oof); Cnew=net_counts(fit_predict_units(D,R,tr,te,'logistic')); CT={}
    for f in range(8):
        A=tr[fold!=f]; inner=np.zeros((len(A),16,8)); fi=np.arange(len(A))%8
        for g in range(8): inner[fi==g]=fit_predict_units(D,R,A[fi!=g],A[fi==g],'logistic')
        CT[f]=net_counts(inner)
    mu,t2=y[tr].mean(),y[tr].var(); v=((y[new]-mu)**2).mean()
    def shrunk(C): w=t2/(t2+pqn/C); return np.sqrt(w*w*pqn/C+(1-w)**2*v)
    BASE=np.c_[P,ST]
    cf={k:np.zeros(1024) for k in KS}; costs={k:[] for k in KS}
    for f in range(8):
        A,Bf=tr[fold!=f],tr[fold==f]; roles=np.zeros((len(y),CO.shape[1])); roles[A]=CT[f]; roles[Bf]=CO[Bf]
        base=np.c_[BASE,roles]; sel=omp(M[A],y[A],base[A],max(KS))
        for k in KS:
            F=feats(M,B,sel[:k],base); cf[k][Bf]=ridge(F[A],y[A],F[Bf])[0]; costs[k].append(cost(sel[:k]))
        print(f"  fold {f} done",flush=True)
    roles=np.zeros((len(y),CO.shape[1])); roles[tr]=CO; roles[te]=Cnew; base=np.c_[BASE,roles]; sel=omp(M[tr],y[tr],base[tr],max(KS))
    print("learned MF4 rows (K = 8 pool) + pulses + 20 weight statistics + predicted roles")
    for k in KS:
        F=feats(M,B,sel[:k],base); pr=ridge(F[tr],y[tr],F[te])[0]; tot=max(max(costs[k]),cost(sel[:k]))
        r=np.sqrt(((cf[k]-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean())
        print(f"  {k:3d} rows {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.3f}x) | new {100*rn:.3f} mean {100*(pr[:-1]-y[new]).mean():+.3f} ({rn**2/(pqn/tot):.3f}x;"
              f" shrunk sampling {100*shrunk(tot):.2f}, {rn**2/shrunk(tot)**2:.2f}x) | mystery {100*(pr[-1]-y[mys]):+.2f}",flush=True)
