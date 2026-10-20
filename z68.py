"""Learned small probe sets on the mystery recipe (ChatGPT round 6 found 16 learned cells give about 0.95 points at
16-28 FE on the AlgZoo recipe).  Candidates: the 2,880 MF4 rows of the one-box K = 4 design (720 arrangements x 4 S
values; pop_unif/cells_k4.npz layout).  Per selected row the readout sees its margin m and its bit; every model also
gets log s (RMS of the selected rows' margins), the 10 pulse quantities and, separately, the 20 weight statistics.
Selection: greedy (orthogonal matching pursuit on the ridge residual, on standardized margins) inside each training
fold, k = 4, 8, 16, 32, 64 rows; ridge readout.  Cost: the selected rows' prefix trie (gfeat's formula), the pulses
(6 FE), 30 operations per row and the 5,000-operation allowance.  Fitted on the 1,024, cross-fitted with 8 folds,
applied unchanged to the 512 new networks (independent labels) and the mystery model.
Usage: python z68.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60
from z62 import ridge
KS=(4,8,16,32,64)
def omp(M,y,base,kmax):
    """greedy rows: at each step the candidate whose standardized margin best correlates with the ridge residual"""
    Z=(M-M.mean(0))/np.maximum(M.std(0),1e-9); sel=[]
    for _ in range(kmax):
        F=np.c_[base,M[:,sel]] if sel else base; r=y-ridge(F,y,F)[0]; c=np.abs(Z.T@(r-r.mean())); c[sel]=-1; sel.append(int(c.argmax()))
    return sel
def feats(M,B,sel,base):
    m=M[:,sel]; s=np.sqrt((m**2).mean(1)); return np.c_[base,m/s[:,None],B[:,sel],np.log(s)]
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    _,_,_,c=gfeat.summary(W,1,4,cells=True); M=c[2].astype(float); B=c[0].astype(float)
    ROWS=gfeat.rows(1,4).reshape(720,4,2,10)[:,:,0].reshape(-1,10)                 # MF4 row of cell a*4+k
    def cost(sel):
        lev,_,nv=gfeat.trie(ROWS[sel]); n=[len(p) for p,_ in lev]
        return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(sel)+5000)/2880+6
    for tag,base in (('pulses',P),('pulses + 20 weight statistics',np.c_[P,ST])):
        print(f"\nlearned MF4 rows + {tag}",flush=True)
        cf={k:np.zeros(1024) for k in KS}; costs={k:[] for k in KS}
        for f in range(8):
            A,Bf=tr[fold!=f],tr[fold==f]; sel=omp(M[A],y[A],base[A],max(KS))
            for k in KS: cf[k][Bf]=ridge(feats(M,B,sel[:k],base)[A],y[A],feats(M,B,sel[:k],base)[Bf])[0]; costs[k].append(cost(sel[:k]))
        sel=omp(M[tr],y[tr],base[tr],max(KS))
        for k in KS:
            pr=ridge(feats(M,B,sel[:k],base)[tr],y[tr],feats(M,B,sel[:k],base)[te])[0]; tot=max(max(costs[k]),cost(sel[:k]))
            r=np.sqrt(((cf[k]-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean())
            print(f"  {k:3d} rows {tot:6.1f} FE (fold designs {min(costs[k]):.1f}-{max(costs[k]):.1f}) | cross-fit {100*r:.3f} ({r**2/(pq/tot):.3f}x) | new {100*rn:.3f} ({rn**2/(pqn/tot):.3f}x) | mystery {100*(pr[-1]-y[mys]):+.2f}",flush=True)
        a=gfeat.ARR[np.array(sel[:16])//4]; k=np.array(sel[:16])%4
        print("  first 16 rows chosen on all 1,024 (positions of M, S, R; S slice 0-3, low to high): "+", ".join(f"({p[0]},{p[1]},{p[2]}) s{q}" for p,q in zip(a,k)))
    print(f"\nfor reference, sampling at 10 / 20 / 40 FE has RMS {', '.join(f'{100*np.sqrt(pqn/C):.2f}' for C in (10,20,40))} points on the new networks; the population mean alone {100*np.sqrt(((y[new]-y[tr].mean())**2).mean()):.2f}")
