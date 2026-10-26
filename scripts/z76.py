"""What do the learned rows test, and can predicted roles make the readable formula of z63 better?
  A. z74's selection on all 1,024 training networks (8-box K = 4 MF4 pool + pulses + weight statistics + predicted
     roles), first 32 rows: positions of M, S, R, Q (0 = first input), their values, the gaps M - S and S - R, and the
     ridge weight of the row's normalized margin (points of accuracy per training SD); tallies of the kinds of row.
  B. Readable formulas on z63's 180 compact lower-slice rows: S-vs-R separation + log s, plus the predicted numbers of
     comparators and max detectors (z67's logistic unit classifier), plus the three pulse quantities; cross-fitted on
     the 1,024 (role counts of training folds from inner out-of-fold classifiers, as in z67) and applied to the 512
     new networks, with coefficients in points per training SD.
Usage: python scripts/z76.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60
from z62 import ridge
from z67 import unit_desc, fit_predict_units, net_counts
from z68 import omp, feats
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    R=np.load('pop_unif/roles_b128_n1024_n512.npz')['roles']; D=unit_desc(*Wnp)
    oof=np.zeros((1024,16,8))
    for f in range(8): A,Bf=tr[fold!=f],tr[fold==f]; oof[Bf]=fit_predict_units(D,R,A,Bf,'logistic')
    CO=net_counts(oof); Cn=net_counts(fit_predict_units(D,R,tr,te,'logistic')); roles=np.zeros((len(y),10)); roles[tr]=CO; roles[te]=Cn
    CT={}
    for f in range(8):
        A=tr[fold!=f]; inner=np.zeros((len(A),16,8)); fi=np.arange(len(A))%8
        for g in range(8): inner[fi==g]=fit_predict_units(D,R,A[fi!=g],A[fi==g],'logistic')
        CT[f]=net_counts(inner)
    print("A. the first 32 learned rows (8-box K = 4 pool), chosen on all 1,024")
    _,_,_,c=gfeat.summary(W,2,4,cells=True); M=c[2].astype(float); B=c[0].astype(float); ROWS=gfeat.rows(2,4).reshape(-1,2,10)[:,0]
    base=np.c_[P,ST,roles]; sel=omp(M[tr],y[tr],base[tr],32); F=feats(M,B,sel,base); _,r=ridge(F[tr],y[tr],F[tr]); nb=base.shape[1]
    tags={'S just above R (S - R < 0.25)':0,'M at the last position':0,'S at the last position':0,'M right after S':0,'S at least 5 steps old':0,'M - S < 0.25':0}
    for i,(j,w) in enumerate(zip(sel,r.coef_[nb:nb+32])):
        x=ROWS[j]; o=np.argsort(-x); pm,ps,pr_,pqp=o[:4]; vm,vs,vr,vq=x[o[:4]]
        t=[]
        if vs-vr<.25: t.append('S~R'); tags['S just above R (S - R < 0.25)']+=1
        if vm-vs<.25: t.append('M~S'); tags['M - S < 0.25']+=1
        if pm==9: t.append('M last'); tags['M at the last position']+=1
        if ps==9: t.append('S last'); tags['S at the last position']+=1
        if pm==ps+1: t.append('M right after S'); tags['M right after S']+=1
        if 9-ps>=5: t.append('S old'); tags['S at least 5 steps old']+=1
        print(f"  {i+1:2d}. M@{pm} {vm:.2f}, S@{ps} {vs:.2f}, R@{pr_} {vr:.2f}, Q@{pqp} {vq:.2f}  weight {100*w:+.2f}  {' '.join(t)}")
    print("  tallies over the 32: "+", ".join(f"{k} {v}" for k,v in tags.items()))
    print("\nB. readable formulas (180 compact lower-slice rows)")
    lower=gfeat.rows(1,2).reshape(720,2,2,10)[z60.ENDS][:,0,0]; lev,_,nv=gfeat.trie(lower); n=[len(p) for p,_ in lev]
    rows_fe=(256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+22*len(lower)+5000)/2880
    _,_,_,c2=gfeat.summary(W,1,2,cells=True); m=c2[2].reshape(-1,720,2)[:,z60.ENDS,0].astype(float); s=np.sqrt((m**2).mean(1)); a=m/s[:,None]
    Q={'S-vs-R separation':(a/(.1+abs(a))).mean(1),'log s':np.log(s),'predicted comparators':roles[:,6],'predicted max detectors':roles[:,4],
       'pulse+ growth':P[:,6],'pulse+ growth SD':P[:,8],'pulse- growth SD':P[:,9]}
    for keys,fe in ((['S-vs-R separation','log s'],rows_fe),(['S-vs-R separation','log s','predicted comparators','predicted max detectors'],rows_fe+1.8),
                    (list(Q),rows_fe+1.8+6)):
        X=np.stack([Q[k] for k in keys],1); cf=np.zeros(1024)
        for f in range(8):
            A,Bf=tr[fold!=f],tr[fold==f]; XA=X.copy()
            if 'predicted comparators' in keys: XA[A,2],XA[A,3]=CT[f][:,6],CT[f][:,4]     # nested role counts for the training folds
            cf[Bf]=ridge(XA[A],y[A],X[Bf])[0]
        p,rr=ridge(X[tr],y[tr],X[te]); r1=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((p[:-1]-y[new])**2).mean())
        print(f"  {' + '.join(keys)}: {fe:.1f} FE | cross-fit {100*r1:.3f} ({r1**2/(pq/fe):.2f}x) | new {100*rn:.3f} ({rn**2/(pqn/fe):.2f}x) | mystery {100*(p[-1]-y[mys]):+.2f}")
        print("    accuracy = "+f"{100*rr.intercept_:.2f}% "+" ".join(f"{100*b:+.2f} x z({k})" for k,b in zip(keys,rr.coef_)))
