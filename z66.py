"""Probes that put lower-input variation into memory, aimed at the network-wide breaking rate of z65.
Base: z63's minimal design, the compact design's 180 MF4 rows at the lower S slice (M 1.388, S 0.732, R 0.577,
Q 0.372, the six lower inputs at mu(Q) = -0.577).  Families (all fixed, structured inputs):
  high / low     the same 180 rows with all six lower inputs at 0.20 (their conditional 90th percentile below Q)
                 or at -1.52 (10th percentile);
  alternating    the six lower inputs alternate 0.20 / -1.52 in order of position, in both phases (360 rows);
  distractors    zero background, the max 1.5 at i and the 2nd max 1 at j (all 90 ordered pairs), every later position
                 after both filled with a distractor at 0.95 or 0.75 (180 rows); the answer is j.
Per family: the share answered right (bit), the mean squashed margin a/(.1+|a|) with a = margin / s (s from the base
rows; for the distractors the row's centred-logit RMS), and for the base-aligned families the mean change of
a/(.1+|a|) from the base row, overall and over rows whose S is at least 4 steps old.
Cost: the union prefix trie of base + probe rows minus the base trie (gfeat's formula) + 30 operations per row.
Each family is added (a) to the minimal estimator's five quantities (S-vs-R separation, log s, three pulse
quantities; forward selection inside the folds, k <= 6, and ridge on all) and (b) to the compact design's 44 named
quantities + 20 weight statistics (ridge on all); fitted on the 1,024, cross-fitted (8 folds), applied unchanged to
the 512 new networks.  The 6-feature model's residual is checked against max detectors and the late pointer.
Usage: python z66.py
"""
import numpy as np, torch as th
from z2 import load
from z3 import ROLES
from seeds_clocks import load as zload
import gfeat, z60
from z62 import named, ridge, forward
HI,LO=0.20,-1.52
def trie_fe(x):
    lev,_,nv=gfeat.trie(x); n=[len(p) for p,_ in lev]
    return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1])/2880
def run(W,x,tgt):
    wi,A,O=[th.tensor(w,dtype=th.float64) for w in W]; xx=th.tensor(x,dtype=th.float64); h=th.zeros(len(wi),len(x),16,dtype=th.float64)
    for t in range(10): h=th.relu(h@A.transpose(1,2)+xx[None,:,t,None]*wi[:,None,:])
    L=(h@O.transpose(1,2)).numpy(); ii=np.arange(len(x)); tl=L[:,ii,tgt]; o=L.copy(); o[:,ii,tgt]=-np.inf
    c=L-L.mean(2,keepdims=True); return L.argmax(2)==tgt[None], tl-o.max(2), np.sqrt((c*c).mean(2))
BASE=gfeat.rows(1,2).reshape(720,2,2,10)[z60.ENDS][:,0,0].astype(np.float64); ARR=gfeat.ARR[z60.ENDS]; BT=ARR[:,1]
LOWER=np.isclose(BASE,BASE.min(1,keepdims=True)); OLD_S=(9-ARR[:,1])>=4
def fam_shift(v):
    x=BASE.copy(); x[LOWER]=v; return x,BT,OLD_S
def fam_alt():
    xs=[]
    for ph in (0,1):
        x=BASE.copy()
        for r in range(len(x)):
            pos=np.where(LOWER[r])[0]; x[r,pos]=[HI if (k+ph)%2==0 else LO for k in range(len(pos))]
        xs.append(x)
    return np.concatenate(xs),np.r_[BT,BT],np.r_[OLD_S,OLD_S]
def fam_distract():
    xs=[]; tg=[]; old=[]
    for d in (.95,.75):
        for i in range(10):
            for j in range(10):
                if i==j: continue
                x=np.zeros(10); x[i]=1.5; x[j]=1.; x[max(i,j)+1:]=d; xs.append(x); tg.append(j); old.append(9-j>=4)
    return np.array(xs),np.array(tg),np.array(old)
FAMS={'lower inputs high':fam_shift(HI),'lower inputs low':fam_shift(LO),'alternating':fam_alt(),'distractors':fam_distract()}

if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8; P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    _,_,_,c=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),c)); Xn,_=named(cells,z60.ENDS,P)
    b0,m0,_=run(Wnp,BASE,BT); s=np.sqrt((m0**2).mean(1)); a0=m0/s[:,None]; soft0=a0/(.1+abs(a0))
    MQ=np.c_[soft0.mean(1),np.log(s),P[:,6],P[:,8],P[:,9]]; base_fe=trie_fe(BASE)+(22*len(BASE)+5000)/2880+6
    print(f"check: base rows reproduce z63's S-vs-R separation to {np.abs(MQ[:,0]-(lambda m: (m/np.sqrt((m**2).mean(1))[:,None]/(.1+abs(m/np.sqrt((m**2).mean(1))[:,None]))).mean(1))(cells['m4'].reshape(-1,720,2)[:,z60.ENDS,0].astype(float))).max():.1e}")
    z=np.load('pop_unif/roles_b128_n1024_n512.npz'); R=z['roles']; mdet=(R==4).sum(1); late=(R==3).any(1).astype(float)
    def score(tag,tot,cf,pr):
        r=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean())
        print(f"  {tag:50s} {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.2f}x) | new {100*rn:.3f} ({rn**2/(pqn/tot):.2f}x) | mystery {100*(pr[-1]-y[mys]):+.2f}",flush=True)
        return cf-y[tr]
    def cfridge(F):
        cf=np.zeros(1024)
        for f in range(8): A,B=tr[fold!=f],tr[fold==f]; cf[B]=ridge(F[A],y[A],F[B])[0]
        return cf,ridge(F[tr],y[tr],F[te])[0]
    def fwd(F,names,kmax=6):
        cf=np.zeros(1024)
        for f in range(8): A,B=tr[fold!=f],tr[fold==f]; ss=forward(F[A],y[A],kmax); cf[B]=ridge(F[A][:,ss],y[A],F[B][:,ss])[0]
        ss=forward(F[tr],y[tr],kmax); return cf,ridge(F[tr][:,ss],y[tr],F[te][:,ss])[0],[names[j] for j in ss]
    mn=['S-vs-R separation','log s','pulse+ growth','pulse+ growth SD','pulse- growth SD']
    print(f"\nbase: minimal design {len(BASE)} rows, {base_fe:.1f} FE (with pulses)")
    e=score('minimal, 5 quantities, ridge',base_fe,*cfridge(MQ)); cf,pr,sel=fwd(MQ,mn,5); score('minimal, forward 5',base_fe,cf,pr)
    cfull=z60.total_cost(2,'4',z60.ENDS,1,1,1,0); score('compact 44 named + 20 weight statistics, ridge',cfull,*cfridge(np.c_[Xn,ST]))
    for nm,(x,tg,old) in FAMS.items():
        bb,mm,rms=run(Wnp,x,tg); base_aligned=nm!='distractors'
        a=mm/(s[:,None] if base_aligned else np.maximum(rms,1e-12)); soft=a/(.1+abs(a))
        F={f'{nm} bit':bb.mean(1),f'{nm} soft':soft.mean(1),f'{nm} bit, old S':bb[:,old].mean(1),f'{nm} soft, old S':soft[:,old].mean(1)}
        if base_aligned:
            k=len(x)//len(BASE); d=soft-np.tile(soft0,(1,k)); F[f'{nm} change']=d.mean(1); F[f'{nm} change, old S']=d[:,np.tile(OLD_S,k)].mean(1)
        D=np.stack(list(F.values()),1); dn=list(F); pfe=trie_fe(np.r_[BASE,x])-trie_fe(BASE)+30*len(x)/2880
        print(f"\n{nm}: {len(x)} rows, +{pfe:.1f} FE; "+"; ".join(f"{k} mean {v[tr].mean():.3f} r(acc) {np.corrcoef(v[tr],y[tr])[0,1]:+.2f} r(resid) {np.corrcoef(v[tr],e)[0,1]:+.2f} r(max-det) {np.corrcoef(v[tr],mdet[tr])[0,1]:+.2f}" for k,v in F.items()),flush=True)
        score('minimal + family, ridge',base_fe+pfe,*cfridge(np.c_[MQ,D]))
        cf,pr,sel=fwd(np.c_[MQ,D],mn+dn,6); r6=score('minimal + family, forward 6',base_fe+pfe,cf,pr); print("    selected: "+"; ".join(sel))
        score('compact 44 + weight statistics + family, ridge',cfull+pfe,*cfridge(np.c_[Xn,ST,D]))
        print(f"    forward-6 residual vs max detectors {np.corrcoef(r6,mdet[tr])[0,1]:+.2f}, late pointer {np.corrcoef(r6,late[tr])[0,1]:+.2f}",flush=True)
