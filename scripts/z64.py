"""A cheap probe of handoffs: does adding it fix what the readable models of z62/z63 miss (networks with max detectors
underestimated, late-pointer networks overestimated)?
Double pulses: sequences that are zero except x_i = 1 and x_j = r at positions i < j (45 pairs).
  handoff  r = 1.25 ("close") or 2 ("clear"): the later input is a new max, so the right answer is i; the network
           must hand the old max over to the 2nd-max slot, at the last step too (j = 9), where a one-step-late
           pointer has no step left;
  new 2nd  r = 0.8 ("close") or 0.5 ("clear"): the later input is a new 2nd max, so the right answer is j.
180 rows.  By positive homogeneity only the ratio r matters.  Per family and r: the share answered right (bit) and
the mean squashed margin m/(0.1+|m|) (soft), m = target logit minus the best other over the RMS of the row's
centred logits (invariant under the exact symmetries), over all 45 pairs and over the 9 with j = 9 (last).
  1. correlation of each probe quantity with accuracy and with neuron roles (z63's cache), on the 1,024;
  2. the probe added to z63's minimal estimator (180 MF4 rows at the lower S slice; S-vs-R separation, log s, the
     pulse quantities): forward selection inside each fold (z62.forward) from the minimal quantities with and
     without the probe;
  3. the probe added to z62's 44 named quantities (all and compact designs): forward selection, k = 3, 6, 12, and
     ridge on all;
  4. the 6-quantity model's residual against roles, with the probe.
Fitted on the 1,024 training networks, cross-fitted with 8 folds (permutation 0), and applied unchanged to the 512
new networks (independent seed-61 labels) and the mystery model.  Cost: the probe's prefix trie (gfeat's formula)
plus 30 operations per row.
Usage: python scripts/z64.py
"""
import itertools, numpy as np, torch as th
from z2 import load
from z3 import ROLES
from seeds_clocks import load as zload
import gfeat, z60
from z62 import named, ridge, forward
PAIRS=[(i,j) for i,j in itertools.combinations(range(10),2)]; FAM=[('handoff close',1.25),('handoff clear',2.),('new 2nd close',.8),('new 2nd clear',.5)]
X=np.zeros((len(FAM)*len(PAIRS),10),np.float32); TGT=[]; LAST=[]
for f,(nm,r) in enumerate(FAM):
    for p,(i,j) in enumerate(PAIRS): X[f*45+p,i]=1; X[f*45+p,j]=r; TGT.append(i if r>1 else j); LAST.append(j==9)
TGT=np.array(TGT); LAST=np.array(LAST)
lev,_,nv=gfeat.trie(X); n=[len(p) for p,_ in lev]
PROBE_FE=(256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1]+30*len(X))/2880
def probe(W):
    wi,A,O=[th.tensor(w,dtype=th.float64) for w in W]; x=th.tensor(X,dtype=th.float64); h=th.zeros(len(wi),len(x),16,dtype=th.float64)
    for t in range(10): h=th.relu(h@A.transpose(1,2)+x[None,:,t,None]*wi[:,None,:])
    L=(h@O.transpose(1,2)).numpy(); c=L-L.mean(2,keepdims=True); rms=np.maximum(np.sqrt((c*c).mean(2)),1e-12)
    ii=np.arange(len(X)); tl=L[:,ii,TGT]; oth=L.copy(); oth[:,ii,TGT]=-np.inf; m=(tl-oth.max(2))/rms; ok=L.argmax(2)==TGT[None]
    F={}
    for f,(nm,r) in enumerate(FAM):
        s=slice(f*45,f*45+45); last=LAST[s]
        F[f'{nm} bit']=ok[:,s].mean(1); F[f'{nm} soft']=(m[:,s]/(.1+abs(m[:,s]))).mean(1)
        F[f'{nm} last bit']=ok[:,s][:,last].mean(1); F[f'{nm} last soft']=(m[:,s][:,last]/(.1+abs(m[:,s][:,last]))).mean(1)
    return np.stack(list(F.values()),1),list(F)

if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    _,_,_,c=gfeat.summary(W,1,2,cells=True); cells=dict(zip(('I4','I3','m4','m3'),c)); P=z60.pulses(*Wnp); ST=gfeat.stats(*Wnp)
    D,dn=probe(Wnp); fold=np.random.default_rng(0).permutation(1024)%8
    z=np.load('pop_unif/roles_b128_n1024_n512.npz'); R=z['roles']; cnt=np.stack([(R==k).sum(1) for k in range(len(ROLES))],1)
    late=(R==3).any(1); event=(R==2).any(1); RA=np.c_[cnt,late,event].astype(float); RN=ROLES+['late pointer','event pointer']
    print(f"probe: {len(X)} rows, {PROBE_FE:.1f} FE (prefix trie {n})")
    def score(tag,tot,cf,pr):
        r=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((pr[:-1]-y[new])**2).mean())
        print(f"  {tag:52s} {tot:6.1f} FE | cross-fit {100*r:.3f} ({r**2/(pq/tot):.2f}x) | new {100*rn:.3f} mean {100*(pr[:-1]-y[new]).mean():+.3f} ({rn**2/(pqn/tot):.2f}x) | mystery {100*(pr[-1]-y[mys]):+.2f}",flush=True)
        return cf-y[tr]
    def fwd(Xq,names,tot,ks,tag):
        cfk={k:np.zeros(1024) for k in ks}
        for f in range(8):
            a,b=tr[fold!=f],tr[fold==f]; s=forward(Xq[a],y[a],max(ks))
            for k in ks: cfk[k][b]=ridge(Xq[a][:,s[:k]],y[a],Xq[b][:,s[:k]])[0]
        s=forward(Xq[tr],y[tr],max(ks)); print(f"  selected ({tag}): "+"; ".join(f"{i+1}. {names[j]}" for i,j in enumerate(s)),flush=True)
        return {k:score(f'{tag}, {k} quantities',tot,cfk[k],ridge(Xq[tr][:,s[:k]],y[tr],Xq[te][:,s[:k]])[0]) for k in ks}
    def allridge(Xq,tot,tag):
        cf=np.zeros(1024)
        for f in range(8): a,b=tr[fold!=f],tr[fold==f]; cf[b]=ridge(Xq[a],y[a],Xq[b])[0]
        return score(tag,tot,cf,ridge(Xq[tr],y[tr],Xq[te])[0])

    print("\n1. the probe quantities on the 1,024")
    for k,nm in enumerate(dn):
        v=D[tr,k]; r=[np.corrcoef(v,RA[tr,j])[0,1] if RA[tr,j].std()>0 else 0 for j in range(RA.shape[1])]; o=np.argsort(-np.abs(r))[:4]
        print(f"  {nm:26s} mean {v.mean():.3f}  r with accuracy {np.corrcoef(v,y[tr])[0,1]:+.2f}; roles: "+", ".join(f"{RN[j]} {r[j]:+.2f}" for j in o))

    print("\n2. the minimal estimator (180 MF4 rows at the lower S slice) with and without the probe")
    xr=gfeat.rows(1,2).reshape(720,2,2,10)[z60.ENDS][:,0,0]; lv,_,nv2=gfeat.trie(xr); n2=[len(p) for p,_ in lv]
    base_fe=(256*sum(n2[:-1])+16*nv2+160*n2[-1]+16*sum(n2[1:])+16*sum(n2)+20*n2[-1]+22*len(xr)+5000)/2880+6
    m=cells['m4'].reshape(-1,720,2)[:,z60.ENDS,0].astype(float); s=np.sqrt((m**2).mean(1)); a=m/s[:,None]
    MQ=np.c_[(a/(.1+abs(a))).mean(1),np.log(s),P[:,6],P[:,8],P[:,9]]; mn=['S-vs-R separation','log s','pulse+ growth','pulse+ growth SD','pulse- growth SD']
    allridge(MQ[:,:2],base_fe-6,'S-vs-R separation + log s (z63)')
    allridge(np.c_[MQ[:,:2],D],base_fe-6+PROBE_FE,'S-vs-R separation + log s + the 16 probe quantities')
    fwd(MQ,mn,base_fe,(2,3,4),'minimal')
    fwd(np.c_[MQ,D],mn+dn,base_fe+PROBE_FE,(2,3,4,6),'minimal + probe')

    print("\n3. z62's 44 named quantities with and without the probe")
    for design,sub in (('all',np.ones(720,bool)),('compact',z60.ENDS)):
        Xn,names=named(cells,sub,P); fe=z60.total_cost(2,'4',sub,1,1,1,0)
        allridge(np.c_[Xn,ST],fe,f'{design}: ridge on the 44 + weight statistics')
        allridge(np.c_[Xn,ST,D],fe+PROBE_FE,f'{design}: ridge on the 44 + weight statistics + probe')
        e0=fwd(Xn,names,fe,(3,6,12),f'{design}')
        e1=fwd(np.c_[Xn,D],names+dn,fe+PROBE_FE,(3,6,12),f'{design} + probe')
        if design=='all':
            print("\n4. the 6-quantity model's residual (estimate - truth) against roles")
            for tag,e in (('without the probe',e0[6]),('with the probe',e1[6])):
                r=[np.corrcoef(e,RA[tr,j])[0,1] for j in range(RA.shape[1])]; o=np.argsort(-np.abs(r))[:4]
                print(f"  {tag:18s} RMS {100*np.sqrt((e**2).mean()):.3f}; "+", ".join(f"{RN[j]} {r[j]:+.2f}" for j in o),flush=True)
