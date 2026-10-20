"""Cheap deterministic features suggested by ARC's earlier work, added to (or replacing parts of) G(1,K):
  bal    the 20 weight statistics in the balanced gauge: W_o columns centred (adding a common vector to every logit
         row changes no argmax), then the positive diagonal rescaling D (an exact symmetry) that minimizes the total
         squared weight, i.e. every hidden unit's incoming norm (W_ih and W_hh row) equals its outgoing norm (W_hh
         column and centred W_o column).  Keeps the overall logit scale, removes the arbitrary per-unit scales.
  arc    ARC's standardization (each unit rescaled so |W_ih| = 1, W_o centred): quartiles of diag W_hh, log row and
         column norms of W_hh and log centred W_o column norms, and the fraction of positive input weights (13).
  tmpl   match to the blog's subcircuit (refs/algzoo_blog_post.html): a pair (a, b) like neurons (2, 4), which
         compute the running max and its last increment (in ARC's scaling w_a = 0, w_b = +1, W_hh[(a,b),(a,b)] =
         [[+1,+1],[-1,-1]], nothing else feeding a or b), and "leave-one-out max" units like 6 and 7 (w = -1, fed
         (+1, 0) or (+1, +1) by (a, b) and nothing else).  Every quantity is invariant under the rescaling symmetry.
         Features: the best pair's distance and its five parts, the number of units within 0.3 of a leave-one-out
         template, and the three best leave-one-out distances (10).
  cov    order-2 cumulant (covariance) propagation of the unconditional N(0, I) input through the network with the
         exact bivariate-Gaussian ReLU closure (z8.propagate 'biv'): the ten logit means, centred and divided by the
         RMS logit SD, the mean off-diagonal logit correlation, and the mean and SD over units of the final gate
         probability Phi(mean / SD) (13).  Cost about 160 FE per network (counted).
Each set is scored as in z55 (nested ridge / RBF readout, 8-fold cross-fit inside the training population; mystery,
zoo seeds and the two weight-noise sets from a model refitted on the whole training population).
Usage: python z56.py            (needs z55's label caches)
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
from z8 import propagate
import gfeat
G={}
P,_=load('pop1k/seed0.npz'); G['zoo']=(P,np.load('pop1k/reference.npz')['acc'][:1024])
P,_=load('pop_unif/b128_n1024_seed4.npz'); G['unif']=(P,np.load('pop_unif/b128_n1024_seed4.acc22.npy'))
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
G['zoo models']=([th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)],np.load('pop1k/reference.npz')['acc'][1024:])
for sg in (0.003,0.01):
    rng=np.random.default_rng(int(sg*10000)); z=Z[0]; e=lambda s_: rng.standard_normal((256,)+s_)
    W=[th.tensor(v,dtype=th.float32) for v in (z[0]*(1+sg*e(z[0].shape)),z[1]+sg*np.abs(z[1]).mean()*e(z[1].shape),z[2]*(1+sg*e(z[2].shape)))]
    G[f'noise {sg}']=(W,np.load(f'pop_unif/mystery_noise{sg}.acc22.npy'))
names=list(G); off=np.cumsum([0]+[len(G[k][1]) for k in names]); ix={k:np.arange(off[i],off[i+1]) for i,k in enumerate(names)}
Wall=[th.cat([G[k][0][i].float() for k in names]) for i in range(3)]; y=np.concatenate([G[k][1] for k in names]); pq=y*(1-y)
wi,Am,O=[w.numpy().astype(float) for w in Wall]; N=len(y)

def balanced(wi,Am,O,it=200):
    Oc=O-O.mean(1,keepdims=True); off=1-np.eye(16); D=np.ones_like(wi)
    for _ in range(it):
        Ar=D[:,:,None]*Am/D[:,None,:]
        inn=(D*wi)**2+((Ar*off)**2).sum(2); out=((Ar*off)**2).sum(1)+((Oc/D[:,None,:])**2).sum(1)
        D=D*(out/inn)**.25
    return D*wi, D[:,:,None]*Am/D[:,None,:], Oc/D[:,None,:]
def arc(wi,Am,O):
    D=1/np.abs(wi); A2=D[:,:,None]*Am/D[:,None,:]; O2=(O-O.mean(1,keepdims=True))/D[:,None,:]
    F=np.stack([np.diagonal(A2,axis1=1,axis2=2),np.log(np.linalg.norm(A2,axis=2)),np.log(np.linalg.norm(A2,axis=1)),np.log(np.linalg.norm(O2,axis=1))],2)
    return np.c_[np.percentile(F,[25,50,75],axis=1).transpose(1,0,2).reshape(len(wi),-1),(wi>0).mean(1)]
def template(wi,Am):
    n=len(wi); aw=np.abs(wi); r=np.arange(n); I=np.eye(16,dtype=bool)
    Aab=Am; Aba=Am.transpose(0,2,1); daa=np.diagonal(Am,axis1=1,axis2=2)
    Sa=((Am*aw[:,None,:])**2).sum(2)                                   # (n,a): sum_c (A_ac |w_c|)^2
    isoa=(Sa[:,:,None]-(daa*aw)[:,:,None]**2-(Aab*aw[:,None,:])**2)/(Aab*aw[:,None,:])**2
    isob=(Sa[:,None,:]-(Aba*aw[:,:,None])**2-(daa*aw)[:,None,:]**2)/aw[:,None,:]**2
    r1=wi[:,:,None]*Aba/wi[:,None,:]
    parts=np.stack([(daa[:,:,None]-1)**2+0*Aab,(daa[:,None,:]+1)**2+0*Aab,(Aab*Aba+1)**2,r1**2,isoa+isob],-1)
    d2=parts.sum(-1)+4*(wi[:,None,:]<=0)+4*(Aab<=0); d2[:,I]=np.inf; d2=np.nan_to_num(d2,nan=np.inf)
    best=d2.reshape(n,-1).argmin(1); a,b=best//16,best%16
    out=np.c_[np.log1p(np.minimum(d2[r,a,b],1e6)),np.log1p(np.minimum(np.abs(parts[r,a,b]),1e6))]
    Aca=Am[r,:,a]*Am[r,a,b][:,None]*aw[r,b][:,None]/aw; Acb=Am[r,:,b]*aw[r,b][:,None]/aw
    other=(((Am*aw[:,None,:])**2).sum(2)-(Am[r,:,a]*aw[r,a][:,None])**2-(Am[r,:,b]*aw[r,b][:,None])**2-(daa*aw)**2)/aw**2
    sc=np.minimum((Aca-1)**2+(Acb-1)**2,(Aca-1)**2+Acb**2)+daa**2+other+4*(wi>=0)
    sc[r,a]=np.inf; sc[r,b]=np.inf; s3=np.sort(sc,1)[:,:3]
    return np.c_[out,(sc<.3).sum(1),np.log1p(np.minimum(s3,1e6))], a, b, sc
def covprop(W):
    out=[]
    for c in range(0,len(W[0]),256):
        Wc=[w[c:c+256].to(gfeat.dev).float() for w in W]
        with th.no_grad():
            m,C=propagate(Wc[0],Wc[1],th.zeros(1,10,device=gfeat.dev),th.ones(1,10,device=gfeat.dev),'biv')
            m,C=m[:,0],C[:,0]; Lm=th.einsum('nj,nkj->nk',m,Wc[2]); LC=Wc[2]@C@Wc[2].transpose(1,2)
            sd=th.diagonal(LC,0,-2,-1).clamp_min(1e-12).sqrt(); rms=sd.pow(2).mean(1,keepdim=True).sqrt()
            corr=LC/(sd[:,:,None]*sd[:,None,:]); offd=(corr.sum((1,2))-10)/90
            g=th.special.ndtr(m/th.diagonal(C,0,-2,-1).clamp_min(1e-12).sqrt())
            out.append(th.cat([(Lm-Lm.mean(1,keepdim=True))/rms,offd[:,None],g.mean(1,keepdim=True),g.std(1,keepdim=True)],1).cpu().numpy())
    F=np.concatenate(out); return np.nan_to_num(F,nan=0.,posinf=0.,neginf=0.)
B={'raw':gfeat.stats(wi,Am,O),'bal':gfeat.stats(*balanced(wi,Am,O)),'arc':arc(wi,Am,O)}
T,ta,tb,tsc=template(wi,Am); B['tmpl']=T; B['cov']=covprop(Wall)
mz=ix['zoo models'][0]; print(f"template on the mystery model: best pair (a, b) = ({ta[mz]}, {tb[mz]}), distance {np.expm1(T[mz,0]):.3f}; "
      f"leave-one-out units within 0.3: {np.where(tsc[mz]<.3)[0].tolist()} (scores {np.round(np.sort(tsc[mz])[:4],3).tolist()})")
for k in ('zoo','unif'): print(f"  {k}: best-pair distance median {np.median(np.expm1(T[ix[k],0])):.2f}, share with a pair within 1.0: {(np.expm1(T[ix[k],0])<1).mean():.2f}, "
                           f"mean number of leave-one-out units {T[ix[k],6].mean():.2f}")
print("balanced gauge check: max relative change of logits on 4,096 inputs (64 networks):",end=' ')
x=th.randn(4096,10,dtype=th.float64); sub=np.arange(0,N,max(1,N//64))
def lg(w,a,o):
    w,a,o=[th.tensor(v[sub]) for v in (w,a,o)]; h=th.zeros(len(sub),4096,16,dtype=th.float64)
    for t in range(10): h=th.relu(h@a.transpose(1,2)+x[None,:,t,None]*w[:,None,:])
    L=h@o.transpose(1,2); return L-L.mean(-1,keepdim=True)
print(f"{((lg(*balanced(wi,Am,O))-lg(wi,Am,O)).abs().max()/lg(wi,Am,O).abs().max()).item():.1e}",flush=True)
COV_FE=(10*(2*16**3+136*20*12+136*30)+2*10*16*16)/2880
SETS={'G (raw weights)':['raw'],'balanced instead of raw':['bal'],'ARC gauge instead of raw':['arc'],'G + template':['raw','tmpl'],
      'G + covariance':['raw','cov'],'balanced + template + covariance':['bal','tmpl','cov'],'G + all three':['raw','bal','tmpl','cov']}
def line(lab,e,tot,i): r=np.sqrt((e**2).mean()); return f"{lab} {100*r:.3f} ({100*e.mean():+.3f}) {r**2/(pq[i].mean()/tot):5.2f}x"
for K in (1,4):
    F,net,C=gfeat.summary(Wall,1,K)
    print(f"\nK = {K}: RMS (mean) in points and MSE/sampling at total cost; zoo models = mystery, seed0-4")
    for tn in ('zoo','unif'):
        tr=ix[tn]; ot='unif' if tn=='zoo' else 'zoo'; fold=np.random.default_rng(0).permutation(len(tr))%8
        for lab,bl in SETS.items():
            X=np.c_[F,*[B[b] for b in bl]]; tot=gfeat.cost(net+COV_FE*('cov' in bl),C,1,X.shape[1],len(tr))
            cf=gfeat.crossfit(X,y,tr,fold); rest=np.setdiff1d(np.arange(N),tr); pr=np.full(N,np.nan); pr[rest]=gfeat.fit_predict(X,y,tr,rest)
            print(f"  train {tn:4s} {lab:34s} {X.shape[1]:3d} feats {tot:7.1f} FE | "+line(f"{tn} cross-fit",cf-y[tr],tot,tr)+" | "+line(ot,pr[ix[ot]]-y[ix[ot]],tot,ix[ot])
                  +" | "+" | ".join(line(k,pr[ix[k]]-y[ix[k]],tot,ix[k]) for k in ('noise 0.003','noise 0.01'))
                  +" | zoo "+" ".join(f"{v:+.2f}" for v in 100*(pr[ix['zoo models']]-y[ix['zoo models']])),flush=True)
