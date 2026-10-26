"""Does training the readout on networks made with the mystery model's own recipe fix its transfer to the mystery
model?  G(1,K) (gfeat.py: ChatGPT round 6's cells + logit scale + 20 raw weight statistics, nested ridge / RBF
readout) trained on
  zoo   the 1,024 zoo-recipe networks (pop1k/seed0.npz; labels pop1k/reference.npz),
  unif  the 1,024 mystery-recipe networks (pop_unif/b128_n1024_seed4.npz: Unif(-1,1) inputs, Adam 0.005, weight
        decay 1e-14, batch 128; labels = accuracy on 2^22 N(0,1) inputs),
  both  the union,
and scored on: each population (8-fold cross-fitted when it is in the training set), the mystery model and zoo seeds
0-4, ARC's weight-noise protocol (256 copies of the mystery model per level, i8.py's noise: relative N(0, sigma^2) on
W_ih and W_o, additive N(0, (sigma mean|W_hh|)^2) on W_hh; sigma = 0.3% and 1%; labels on 2^22 inputs), and the
64-network mystery-recipe groups pop_unif/b<batch>_seed<s>.npz (other seeds; pop_unif/z55.txt predates the batch-64
group, which z58 scores).
MSE / sampling at the estimator's total cost (gfeat.cost; the readout cost grows with the number of supports).
Usage: python scripts/z55.py
"""
import glob, os, numpy as np, torch as th
from batch_train import accuracy
from z2 import load
from seeds_clocks import load as zload
import gfeat
dev=gfeat.dev
def labels(W,path,log2=22,seed=55):
    if os.path.exists(path): return np.load(path)
    g=th.Generator(device=dev).manual_seed(seed); x=th.randn(2**log2,10,generator=g,device=dev); P=[w.to(dev).float() for w in W]
    a=np.concatenate([accuracy([p[i:i+256] for p in P],x) for i in range(0,len(P[0]),256)]); np.save(path,a); return a
ref=np.load('pop1k/reference.npz')['acc']
G={}                                                     # name -> (weights, labels)
P,_=load('pop1k/seed0.npz'); G['zoo']=(P,ref[:1024])
P,_=load('pop_unif/b128_n1024_seed4.npz'); G['unif']=(P,labels(P,'pop_unif/b128_n1024_seed4.acc22.npy'))
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
G['zoo models']=([th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)],ref[1024:])
for sg in (0.003,0.01):
    rng=np.random.default_rng(int(sg*10000)); z=Z[0]; e=lambda s_: rng.standard_normal((256,)+s_)
    W=[th.tensor(v,dtype=th.float32) for v in (z[0]*(1+sg*e(z[0].shape)),z[1]+sg*np.abs(z[1]).mean()*e(z[1].shape),z[2]*(1+sg*e(z[2].shape)))]
    G[f'noise {sg}']=(W,labels(W,f'pop_unif/mystery_noise{sg}.acc22.npy'))
for f in sorted([f for f in glob.glob('pop_unif/b*_seed[0-9].npz') if 'n1024' not in f],key=lambda s:int(s.split('/b')[1].split('_')[0])):
    P,_=load(f); G[f"unif {f.split('/')[1][:-4]}"]=(P,labels(P,f[:-4]+'.acc22.npy'))
names=list(G); off=np.cumsum([0]+[len(G[k][1]) for k in names]); ix={k:np.arange(off[i],off[i+1]) for i,k in enumerate(names)}
Wall=[th.cat([G[k][0][i].float() for k in names]) for i in range(3)]; y=np.concatenate([G[k][1] for k in names]); pq=y*(1-y)
WS=gfeat.stats(*[w.numpy().astype(float) for w in Wall])
for k in names: print(f"{k:22s} n {len(ix[k]):5d}  accuracy mean {100*y[ix[k]].mean():.2f}%  SD {100*y[ix[k]].std():.2f}  range {100*y[ix[k]].min():.2f}-{100*y[ix[k]].max():.2f}")
print(f"mystery accuracy {100*y[ix['zoo models'][0]]:.3f}%",flush=True)
TRAIN={'zoo':ix['zoo'],'unif':ix['unif'],'both':np.r_[ix['zoo'],ix['unif']]}
def line(lab,e,tot,k):
    i=ix[k] if k in ix else k; r=np.sqrt((e**2).mean()); return f"{lab} RMS {100*r:.3f} mean {100*e.mean():+.3f} MSE/s {r**2/(pq[i].mean()/tot):6.2f}"
for K in (1,2,4):
    F,net,C=gfeat.summary(Wall,1,K); X=np.c_[WS,F]
    print(f"\nG(1,{K}): {X.shape[1]} features, network rows {net:.1f} FE")
    for tn,tr in TRAIN.items():
        tot=gfeat.cost(net,C,1,X.shape[1],len(tr)); out=[f"  train {tn:4s} ({tot:7.1f} FE):"]
        cf=gfeat.crossfit(X,y,tr,np.random.default_rng(0).permutation(len(tr))%8)
        for k in ('zoo','unif'):
            if k in tn or tn=='both': m=np.isin(tr,ix[k]); out.append(line(f"{k} (cross-fit)",cf[m]-y[tr[m]],tot,k))
        rest=np.setdiff1d(np.arange(len(y)),tr); pr=np.full(len(y),np.nan); pr[rest]=gfeat.fit_predict(X,y,tr,rest)
        for k in ('zoo','unif'):
            if k not in tn and tn!='both': out.append(line(k,pr[ix[k]]-y[ix[k]],tot,k))
        print(" | ".join(out),flush=True)
        print("     zoo models (mystery, seed0-4): "+" ".join(f"{v:+.3f}" for v in 100*(pr[ix['zoo models']]-y[ix['zoo models']]))+"   "
              +" | ".join(line(k,pr[ix[k]]-y[ix[k]],tot,k) for k in names if k.startswith('noise') or k.startswith('unif b')),flush=True)
