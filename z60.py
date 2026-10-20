"""Independent check of ChatGPT round 7's cheaper G variants (chatgpt2/round7/answer.md), rebuilt from its
description on the z59 / z46 cell caches and fitted with gfeat's nested ridge / RBF readout (full kernel, 1,024
supports; folds by a random permutation, inner folds keyed on the 1,024 population indices only):
  rows    'both' = MF4 + MF3 rows (as G), '4' = MF4 rows only;
  subset  'all' = the 720 (M, S, R) arrangements, 'ends' = the 180 with R at the first or last free position;
  P       two signed input pulses: h_0 = ReLU(+-W_ih), h_{a+1} = ReLU(W_hh h_a), L_a = W_o h_a for ages a = 0..9 (the
          network on the sequences +-e_{9-a}); per sign the mean margin of output 9-a over the best other output and
          its mean over ages 5-9, both divided by the RMS of the centred logits; the share of ages where output 9-a
          wins; mean and SD over ages of the change in log RMS centred logit (10 features, charged 6 FE);
  H       per S slice, means of MF4's bit and a/(.1+|a|) over six arrangement groups: M last, S last, M right after
          S, M after S, R after S, |M - S| >= 5 (12 K features).
Cost: the prefix trie of exactly the rows used (gfeat's formula), 40 (both) or 22 (MF4 only) scalar ops per cell,
2 per cell and group for H, 6 FE for P, the 5,000-op allowance and the RBF readout over 1,024 supports.
External groups (mystery recipe): 64 more batch-128 networks, 64 batch-64, the mystery model; (AlgZoo recipe): zoo
seeds 0-4, from a model refitted on all 1,024.
Usage: python z60.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat
ARR=gfeat.ARR; MASKS=[ARR[:,0]==9,ARR[:,1]==9,ARR[:,0]==ARR[:,1]+1,ARR[:,0]>ARR[:,1],ARR[:,2]>ARR[:,1],abs(ARR[:,0]-ARR[:,1])>=5]
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; ENDS=np.array([f.index(r[2]) in (0,7) for f,r in zip(free,ARR)])

def pulses(wi,Am,O):
    h=np.maximum(np.stack([wi,-wi],2),0); L=[]
    for a in range(10):
        L.append(O@h)
        if a<9: h=np.maximum(Am@h,0)
    L=np.stack(L,1); c=L-L.mean(2,keepdims=True); amp=np.maximum(np.sqrt((c*c).mean(2)),1e-12)   # (N,age,10,2), (N,age,2)
    t=9-np.arange(10); mine=L[:,np.arange(10),t]; oth=L.copy(); oth[:,np.arange(10),t]=-np.inf
    mg=(mine-oth.max(2))/amp; ok=(L.argmax(2)==t[None,:,None]); gr=np.diff(np.log(amp),axis=1)
    return np.concatenate([mg.mean(1),ok.mean(1),mg[:,5:].mean(1),gr.mean(1),gr.std(1)],1)

def row_cost(K,rows,sub):
    x=gfeat.rows(1,K).reshape(720,K,2,10)[sub][:,:,[0] if rows=='4' else [0,1]].reshape(-1,10)
    lev,_,nv=gfeat.trie(x); n=[len(p) for p,_ in lev]
    return (256*sum(n[:-1])+16*nv+160*n[-1]+16*sum(n[1:])+16*sum(n)+20*n[-1])/2880

def features(z,K,rows,sub,P,H,st):
    i4,i3,m4,m3=[z[k].reshape(len(st),720,K)[:,sub].astype(float) for k in ('I4','I3','m4','m3')]
    s=np.sqrt((m4**2).mean((1,2)))[:,None,None]; ch=[]
    for I,m in ((i4,m4),)+(((i3,m3),) if rows=='both' else ()):
        a=m/s; ch+=[I,a,abs(a),a/(.1+abs(a)),a/(.03+abs(a))]
    f=np.stack(ch,-1); X=[st,s[:,0],np.log(s[:,0]),m4.mean((1,2))[:,None]]+([m3.mean((1,2))[:,None]] if rows=='both' else [])
    X+=[f.mean((1,2)),f.mean(1).reshape(len(st),-1)]
    if P is not None: X.append(P)
    if H:
        v=np.stack([i4,m4/s/(.1+abs(m4/s))],-1)
        X+=[v[:,msk[sub]].mean(1).reshape(len(st),-1) for msk in MASKS]
    return np.concatenate(X,1)

def total_cost(K,rows,sub,P,H,d,n_sup=1024):
    ops=(40 if rows=='both' else 22)*int(sub.sum())*K+(2*K*sum(int(m[sub].sum()) for m in MASKS) if H else 0)
    return row_cost(K,rows,sub)+6*P+(ops+5000+2*n_sup*d+34*n_sup+2*d+3)/2880

def run(tag,z,W,y,ext,K,rows,sub,P,H):
    st=gfeat.stats(*W); X=features(z,K,rows,sub,pulses(*W) if P else None,H,st); tot=total_cost(K,rows,sub,P,H,X.shape[1])
    pop=np.arange(1024); yp=y[:1024]; cf=gfeat.crossfit(X[:1024],yp,pop,np.random.default_rng(0).permutation(1024)%8)
    e=cf-yp; r=np.sqrt((e**2).mean()); out=[f"{tag:34s} {X.shape[1]:3d} feats {tot:7.1f} FE | cross-fit RMS {100*r:.3f} mean {100*e.mean():+.3f} MSE/s {r**2/((yp*(1-yp)).mean()/tot):.3f}"]
    pr=gfeat.fit_predict(X,y,pop,np.arange(1024,len(y)))
    for k,ix in ext.items():
        e=pr[ix-1024]-y[ix]; out.append(f"{k} {100*np.sqrt((e**2).mean()):.3f}/{100*e.mean():+.3f}" if len(ix)>1 else f"{k} {100*e[0]:+.3f}")
    print(" | ".join(out),flush=True)

ALL=np.ones(720,bool)
SETS=[('G(1,1)',1,'both',ALL,0,0),('G(1,1) + P',1,'both',ALL,1,0),('G(1,2)',2,'both',ALL,0,0),('G(1,4)',4,'both',ALL,0,0),
      ('MF4 K=2',2,'4',ALL,0,0),('MF4 K=2 + P',2,'4',ALL,1,0),('MF4 K=2 + P + H',2,'4',ALL,1,1),('MF4 K=4 + P + H',4,'4',ALL,1,1),
      ('ends K=1 both + P',1,'both',ENDS,1,0),('ends K=1 MF4 + P',1,'4',ENDS,1,0),('ends K=2 MF4 + P',2,'4',ENDS,1,0)]
if __name__=='__main__':
    for recipe in ('unif','zoo'):
        print(f"\n{recipe} recipe",flush=True)
        for tag,K,rows,sub,P,H in SETS:
            if recipe=='unif':
                z=np.load(f'pop_unif/cells_k{K}.npz'); y=z['acc']; g=z['group']
                ext={'b128':np.where(g=='b128_seed0')[0],'b64':np.where(g=='b64_seed5')[0],'mystery':np.where(g=='mystery')[0]}
                Ws=[load(f)[0] for f in ('pop_unif/b128_n1024_seed4.npz','pop_unif/b128_seed0.npz','pop_unif/b64_seed5.npz')]
                mz=zload('weights.npz'); W=[np.concatenate([w[i].numpy().astype(float) for w in Ws]+[mz[i][None].astype(float)]) for i in range(3)]
            else:
                z=np.load(f'pop1k/seed0_z46_cells_b1_k{K}.npz'); y=np.load('pop1k/reference.npz')['acc']
                P0,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
                W=[np.concatenate([P0[i].numpy().astype(float),np.stack([q[i] for q in Z]).astype(float)]) for i in range(3)]
                ext={f'seed{i}':np.array([1025+i]) for i in range(5)}
            run(tag,z,W,y,ext,K,rows,sub,P,H)
