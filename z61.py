"""Confirmation of round 7 on networks nobody has seen: 512 new mystery-recipe networks (batch 128, seed 6,
pop_unif/b128_n512_seed6.npz, trained after round 7 was answered), labelled on an independent bank of 2^22 N(0,1)
inputs (seed 61; the training labels use seed 55, and the new networks are also labelled on that bank to measure the
bank's effect).  Every estimator below was fixed before the new networks finished training; each is fitted on the
1,024 training networks only (pop_unif/b128_n1024_seed4.npz) and applied unchanged.
  ChatGPT round 7's compiled models (chatgpt2/round7/runs/*.model.npz, made by its round7_estimator.py on our
  machine), queried with round7_estimator.predict_weights on each new network's weights; costs from its ledger.
  z60's rebuilds (gfeat readout: hyperparameters by 8-fold CV on the 1,024, refit on all of them); costs by
  z60.total_cost with 1,024 supports.
Reports, per estimator: total FE, RMS and mean error on the new networks, MSE / sampling at that cost with a
bootstrap 95% interval over networks (2,000 resamples), and the same against the seed-55 labels.
Usage: python z61.py
"""
import os, sys, json, numpy as np, torch as th
from z2 import load
from batch_train import accuracy
import gfeat, z60
sys.path.insert(0,'chatgpt2/round7'); import round7_estimator as r7
NEW='pop_unif/b128_n512_seed6'
def labels(W,path,seed,log2=22):
    if os.path.exists(path): return np.load(path)
    g=th.Generator(device=gfeat.dev).manual_seed(seed); x=th.randn(2**log2,10,generator=g,device=gfeat.dev); P=[w.to(gfeat.dev).float() for w in W]
    a=np.concatenate([accuracy([p[i:i+256] for p in P],x) for i in range(0,len(P[0]),256)]); np.save(path,a); return a
Wn,_=load(NEW+'.npz'); yn=labels(Wn,NEW+'.acc22s61.npy',61); y55=labels(Wn,NEW+'.acc22.npy',55)
Wt,_=load('pop_unif/b128_n1024_seed4.npz'); yt=np.load('pop_unif/b128_n1024_seed4.acc22.npy'); ym=0.95304775
print(f"new networks: {len(yn)}, accuracy {100*yn.mean():.2f}% SD {100*yn.std():.2f} (training 1,024: {100*yt.mean():.2f}% SD {100*yt.std():.2f}); "
      f"mystery percentile {100*(yn<ym).mean():.1f}")
print(f"label banks (seed 61 - seed 55): mean {100*(yn-y55).mean():+.4f} points, RMS {100*np.sqrt(((yn-y55)**2).mean()):.4f}",flush=True)
rng=np.random.default_rng(61); BOOT=rng.integers(0,len(yn),(2000,len(yn)))
def line(tag,tot,pr):
    out=[f"{tag:40s} {tot:7.1f} FE"]
    for nm,y in (('bank 61',yn),('bank 55',y55)):
        e=pr-y; r=np.sqrt((e**2).mean()); pq=(y*(1-y)).mean(); q=(e[BOOT]**2).mean(1)/((y[BOOT]*(1-y[BOOT])).mean(1)/tot)
        out.append(f"{nm}: RMS {100*r:.3f} mean {100*e.mean():+.3f} MSE/s {r**2/(pq/tot):.3f} [{np.percentile(q,2.5):.3f}, {np.percentile(q,97.5):.3f}]")
    print(" | ".join(out),flush=True)

print("\nChatGPT round 7's compiled models")
wn=[w.numpy().astype(float) for w in Wn]
for nm in ('G1','mf4_k2_nopulse','compact','mf4_k2_small','mf4_k2','mf4_k4'):
    m=dict(np.load(f'chatgpt2/round7/runs/{nm}.model.npz',allow_pickle=False)); tot=json.load(open(f'chatgpt2/round7/runs/{nm}.json'))['costs']['total_fe']
    pr=np.array([r7.predict_weights(m,[w[i] for w in wn]) for i in range(len(yn))]); line(nm,tot,pr)

print("\nz60 rebuilds (gfeat readout)")
Wa=[th.cat([a.float(),b.float()]) for a,b in zip(Wt,Wn)]; Wa_np=[w.numpy().astype(float) for w in Wa]; st=gfeat.stats(*Wa_np); P=z60.pulses(*Wa_np)
ya=np.r_[yt,np.full(len(yn),np.nan)]; tr=np.arange(1024); te=np.arange(1024,len(ya)); cells={}
for tag,K,rows,sub,p,H in z60.SETS:
    if K not in cells:
        _,_,_,c=gfeat.summary(Wa,1,K,cells=True); cells[K]=dict(zip(('I4','I3','m4','m3'),c))
    X=z60.features(cells[K],K,rows,sub,P if p else None,H,st); tot=z60.total_cost(K,rows,sub,p,H,X.shape[1])
    line(tag,tot,gfeat.fit_predict(X,ya,tr,te))
