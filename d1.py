import numpy as np, json
from seeds_clocks import load, run, T
x=np.random.default_rng(7).standard_normal((200000,T)); N=len(x)
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]; g=srt[:,1]-srt[:,2]
tr=np.arange(N)<150000; te=~tr; easy=te&(g>0.3)
def cellfit(L,keys,F):
    P=np.zeros_like(L)
    for k in np.unique(keys):
        s=keys==k; c,*_=np.linalg.lstsq(F[s&tr],L[s&tr],rcond=None); P[s]=F[s]@c
    return P
def metrics(path):
    Wih,Whh,Wo=load(path); hs=run(x,Wih,Whh); h=hs[:,-1]; L=h@Wo.T; pm=L.argmax(1); L=L-L.mean(1,keepdims=True)
    err=pm!=o[:,1]
    PB=cellfit(L,aM,x); PB[np.arange(N),o[:,0]]=-1e9
    PA=cellfit(L,aM*10+aS,np.c_[M,S,x[:,9],np.ones(N)])
    # max-clock: centroids of h/M by ageM (2..9), top-2 PCA plane rotation
    k=M>0.3; C=np.stack([(h[k]/M[k,None])[aM[k]==a].mean(0) for a in range(2,10)]); C=C-C.mean(0)
    U,s_,Vt=np.linalg.svd(C,full_matrices=False); P=C@Vt[:2].T; st=np.abs(np.degrees(np.diff(np.unwrap(np.arctan2(P[:,1],P[:,0])))))
    return dict(acc=1-err.mean(), tie=err[g<.05].mean(), floor=err[g>.5].mean(),
                B=(PB.argmax(1)==pm)[easy].mean(), A=(PA.argmax(1)==pm)[te].mean(),
                clock=float(np.median(st)), clockvar=float((s_[:2]**2).sum()/(s_**2).sum()+0), clockjit=float(st.std()))
res={}
for s in range(5):
    print(f"\n=== seed {s}\n {'train seqs':>10s}   acc   err@tie err@gap>.5 | memory(B) selection(A) | max clock: step ± jitter, plane var")
    for e in [0]+list(range(20,31)):
        m=metrics(f"ckpt/s{s}_e{e}.npz"); res[f"{s}_{e}"]=m
        print(f" {('2^%d'%e) if e else '0':>10s} {m['acc']:.3f}  {m['tie']:.3f}   {m['floor']:.3f}    |  {m['B']:.3f}     {m['A']:.3f}    | {m['clock']:6.1f}° ± {m['clockjit']:5.1f}, {m['clockvar']:.2f}")
json.dump(res,open("ckpt_metrics.json","w"))
