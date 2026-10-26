from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(2000000, seed=1); lg, hs = run(x); pm=lg.argmax(1); N=len(x)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1); ages=9-o
Lr=np.take_along_axis(lg,o,1)
D23=Lr[:,1]-Lr[:,2]; D21=Lr[:,1]-Lr[:,0]
tr=np.arange(N)<1500000; te=~tr
feats=np.c_[Xr[:,:5], x[:,9], np.maximum(x[:,9],0)]
def cellfit(D, keys, F):
    pred=np.zeros(N); A=np.c_[F,np.ones(N)]
    for k in np.unique(keys):
        s=keys==k; st=s&tr
        if st.sum()<4*A.shape[1]: pred[s]=0; continue
        c,*_=np.linalg.lstsq(A[st],D[st],rcond=None); pred[s]=A[s]@c
    return pred
for nm,keys in [("aS,a3",ages[:,1]*10+ages[:,2]),("aM,aS,a3",ages[:,0]*100+ages[:,1]*10+ages[:,2])]:
    p=cellfit(D23,keys,feats)
    r2=1-((p-D23)[te]**2).mean()/D23[te].var()
    near=te&(np.abs(Xr[:,1]-Xr[:,2])<0.2)
    print(f"D23 cells {nm}: R2 {r2:.3f}; sign agreement all {((p>0)==(D23>0))[te].mean():.4f}, near ties {((p>0)==(D23>0))[near].mean():.4f}")
for nm,keys in [("aM,aS",ages[:,0]*10+ages[:,1]),("aM,aS,a3",ages[:,0]*100+ages[:,1]*10+ages[:,2])]:
    p=cellfit(D21,keys,feats)
    r2=1-((p-D21)[te]**2).mean()/D21[te].var()
    print(f"D21 cells {nm}: R2 {r2:.3f}; sign agreement {((p>0)==(D21>0))[te].mean():.4f}")
