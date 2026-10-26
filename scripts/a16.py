from common import *
x = sample(2000000); lg, hs = run(x); pm=lg.argmax(1)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1); ages=9-o
h=hs[:,9]; tr=np.arange(len(x))<1500000; te=~tr
def fit(keys, vals, name):
    # keys: (N,) int cell ids ; vals: (N,k) regressors
    pred=np.zeros_like(h); u=np.unique(keys)
    A=np.c_[vals,np.ones(len(x))]
    for k in u:
        s=keys==k; st=s&tr
        if st.sum()<A.shape[1]*3: pred[s]=h[tr].mean(0); continue
        c,*_=np.linalg.lstsq(A[st],h[st],rcond=None); pred[s]=A[s]@c
    p=(pred@Wo.T).argmax(1)
    print(f"{name}: cells={len(u)}  test agreement with model {(p==pm)[te].mean():.4f}  surrogate acc {(p==o[:,1])[te].mean():.4f}")
x9=x[:,9:10]
fit(ages[:,0]*10+ages[:,1], np.c_[Xr[:,:2],x9], "cells(aM,aS) lin[M,S,x9]")
fit(ages[:,0]*10+ages[:,1], np.c_[Xr[:,:3],x9], "cells(aM,aS) lin[M,S,R3,x9]")
fit(ages[:,0]*100+ages[:,1]*10+ages[:,2], np.c_[Xr[:,:3],x9], "cells(aM,aS,a3) lin[M,S,R3,x9]")
fit(ages[:,0]*100+ages[:,1]*10+ages[:,2], np.c_[Xr[:,:4],x9], "cells(aM,aS,a3) lin[M,S,R3,R4,x9]")
print("model acc", (pm==o[:,1]).mean())
