from common import *
from sklearn.tree import DecisionTreeClassifier
x = sample(2000000, seed=3); lg, hs = run(x); pm=lg.argmax(1); N=len(x)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1); ages=9-o
rank_pred=np.argsort(o,1)[np.arange(N),pm]; y=np.minimum(rank_pred,3)  # 0 max,1 second,2 third,3 other
M=Xr[:,[0]]
F=np.c_[ages[:,:5], Xr[:,:6]/M, (Xr[:,0]-Xr[:,1])/M[:,0], (Xr[:,1]-Xr[:,2])/M[:,0], (Xr[:,2]-Xr[:,3])/M[:,0], x[:,9]/M[:,0], (ages[:,1]>ages[:,0]), (ages[:,2]>ages[:,1])]
tr=slice(0,1500000); te=slice(1500000,None)
for d in [4,8,12,16,24]:
    t=DecisionTreeClassifier(max_depth=d, min_samples_leaf=50).fit(F[tr],y[tr])
    pr=t.predict(F[te]); 
    # map back to position
    pos=o[np.arange(N)[te], np.minimum(pr,9)]
    agree=(pos==pm[te]).mean()
    print(f"depth {d}: leaves {t.get_n_leaves()}  agreement {agree:.4f}  (baseline {(o[te,1]==pm[te]).mean():.4f}, ceiling top3 {(rank_pred[te]<3).mean():.4f})")
