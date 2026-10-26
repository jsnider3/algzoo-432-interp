"""Bucket neurons into roles across the population (z2 features) and test whether a network's mix of roles
predicts its accuracy and its error by failure region.

Roles, by REPORT's definitions (2.2, 7.9, 7.14):
  dead          active < 2% of steps
  always-on     fires after >= 95% of "neither" steps and on >= 90% of new maxes (memory channel)
  event-ptr     write pointer firing on the event (seed 2 n10, seed 3 n9): on >= 70% of new maxes and >= 50% of
                new 2nd maxes, <= 30% of "neither" (7.9's event selectivity: 0.98 / 0.90 / 0.19 for seed 3 n9)
  late-ptr      one-step-late write pointer (mystery n5): the same rates one step after the event
  max-det       new-max detector (mystery n4): on >= 70% of new maxes, < 50% of new 2nd maxes, <= 10% of "neither"
  late-marker   fires one step after a new max only (mystery n9)
  comparator    negative input weight (|w| > 2), fires on >= 70% of "neither", <= 40% of new maxes: ReLU(theta - x)
  other         everything else (clock-like, partial)
Checked against the zoo models: mystery n4/n5/n9, seed 2 n10/n15 and seed 3 n9 get the roles REPORT gives them.
Per network, "selectivity" = max over neurons of min(rate on new max, rate on new 2nd max) - rate on "neither",
on the event or one step after it, whichever is larger (a continuous version of the pointer roles).
Unsupervised clusters (k-means on standardized features) are reported next to the rules.

Prediction: 5-fold cross-validated, from (i) role counts, (ii) role counts + per-role mean features, (iii) cluster
counts + per-cluster mean features, and as generic baselines (iv) each feature sorted over neurons and (v) weights
only (sorted input weights, self-loops, |eigenvalues| of W_hh, readout norms). A predictor with RMSE r is worth
p(1-p)/r^2 samples.

Usage: python scripts/z3.py pop1k/snap_2e28.roles.npz [pop1k/snap_2e28.npz]
"""
import sys, numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROLES=["dead","always-on","event-ptr","late-ptr","max-det","late-marker","comparator","other"]

def roles(F, nm):
    f={k:F[...,i] for i,k in enumerate(nm)}; r=np.full(F.shape[:2],7)
    r[(f["w_in"]<-2)&(f["on_neither"]>=.7)&(f["on_newmax"]<=.4)]=6
    r[(f["after_newmax"]>=.7)&(f["after_new2nd"]<.3)&(f["after_neither"]<=.1)]=5
    r[(f["on_newmax"]>=.7)&(f["on_new2nd"]<.5)&(f["on_neither"]<=.1)]=4
    r[(f["after_newmax"]>=.7)&(f["after_new2nd"]>=.5)&(f["after_neither"]<=.3)]=3
    r[(f["on_newmax"]>=.7)&(f["on_new2nd"]>=.5)&(f["on_neither"]<=.3)]=2
    r[(f["after_neither"]>=.95)&(f["on_newmax"]>=.9)]=1
    r[f["rate"]<.02]=0
    return r

def selectivity(F, nm):
    f={k:F[...,i] for i,k in enumerate(nm)}
    on=np.minimum(f["on_newmax"],f["on_new2nd"])-f["on_neither"]; late=np.minimum(f["after_newmax"],f["after_new2nd"])-f["after_neither"]
    return np.maximum(on.max(1),late.max(1))

def bag(lab, F, K, with_means=True):
    """per network: count of neurons per label, and (optionally) mean feature vector of each label's neurons"""
    N,H,D=F.shape; cnt=np.stack([(lab==k).sum(1) for k in range(K)],1); cols=[cnt]
    if with_means:
        for k in range(K):
            m=(lab==k)[...,None]; cols.append(np.where(cnt[:,k:k+1]>0,(F*m).sum(1)/np.maximum(cnt[:,k:k+1],1),0))
    return np.concatenate(cols,1).astype(float)

def cv_rmse(X, y, seed=0):
    kf=KFold(5,shuffle=True,random_state=seed); out={}
    for nm,mdl in [("ridge",make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13)))),
                   ("gbm",HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=15,random_state=0))]:
        out[nm]=np.sqrt(np.mean((cross_val_predict(mdl,X,y,cv=kf)-y)**2))
    return out

if __name__=="__main__":
    z=np.load(sys.argv[1]); F,nm,acc=z["feats"],list(z["names"]),z["acc"]; N,H,D=F.shape
    E,n=z["strata_err"],z["strata_n"]
    p=acc.mean(); print(f"{N} networks; accuracy mean {p:.4f}, SD across networks {acc.std()*100:.2f} points")
    r=roles(F,nm); C=np.stack([(r==k).sum(1) for k in range(len(ROLES))],1)
    print("\nrole counts per network (mean; % of networks with >= 1):")
    for k,rn in enumerate(ROLES): print(f"  {rn:11s} {C[:,k].mean():5.2f}  {100*(C[:,k]>0).mean():5.1f}%")
    sel=(C[:,2]+C[:,3])>0; sc=selectivity(F,nm)
    print(f"\nnetworks with a pointer (event-ptr or late-ptr): {sel.mean()*100:.1f}%; accuracy {acc[sel].mean():.4f} vs {acc[~sel].mean():.4f} without")
    q=np.digitize(sc,[.2,.4,.6,.8]); print("  by selectivity score: "+"  ".join(f"{lo}: {acc[q==j].mean():.3f} (n={np.sum(q==j)})" for j,lo in enumerate(["<.2",".2-.4",".4-.6",".6-.8",">.8"]) if np.sum(q==j)))
    for k in (1,2,3,6):
        print(f"  by # {ROLES[k]:10s}: "+"  ".join(f"{c}: {acc[C[:,k]==c].mean():.3f} (n={np.sum(C[:,k]==c)})" for c in range(0,8) if np.sum(C[:,k]==c)>=10))
    # unsupervised clusters over all neurons
    Fz=F.reshape(-1,D).copy(); Fz=(Fz-Fz.mean(0))/Fz.std(0).clip(1e-9); K=12
    km=KMeans(K,n_init=4,random_state=0).fit(Fz); cl=km.labels_.reshape(N,H)
    print(f"\n{K} k-means clusters: size, share in each rule role, and centroid of key features")
    key=["w_in","rate","on_newmax","on_new2nd","on_neither","after_newmax","after_new2nd","after_neither","thr_R2","eta_ageM","eta_ageS"]
    print("  cl  size  "+" ".join(f"{k[:9]:>9s}" for k in key)+"   dominant role")
    for k in np.argsort(-np.bincount(km.labels_)):
        m=cl==k; cen=F[m].mean(0); rr=np.bincount(r[m],minlength=len(ROLES))/m.sum()
        print(f"  {k:2d} {m.sum():5d}  "+" ".join(f"{cen[nm.index(q)]:9.2f}" for q in key)+f"   {ROLES[rr.argmax()]} {rr.max()*100:.0f}%")
    # prediction of accuracy
    designs={"selectivity score only":sc[:,None],"role counts":bag(r,F,len(ROLES),False),"role counts + role means":bag(r,F,len(ROLES)),
             "role counts + means + selectivity":np.c_[bag(r,F,len(ROLES)),sc],
             f"{K} cluster counts + means":bag(cl,F,K),"sorted neuron features":np.sort(F,1).reshape(N,-1)}
    if len(sys.argv)>2:
        w=np.load(sys.argv[2]); s=max(int(k[1:-4]) for k in w.files if k.startswith("s") and k.endswith("_Wih")); Wi,Wh,Wo=(w[f"s{s}_{k}"] for k in ("Wih","Whh","Wo"))
        designs["weights only (sorted)"]=np.concatenate([np.sort(Wi,1),np.sort(np.diagonal(Wh,0,1,2),1),np.sort(np.abs(np.linalg.eigvals(Wh)),1),
                                                       np.sort(np.linalg.norm(Wo,axis=1),1)],1)
    print(f"\npredicting accuracy (5-fold CV RMSE in points; worth p(1-p)/RMSE^2 samples at p = {p:.3f}); predict-the-mean: {acc.std()*100:.2f}")
    for dn,X in designs.items():
        res=cv_rmse(X,acc); print(f"  {dn:28s} "+"   ".join(f"{k} {v*100:.2f} (~{p*(1-p)/v**2:,.0f} samples)" for k,v in res.items()))
    # error by failure region: which roles go with which errors
    lab=np.arange(len(n)); a,b=(lab-4)//8,(lab-4)%8; pool=n.sum()             # z1.task_strata: a, b bin indices
    reg={"S<0 (blind spot)":lab<4,"M-S near tie (a<.05)":(lab>=4)&(a<=2),"S-R near tie (b<.03)":(lab>=4)&(a>2)&(b<=2),
         "middle":(lab>=4)&(a>2)&(b>2)&~((a>=4)&(b>=5)),"clean (a,b >= .1)":(lab>=4)&(a>=4)&(b>=5)}
    X=np.c_[bag(r,F,len(ROLES)),sc]; kf=KFold(5,shuffle=True,random_state=0)
    print(f"\nerror by region: mass, error rate, contribution to error and its SD across networks (points), CV R^2 from roles,"
          f"\n  and correlation of the contribution with role counts ({', '.join(ROLES[1:7])})")
    for rn,m in reg.items():
        c=E[:,m].sum(1)/pool; mass=n[m].sum()/pool
        pr=cross_val_predict(HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=15,random_state=0),X,c,cv=kf)
        print(f"  {rn:22s} {mass*100:5.1f}% {c.mean()/mass*100:5.1f}% {c.mean()*100:5.2f} {c.std()*100:5.2f} {1-np.mean((pr-c)**2)/c.var():5.2f} | "
              +" ".join(f"{np.corrcoef(C[:,k],c)[0,1]:+.2f}" for k in range(1,7)))
    np.savez(sys.argv[1].replace(".roles.npz",".buckets.npz"),roles=r,clusters=cl,centroids=km.cluster_centers_)
