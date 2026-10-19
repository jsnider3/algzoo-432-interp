"""Informative inputs: which single inputs best reveal a network's accuracy, and which circuit features decide whether
a network gets them right (needs z4's error masks and z2's per-neuron features).

Per input x, over the population: u(x) = share of networks wrong on x, and
  informativeness I(x) = Cov(f(x), acc)^2 / Var(f(x)) = corr(f(x), acc)^2 * Var(acc),
the accuracy variance a linear fit on that one outcome would explain. Inputs are split into training and held-out
networks (half each); ranking and greedy selection use the training half only.
  1. where the most informative inputs fall (z3 region, handoff/direct write, steps since the last event)
  2. greedy forward selection of K inputs (each step adds the input that most reduces the training residual);
     held-out RMSE of a ridge fit on those K outcomes, against K random inputs (plain sampling, affine-calibrated)
  3. for the top inputs, how well circuit features (role counts + per-role means, z3) predict which networks fail
Usage: python z7.py pop1k/snap_2e28.npz   (reads pop1k/z4_snap_2e28.npz and pop1k/snap_2e28.roles.npz)
"""
import os, sys, numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from z3 import roles, bag, ROLES
from z4 import input_vars

def mech_label(V):
    ev=np.where(V["aS"]>V["aM"],"handoff","direct"); k=np.minimum(np.minimum(V["aM"],V["aS"]),3)
    reg=np.array(["S<0","M-S tie","S-R tie","middle","clean"])[V["reg"]]
    return np.array([f"{r}, {e} k={'3+' if kk==3 else kk}" for r,e,kk in zip(reg,ev,k)])

def greedy(Ef, acc, K, cand):
    """forward selection on training networks: Ef (n,B) float outcomes, returns chosen input indices"""
    X=Ef[:,cand]-Ef[:,cand].mean(0); y=acc-acc.mean(); r=y.copy(); Q=np.zeros((len(y),0)); chosen=[]
    for _ in range(K):
        Xr=X-Q@(Q.T@X); nx=(Xr**2).sum(0); score=np.where(nx>1e-9,(Xr.T@r)**2/np.maximum(nx,1e-12),-1)
        j=int(score.argmax()); chosen.append(cand[j]); q=Xr[:,j]/np.sqrt(nx[j]); Q=np.c_[Q,q]; r=r-q*(q@r)
    return np.array(chosen)

if __name__=="__main__":
    d,f=os.path.split(sys.argv[1]); z=np.load(os.path.join(d,"z4_"+f)); x=z["x"]
    E=np.unpackbits(z["E"],1)[:,:len(x)].astype(bool); N,B=E.shape; acc=1-E.mean(1); V=input_vars(x); lab=mech_label(V)
    r=np.load(sys.argv[1].replace(".npz",".roles.npz")); F,nm=r["feats"],list(r["names"]); R=roles(F,nm); Xf=bag(R,F,8)
    tr=np.random.default_rng(0).permutation(N)<N//2; te=~tr; Ef=E.astype(np.float32)
    u=Ef[tr].mean(0); c=((Ef[tr]-u)*(acc[tr]-acc[tr].mean())[:,None]).mean(0); v=u*(1-u)
    I=np.where(v>0,c**2/np.maximum(v,1e-12),0); corr=np.where(v>0,c/np.sqrt(np.maximum(v,1e-12))/acc[tr].std(),0)
    print(f"{N} networks, {B} inputs; accuracy {acc.mean():.4f} +- {acc.std()*100:.2f} points")
    print(f"single-input informativeness: best corr {corr.min():+.3f}; inputs with |corr| > .3: {np.mean(np.abs(corr)>.3)*100:.2f}%, > .5: {np.mean(np.abs(corr)>.5)*100:.3f}%")
    top=np.argsort(-I)[:500]; cells,cnt=np.unique(lab[top],return_counts=True); base={c_:np.mean(lab==c_) for c_ in cells}
    print("\nwhere the 500 most informative inputs fall (share of top 500 vs share of all inputs; mean corr there):")
    for i in np.argsort(-cnt)[:12]:
        m=lab[top]==cells[i]; print(f"  {cells[i]:28s} {cnt[i]/5:5.1f}% vs {base[cells[i]]*100:5.2f}%   corr {corr[top][m].mean():+.2f}   error rate {u[top][m].mean()*100:4.1f}%")
    print("\nheld-out RMSE (points) predicting accuracy from K outcomes: greedy-selected inputs (ridge) vs K random inputs (affine-calibrated mean)")
    cand=np.argsort(-I)[:8000]; rng=np.random.default_rng(1)
    for K in (16,64,256,1024):
        ch=greedy(Ef[tr],acc[tr],K,cand)
        mdl=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,13))).fit(Ef[tr][:,ch],acc[tr])
        sel=np.sqrt(np.mean((mdl.predict(Ef[te][:,ch])-acc[te])**2))*100
        rr=rng.choice(B,K,replace=False); A=np.c_[np.ones(N),Ef[:,rr].mean(1)]; cf=np.linalg.lstsq(A[tr],acc[tr],rcond=None)[0]
        rnd=np.sqrt(np.mean((A[te]@cf-acc[te])**2))*100
        print(f"  K={K:5d}: selected {sel:.3f}   random {rnd:.3f}   (sampling SE {np.sqrt(acc.mean()*(1-acc.mean())/K)*100:.2f})")
        if K==64: ch64=ch
    print("\nfor the first 10 selected inputs: error rate, corr with accuracy, cell, and how well circuit features predict which networks fail (CV AUC)")
    from sklearn.metrics import roc_auc_score
    for j in ch64[:10]:
        y=E[:,j]
        if y.sum()<10 or (~y).sum()<10: continue
        p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000)),Xf,y,cv=5,method="predict_proba")[:,1]
        cm=np.corrcoef((R==6).sum(1),y)[0,1]
        print(f"  input {j:6d}: wrong {y.mean()*100:5.1f}%  corr {corr[j]:+.2f}  {lab[j]:28s} AUC from roles {roc_auc_score(y,p):.2f}  corr with #comparators {cm:+.2f}"
              f"   x = {np.array2string(x[j],precision=2,separator=' ')}")
