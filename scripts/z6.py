"""Write and handoff latency across a population (needs z4's error masks).

For non-tie inputs (z3 regions middle + clean), the per-network error rate when the 2nd max was
  handoff  the old max, displaced by a new max that arrived k steps before the end (age of max = k < age of 2nd)
  direct   written directly as a new 2nd max k steps before the end (age of 2nd = k < age of max)
for k = 0..3. Prints the population means, the relation to comparator count and pointer type (z3 roles), how well
the 8-number latency profile predicts accuracy, and how well network features predict each latency rate.
Usage: python scripts/z6.py pop1k/snap_2e28.npz [pop1k/snap_2e28.roles.npz]   (reads pop1k/z4_snap_2e28.npz)
"""
import os, sys, numpy as np
from z3 import roles, bag, cv_rmse
from z4 import input_vars

if __name__=="__main__":
    d,f=os.path.split(sys.argv[1]); z=np.load(os.path.join(d,"z4_"+f)); x=z["x"]
    E=np.unpackbits(z["E"],1)[:,:len(x)].astype(bool); V=input_vars(x); acc=1-E.mean(1)
    lab=sys.argv[2] if len(sys.argv)>2 else sys.argv[1].replace(".npz",".roles.npz")
    r=np.load(lab); F,nm=r["feats"],list(r["names"]); R=roles(F,nm); C=np.stack([(R==k).sum(1) for k in range(8)],1)
    nt=V["reg"]>=3
    H=np.stack([E[:,nt&(V["aM"]==k)&(V["aS"]>k)].mean(1) for k in range(4)],1)
    D=np.stack([E[:,nt&(V["aS"]==k)&(V["aM"]>k)].mean(1) for k in range(4)],1)
    print(f"{len(E)} networks, accuracy {acc.mean():.4f} +- {acc.std()*100:.2f} points; non-tie error rate (%) k steps after the event:")
    for k in range(4):
        print(f"  k={k}: handoff {H[:,k].mean()*100:5.1f} (SD {H[:,k].std()*100:4.1f}, corr with accuracy {np.corrcoef(H[:,k],acc)[0,1]:+.2f})"
              f"   direct {D[:,k].mean()*100:5.1f} (SD {D[:,k].std()*100:4.1f}, corr {np.corrcoef(D[:,k],acc)[0,1]:+.2f})")
    print("handoff k=0 by comparator count: "+"  ".join(f"{c}: {H[C[:,6]==c,0].mean()*100:.1f}% (n={np.sum(C[:,6]==c)})" for c in range(8) if np.sum(C[:,6]==c)>=10))
    for nm2,m in [("event pointer only",(C[:,2]>0)&(C[:,3]==0)),("late pointer only",(C[:,3]>0)&(C[:,2]==0)),("no pointer",(C[:,2]==0)&(C[:,3]==0))]:
        print(f"  {nm2:18s} n={m.sum():4d}: handoff k=0 {H[m,0].mean()*100:5.1f}%, k=1 {H[m,1].mean()*100:4.1f}%; accuracy {acc[m].mean():.4f}")
    print("accuracy from the 8-number latency profile, CV RMSE (points): "+"  ".join(f"{k} {v*100:.3f}" for k,v in cv_rmse(np.c_[H,D],acc).items()))
    w=np.load(sys.argv[1]); s=max(int(k[1:-4]) for k in w.files if k.startswith("s") and k.endswith("_Wih")); Wi,Wh,Wo=(w[f"s{s}_{k}"] for k in ("Wih","Whh","Wo"))
    Wf=np.concatenate([np.sort(Wi,1),np.sort(np.diagonal(Wh,0,1,2),1),np.sort(np.abs(np.linalg.eigvals(Wh)),1),np.sort(np.linalg.norm(Wo,axis=1),1)],1)
    for nm2,y in [("handoff k=0",H[:,0]),("handoff k=1",H[:,1]),("direct k=0",D[:,0])]:
        print(f"  {nm2} (SD {y.std()*100:.2f}%) predicted from "+"; ".join(f"{fn}: "+" ".join(f"{k} R2 {1-(v/y.std())**2:.2f}" for k,v in cv_rmse(X,y).items())
              for fn,X in [("roles+means",bag(R,F,8)),("weights",Wf)]))
