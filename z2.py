"""Neuron roles across a population: per-neuron features that REPORT's role definitions use, and per-network
error rates by failure region, for asking whether a network's mix of neuron roles predicts its errors.

Per neuron (index-free, so neurons can be pooled and bucketed across networks), on B probe inputs:
  weights      w_in = W_ih[i], self-loop W_hh[i,i], log norms of its recurrent row/column and readout column
  firing       active rate; P(active at t | event at t) and P(active at t | event at t-1) for event = new max,
               new 2nd max, neither (2.2, 7.9, 7.14: detectors, pointers, one-step-late pointers, always-on)
  threshold    pre-activation z_t ~ c_x x_t + c_M M_{t-1} + c_S S_{t-1}: unit direction (c_x, c_M, c_S) and R^2
               (comparators: threshold -(c_M M + c_S S)/c_x, 2.2)
  final state  eta^2 of h_T/|x| on the max's age and on the 2nd max's age (clocks, markers, 2.3); corr with M, S
  readout      DFT of W_o[:, i] over output positions: DC share, dominant frequency, its share (2.5)
Per network, on a shared eval pool: accuracy and error counts in 60 task strata (z1.task_strata: 4 for S<0,
plus a 7x8 grid in (M-S)/|M| and (S-R)/|M|; REPORT 7.15's "60 finer gap strata").

Usage: python z2.py pop1k/snap_2e28.npz [--device cpu] -> pop1k/snap_2e28.roles.npz
"""
import argparse, math, numpy as np, torch as th
from batch_train import forward, target
from z1 import task_strata
FEATS=["w_in","self","log_rec_in","log_rec_out","log_out","rate",
       "on_newmax","on_new2nd","on_neither","after_newmax","after_new2nd","after_neither",
       "thr_cx","thr_cM","thr_cS","thr_R2","eta_ageM","eta_ageS","corr_M","corr_S","mean_hT",
       "out_dc","out_kdom","out_domshare"]

def load(path):
    z=np.load(path); s=max(int(k[1:-4]) for k in z.files if k.startswith("s") and k.endswith("_Wih"))
    return [th.tensor(z[f"s{s}_{k}"],dtype=th.float32) for k in ("Wih","Whh","Wo")], s

def task_vars(x):
    """per (b,t): M, S of x[:, :t] (running max and 2nd max before t), event at t (0 new max, 1 new 2nd, 2 neither);
    final ages of the max and 2nd max"""
    B,T=x.shape; M=th.full((B,T),-math.inf); S=th.full((B,T),-math.inf); ev=th.full((B,T),-1,dtype=th.long)
    m=th.full((B,),-math.inf); s=th.full((B,),-math.inf)
    for t in range(T):
        M[:,t]=m; S[:,t]=s; xt=x[:,t]
        ev[:,t]=th.where(xt>m,0,th.where(xt>s,1,2))
        s=th.where(xt>m,m,th.maximum(s,xt)); m=th.maximum(m,xt)
    o=x.argsort(1); return M,S,ev,T-1-o[:,-1],T-1-o[:,-2]

def eta2(v, g, G):
    """correlation ratio of v (n,B,H) on group labels g (B,) in 0..G-1"""
    oh=th.nn.functional.one_hot(g,G).float(); cnt=oh.sum(0).clamp(min=1)
    mu=th.einsum("nbh,bg->ngh",v,oh)/cnt[None,:,None]
    tot=v.var(1,unbiased=False); between=th.einsum("ngh,g->nh",(mu-v.mean(1,keepdim=True))**2,cnt)/v.shape[1]
    return between/tot.clamp(min=1e-12)

@th.no_grad()
def neuron_features(P, x, chunk=128):
    Wih,Whh,Wo=P; N,H=Wih.shape; B,T=x.shape; M,S,ev,aM,aS=task_vars(x); xn=x.norm(dim=1)
    out=[]
    for c in range(0,N,chunk):
        wi,wh,wo=Wih[c:c+chunk],Whh[c:c+chunk],Wo[c:c+chunk]; n=len(wi)
        h=th.zeros(n,B,H); hs=[]; zs=[]
        for t in range(T):
            z=x[None,:,t,None]*wi[:,None,:]+h@wh.transpose(1,2); h=th.relu(z); hs.append(h); zs.append(z)
        A=th.stack(hs,2)>0; Z=th.stack(zs,2)                                    # (n,B,T,H)
        f=[wi, th.diagonal(wh,dim1=1,dim2=2), wh.norm(dim=2).log(), wh.norm(dim=1).log(), wo.norm(dim=1).log()]
        A2=A[:,:,2:].float(); f.append(A2.mean((1,2)))
        for e in range(3):                                                      # firing on / one step after events
            msk=(ev[:,2:]==e).float(); f.append(th.einsum("nbth,bt->nh",A2,msk)/msk.sum())
        for e in range(3):
            msk=(ev[:,2:-1]==e).float(); f.append(th.einsum("nbth,bt->nh",A[:,:,3:].float(),msk)/msk.sum())
        X=th.stack([x[:,2:],M[:,2:],S[:,2:]],-1).reshape(-1,3)                  # threshold fit, t >= 2
        Zt=Z[:,:,2:].reshape(n,-1,H); coef=th.linalg.lstsq(X[None].expand(n,-1,-1),Zt).solution   # (n,3,H)
        r2=1-((Zt-X@coef)**2).sum(1)/((Zt-Zt.mean(1,keepdim=True))**2).sum(1).clamp(min=1e-12)
        u=coef/coef.norm(dim=1,keepdim=True).clamp(min=1e-12); f+= [u[:,0],u[:,1],u[:,2],r2]
        hT=hs[-1]/xn[None,:,None]                                               # final state per unit |x| (homogeneity)
        f+= [eta2(hT,aM,T), eta2(hT,aS,T)]
        o=x.sort(1).values; Mv,Sv=o[:,-1]/xn,o[:,-2]/xn                          # M, S of all 10 inputs
        for v in (Mv,Sv):
            vc=(v-v.mean())/v.std(); hc=(hT-hT.mean(1,keepdim=True))/hT.std(1,keepdim=True).clamp(min=1e-12)
            f.append((hc*vc[None,:,None]).mean(1))
        f.append(hT.mean(1))
        F=th.fft.rfft(wo,dim=1).abs()                                           # (n,6,H) over output positions
        sh=F**2/(F**2).sum(1,keepdim=True).clamp(min=1e-12); kd=sh[:,1:].argmax(1)+1
        f+= [sh[:,0], kd.float(), sh[:,1:].max(1).values]
        out.append(th.stack(f,-1))
    return th.cat(out).numpy()                                                  # (N,H,F)

@th.no_grad()
def strata_errors(P, x, chunk=4096):
    lab=th.tensor(task_strata(x.numpy())); K=int(lab.max())+1; N=P[0].shape[0]; err=th.zeros(N,K)
    for i in range(0,len(x),chunk):
        xc=x[i:i+chunk]; e=(forward(*P,xc[None].expand(N,-1,-1)).argmax(-1)!=target(xc)[None]).float()
        err+=e@th.nn.functional.one_hot(lab[i:i+chunk],K).float()
    return err.numpy(), np.bincount(lab.numpy(),minlength=K)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("pop"); ap.add_argument("--probe",type=int,default=4096)
    ap.add_argument("--pool",type=int,default=2**18); ap.add_argument("--threads",type=int,default=6); a=ap.parse_args()
    th.set_num_threads(a.threads); P,s=load(a.pop)
    g=th.Generator().manual_seed(11); xp=th.randn(a.probe,10,generator=g); xe=th.randn(a.pool,10,generator=g)
    F=neuron_features(P,xp); E,n=strata_errors(P,xe); acc=1-E.sum(1)/a.pool
    dst=a.pop.replace(".npz",".roles.npz"); np.savez(dst,feats=F,names=np.array(FEATS),strata_err=E,strata_n=n,acc=acc,snapshot=s)
    print(f"{len(acc)} networks (snapshot s{s}); accuracy 5/50/95%: {np.percentile(acc,[5,50,95]).round(4)}; saved {dst}")
