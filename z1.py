"""Population ceiling for precomputed (network-independent) accuracy estimators.

For a population of networks, evaluate every network on one fixed pool of inputs. For each network i, the rest of the
population (leave-one-out) gives a precomputable score of each input: pbar_-i(x), the fraction of other networks
that get x wrong. Then measure, at equal evaluations, how much a design built only from that score can beat plain
sampling for network i:
  - stratified sampling over quantile bins of pbar_-i, allocation from the population's own stratum error rates
    (a fully precomputable design), and the Neyman bound with network i's true stratum rates (an oracle ceiling);
  - the same with REPORT 7.15's task-only strata (sign of S, normalized gaps), for comparison.
Gains are variance ratios (= MSE ratios for unbiased estimators) at any budget, computed exactly on the pool. A design
whose strata are known only on pool points estimates the pool's accuracy, which differs from the true accuracy by
sqrt(p(1-p)/pool size), about 0.01 points for the 2^22 pool here; that is negligible for budgets well below the pool.

Usage: python z1.py pop1k/seed0.npz      (or no argument: the 22 networks already in the repo, as a smoke test)
"""
import sys, glob, numpy as np, torch as th
from batch_train import forward, target
dev="cuda"
def load_population(path=None):
    if path:
        z=np.load(path); W=[z[f"s{2**20}_{k}"] for k in ("Wih","Whh","Wo")]; return [th.tensor(w,device=dev,dtype=th.float32) for w in W]
    ws=[np.load(p) for p in ["weights.npz"]+[f"weights_seed{s}.npz" for s in range(5)]+sorted(glob.glob("pop/p*.npz"))]
    return [th.tensor(np.stack([w[k] for w in ws]),device=dev,dtype=th.float32).reshape(len(ws),*s)
            for k,s in [("rnn.weight_ih_l0",(16,)),("rnn.weight_hh_l0",(16,16)),("linear.weight",(10,16))]]
@th.no_grad()
def error_matrix(P, x):
    """(N, len(x)) bool: network i wrong on input x"""
    N=P[0].shape[0]; chunk=max(1,2**24//N); out=[]
    for i in range(0,len(x),chunk):
        xc=x[i:i+chunk]; out.append(forward(*P, xc[None].expand(N,-1,-1)).argmax(-1)!=target(xc)[None])
    return th.cat(out,1)
def task_strata(x):
    s=-np.sort(-x,1); M,S,R=s[:,0],s[:,1],s[:,2]; a=(M-S)/np.abs(M); b=(S-R)/np.abs(M)
    ab=np.digitize(a,[.01,.02,.05,.1,.2,.4])*8+np.digitize(b,[.005,.01,.03,.06,.1,.2,.4])
    return np.where(S<0,np.digitize(S,[-1,-.5,-.2]),4+ab)
def strat_gain(e, lab, alloc_rate=None, defensive=0.1):
    """variance ratio plain/stratified for indicator e (n,), strata lab, allocation n_h ∝ W_h·sd_h with sd from
    alloc_rate (per-stratum error rates used for allocation; None = e's own rates, i.e. the Neyman bound).
    A precomputed allocation keeps a `defensive` share proportional to W_h, so strata where the population
    never errs (but this network might) are still sampled."""
    K=lab.max()+1; W=np.bincount(lab,minlength=K)/len(lab); p=np.bincount(lab,weights=e,minlength=K)/np.maximum(np.bincount(lab,minlength=K),1)
    q=p if alloc_rate is None else alloc_rate; sd_alloc=np.sqrt(np.clip(q*(1-q),1e-6,None))*(W>0)
    frac=W*sd_alloc/np.sum(W*sd_alloc)
    if alloc_rate is not None: frac=(1-defensive)*frac+defensive*W
    var_strat=np.sum(np.where(frac>0, W**2*p*(1-p)/np.maximum(frac,1e-12), 0))   # per unit budget
    m=e.mean(); return m*(1-m)/var_strat
if __name__=="__main__":
    P=load_population(sys.argv[1] if len(sys.argv)>1 else None); N=P[0].shape[0]
    g=th.Generator(device=dev); g.manual_seed(2025); x=th.randn(2**22,10,generator=g,device=dev)
    E=error_matrix(P,x); acc=1-E.float().mean(1).cpu().numpy()
    print(f"{N} networks on {len(x):,} pooled inputs; accuracy 5/50/95%: {np.percentile(acc,[5,50,95]).round(4)}")
    tot=E.float().sum(0).cpu().numpy(); tl=task_strata(x.cpu().numpy()); K=20; rows=[]
    def pop_rates(lab, e):      # other networks' error rate in each stratum, from column sums (leave-one-out)
        n=np.bincount(lab); return np.bincount(lab,weights=tot-e,minlength=len(n))/np.maximum(n,1)/(N-1)
    for i in range(N):
        e=E[i].float().cpu().numpy(); pb=(tot-e)/(N-1)                       # leave-one-out population score
        # strata: inputs no other network gets wrong, then quantile bins of the rest
        nz=pb>0; edges=np.unique(np.quantile(pb[nz],np.linspace(0,1,K)[1:-1])) if nz.any() else []
        lab=np.where(nz,1+np.digitize(pb,edges),0)
        rows.append((acc[i], strat_gain(e,lab,pop_rates(lab,e)), strat_gain(e,lab), strat_gain(e,tl,pop_rates(tl,e)), strat_gain(e,tl),
                     np.corrcoef(e,pb)[0,1]))
    R=np.array(rows)
    print("per-network variance gains over plain sampling (median [5%, 95%] across networks):")
    for j,nm in enumerate(["population-score strata, population allocation (precomputable, 10% defensive)","population-score strata, Neyman oracle",
                           "task strata, population allocation (precomputable, 10% defensive)","task strata, Neyman oracle","corr(error, population score)"],1):
        print(f"  {nm:78s} {np.median(R[:,j]):6.2f}  [{np.percentile(R[:,j],5):.2f}, {np.percentile(R[:,j],95):.2f}]")
    if N<=32:
        for r in R: print("   acc %.4f | pop-strata %.2f / %.2f | task-strata %.2f / %.2f | corr %.2f"%tuple(r))
