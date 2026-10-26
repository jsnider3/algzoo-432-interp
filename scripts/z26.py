"""How many linear regions does one arrangement's value space hold?  For a closed form built from one linear map per
activation pattern (z25), the cost is one forward per (arrangement, pattern), so it matters how many patterns carry the
probability mass.  For R arrangements per network (a fixed spread of positions), 2^15 draws of the top-4 values from
their order-statistic law (lower inputs at mu(Q)), count the distinct activation patterns and the mass held by the
1, 10, 100 most common.  Every 64th population network + the 6 zoo models.
Also the coverage curve of the K most common patterns, the Good-Turing estimate of the unseen mass, and the oracle
bound: the K best patterns of every arrangement for 5,040 K forwards, against sampling at that cost.
Usage: python scripts/z26.py [arrangements, default 40] [log2 draws, default 15]
"""
import sys, itertools, numpy as np, torch as th
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; R=int(sys.argv[1]) if len(sys.argv)>1 else 40; ld=int(sys.argv[2]) if len(sys.argv)>2 else 15
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=list(itertools.permutations(range(10),4)); sel=[ARR[i] for i in np.linspace(0,len(ARR)-1,R).astype(int)]
g=th.Generator().manual_seed(89); res=np.zeros((N,R,4)); KS=np.array([1,3,10,30,100,300,1000]); cov=np.zeros((N,R,len(KS))); gt=np.zeros((N,R))
with th.no_grad():
    for r,arr in enumerate(sel):
        x=th.randn(2**ld*3,10,generator=g).to(dev); top=x.topk(4,-1).values[:2**ld]               # top-4 values, ordered
        X=mu_below(top[:,3:4]).expand(len(top),10).clone()
        for c in range(4): X[:,arr[c]]=top[:,c]
        for i in range(N):
            w=[q[i] for q in W]; h=th.zeros(len(X),16,device=dev); G=[]
            for t in range(10):
                pre=X[:,t:t+1]*w[0]+h@w[1].T; G.append(pre>0); h=th.relu(pre)
            G=th.stack(G,1).reshape(len(X),-1).cpu().numpy()
            _,cnt=np.unique(np.packbits(G,axis=1),axis=0,return_counts=True); cnt=np.sort(cnt)[::-1]/len(X)
            res[i,r]=[len(cnt),cnt[:1].sum(),cnt[:10].sum(),cnt[:100].sum()]; cs=np.cumsum(cnt)
            cov[i,r]=cs[np.minimum(KS,len(cs))-1]; gt[i,r]=(cnt*len(X)==1).sum()/len(X)        # Good-Turing: unseen mass ~ singletons / draws
print(f"{N-6} population networks (every 64th) + 6 zoo; {R} arrangements, 2^{ld} value draws each (lower inputs at mu(Q))")
for k,lab in enumerate(["distinct activation patterns seen","mass of the most common pattern","mass of the 10 most common","mass of the 100 most common"]):
    v=res[:,:,k].mean(1); fmt="{:.0f}" if k==0 else "{:.3f}"
    print(f"  {lab:36s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
print("  coverage by the K most common patterns (seen mass; Good-Turing unseen mass subtracted below):")
for j,k_ in enumerate(KS):
    v=cov[:,:,j].mean(1)*(1-gt.mean(1)); print(f"    K = {k_:5d}: pop median {np.median(v[pop]):.4f} [p10 {np.percentile(v[pop],10):.4f}, p90 {np.percentile(v[pop],90):.4f}]")
print(f"  Good-Turing estimate of mass in patterns never seen: pop median {np.median(gt.mean(1)[pop]):.4f}")
# oracle closed form: K best patterns per arrangement, one forward each; uncovered mass u(K); against sampling at 5,040 K forwards
pq=0.07
for lab,f in (("guaranteed (error <= uncovered mass)",1.),("with z25's fallback error rate (1.4% of uncovered mass)",0.014)):
    print(f"  oracle closed form, {lab}:")
    for j,k_ in enumerate(KS):
        u=1-cov[:,:,j].mean(1)*(1-gt.mean(1)); e=100*f*np.median(u[pop]); c=5040*k_
        print(f"    K = {k_:5d}: build {c:>9,} forwards, error <= {e:.3f} points; sampling at that cost {100*np.sqrt(pq/c):.3f}")
np.savez('pop1k/seed0_z26.npz',res=res,cov=cov,gt=gt,KS=KS)
