"""How many linear regions does one arrangement's value space hold?  For a closed form built from one linear map per
activation pattern (z25), the cost is one forward per (arrangement, pattern), so it matters how many patterns carry the
probability mass.  For R arrangements per network (a fixed spread of positions), 2^15 draws of the top-4 values from
their order-statistic law (lower inputs at mu(Q)), count the distinct activation patterns and the mass held by the
1, 10, 100 most common.  Every 64th population network + the 6 zoo models.
Usage: python z26.py [arrangements, default 40] [log2 draws, default 15]
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
g=th.Generator().manual_seed(89); res=np.zeros((N,R,4))
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
            res[i,r]=[len(cnt),cnt[:1].sum(),cnt[:10].sum(),cnt[:100].sum()]
print(f"{N-6} population networks (every 64th) + 6 zoo; {R} arrangements, 2^{ld} value draws each (lower inputs at mu(Q))")
for k,lab in enumerate(["distinct activation patterns seen","mass of the most common pattern","mass of the 10 most common","mass of the 100 most common"]):
    v=res[:,:,k].mean(1); fmt="{:.0f}" if k==0 else "{:.3f}"
    print(f"  {lab:36s} pop median {fmt.format(np.median(v[pop]))} [p10 {fmt.format(np.percentile(v[pop],10))}, p90 {fmt.format(np.percentile(v[pop],90))}]  zoo "+" ".join(fmt.format(z) for z in v[-6:]))
np.save('pop1k/seed0_z26.npy',res)
