"""The linear-region closed form with its patterns chosen by an oracle, measured rather than extrapolated (z26 used z25's
fallback rate).  For 40 arrangements per network (lower inputs at mu(Q)):
  training draws (2^14)  give the arrangement's activation patterns; each pattern's exact linear logit map
                         (sum_t c_t x_t) and its centroid in the CDF coordinates of z21
  selection              K patterns, either the K most massive, or greedily the pattern whose map (used for every draw
                         whose nearest selected centroid it is) most reduces disagreement with the network
  held-out draws (2^14)  a draw in a selected pattern is answered exactly; any other draw by its nearest selected
                         pattern's map
Reported: disagreement with the network's answer and the net error of the anchor accuracy (points), averaged over the
40 arrangements, against sampling at the closed form's build cost of 5,040 K forwards (selection is free: an oracle).
Every 64th population network + the 6 zoo models.  Usage: python scripts/z30.py [Kmax, default 100] [arrangements, default 40]
"""
import sys, itertools, numpy as np, torch as th
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; Kmax=int(sys.argv[1]) if len(sys.argv)>1 else 100; KS=[k for k in (1,3,10,30,100,300) if k<=Kmax]
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
ARR=list(itertools.permutations(range(10),4)); NA=int(sys.argv[2]) if len(sys.argv)>2 else 40; sel=[ARR[i] for i in np.linspace(0,len(ARR)-1,NA).astype(int)]
def run(w,X):
    """answers' logits, gate pattern hash, linear maps c (B, 10, 10)"""
    h=th.zeros(len(X),16,device=dev); G=[]
    for t in range(10):
        pre=X[:,t:t+1]*w[0]+h@w[1].T; G.append(pre>0); h=th.relu(pre)
    G=th.stack(G,1); J=w[2][None].expand(len(X),10,16)*G[:,9,None,:].float(); C=[None]*10
    for t in range(9,-1,-1):
        C[t]=J@w[0]
        if t>0: J=(J@w[1])*G[:,t-1,None,:].float()
    key=G.reshape(len(X),-1).long()
    hsh=(key*th.randint(1,2**31,(160,),generator=th.Generator().manual_seed(5)).to(dev)).sum(-1)
    return h@w[2].T, hsh, th.stack(C,1)
def draws(arr,n,g):
    v=th.randn(n*3,10,generator=g).to(dev).topk(4,-1).values[:n]
    X=mu_below(v[:,3:4]).expand(n,10).clone()
    for c in range(4): X[:,arr[c]]=v[:,c]
    f=th.special.ndtr(v); cc=th.stack([f[:,3],(f[:,2]-f[:,3])/(1-f[:,3]),(f[:,0]-f[:,2])/(1-f[:,2]),(f[:,1]-f[:,2])/(f[:,0]-f[:,2])],-1)
    return X,cc
res={m:np.zeros((N,len(KS),len(sel),2)) for m in ('mass','greedy')}
g=th.Generator().manual_seed(103); n=2**14
with th.no_grad():
    for ai,arr in enumerate(sel):
        Xt,ct=draws(arr,n,g); Xh,ch=draws(arr,n,g); tgt=arr[1]
        for i in range(N):
            w=[q[i] for q in W]
            Lt,ht,Ct=run(w,Xt); Lh,hh,_=run(w,Xh); It=Lt.argmax(-1)==tgt; Ih=Lh.argmax(-1)==tgt
            u,inv,cnt=th.unique(ht,return_inverse=True,return_counts=True)
            order=cnt.argsort(descending=True)[:300]; M=len(order)                       # candidate patterns
            first=th.full((len(u),),-1,dtype=th.long,device=dev); first.scatter_(0,inv,th.arange(n,device=dev))   # one draw per pattern
            Cc=Ct[first[order]]                                                         # (M, 10, 10) maps
            cen=th.stack([ct[inv==k].mean(0) for k in order.tolist()])                   # (M, 4) centroids
            def answers(Xq,cq,hq,setk):
                """closed-form answers for draws Xq with the selected candidate indices setk"""
                d=((cq[:,None,:]-cen[setk][None])**2).sum(-1); nn=setk[d.argmin(1)]
                Lc=(Xq[:,:,None]*Cc[nn]).sum(1); a=Lc.argmax(-1)==tgt
                own=(hq[:,None]==u[order[setk]][None]).any(1)
                return a,own
            # greedy on the training draws: answers of each candidate's map on all training draws, distances

            Acand=th.stack([((Xt[:,:,None]*Cc[k][None]).sum(1).argmax(-1)==tgt) for k in range(M)])   # (M, n)
            Dcand=((ct[None,:,:]-cen[:,None,:])**2).sum(-1)                                # (M, n)
            own_t=(ht[None,:]==u[order][:,None])                                           # (M, n): draw lies in candidate's pattern
            chosen=[]; curd=th.full((n,),float('inf'),device=dev); cura=th.zeros(n,dtype=th.bool,device=dev); exact=th.zeros(n,dtype=th.bool,device=dev)
            gsets={}
            for step in range(min(max(KS),M)):
                newd=th.minimum(curd[None],Dcand); take=Dcand<curd[None]
                newa=th.where(take,Acand,cura[None]); newe=exact[None]|own_t
                agree=(th.where(newe,It[None],newa)==It[None]).sum(1).float()
                if chosen: agree[th.tensor(chosen,device=dev)]=-1
                k=int(agree.argmax()); chosen.append(k)
                cura=th.where(Dcand[k]<curd,Acand[k],cura); curd=th.minimum(curd,Dcand[k]); exact|=own_t[k]
                if step+1 in KS: gsets[step+1]=th.tensor(chosen,device=dev)
            for j,K in enumerate(KS):
                for mname,setk in (('mass',th.arange(min(K,M),device=dev)),('greedy',gsets.get(min(K,M),th.tensor(chosen,device=dev)))):
                    a,own=answers(Xh,ch,hh,setk); Icf=th.where(own,Ih,a)
                    res[mname][i,j,ai]=[(Icf!=Ih).float().mean().item(),(Icf.float().mean()-Ih.float().mean()).item()]
print(f"{N-6} population networks (every 64th) + 6 zoo; {len(sel)} arrangements, 2^14 training and 2^14 held-out draws each; oracle-chosen patterns")
for mname in ('mass','greedy'):
    print(f"  patterns chosen by {mname}:")
    for j,K in enumerate(KS):
        r=res[mname][:,j]; c=5040*K; e=100*r[pop,:,1]; tot=e.mean(1); noise2=e.var(1)/e.shape[1]      # arrangement-sampling noise of the total
        print(f"    K = {K:4d}: build {c:>9,} forwards; disagreement {100*np.median(r[pop,:,0].mean(1)):.2f}%; net error RMS {np.sqrt(np.mean(tot**2)):.3f}"
              f" (arrangement noise {np.sqrt(noise2.mean()):.3f}, removed {np.sqrt(max(0,np.mean(tot**2)-noise2.mean())):.3f}) points; sampling at that cost {100*np.sqrt(0.07/c):.3f}")
np.savez('pop1k/seed0_z30.npz',**res)
