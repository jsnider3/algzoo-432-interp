"""Does the correction amortize across arrangements?  256 independent draws per MF4 arrangement (a measurement only),
on the 64 development networks (0::16) and the zoo: the noise-corrected SD across arrangement means of true - MF4
(5,040 arrangements) and of true - MF3 (grouped into the 720 MF3 arrangements), and the share explained by the
positions of M and S.  Checks ChatGPT round 5's 2.05 / 3.94 points and 16.9% / 21.9%.  Usage: python z43.py
"""
import itertools, numpy as np, torch as th
from batch_train import forward
from z2 import load
from seeds_clocks import load as zload
from z10 import mu_below
dev='cuda'; reps=256
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
ids=np.r_[np.arange(0,1024,16),np.arange(1024,1030)]; W=[th.cat([a,b])[ids].to(dev) for a,b in zip(P,Wz)]; N=len(ids)
arr=th.tensor(list(itertools.permutations(range(10),4)),device=dev); A=len(arr)
low=th.tensor([[i for i in range(10) if i not in r] for r in arr.tolist()],device=dev)
g=th.Generator(device=dev).manual_seed(11); S1=th.zeros(2,N,A,device=dev,dtype=th.float64); S2=th.zeros_like(S1)
with th.no_grad():
    for st in range(0,A,120):
        a=arr[st:st+120].repeat_interleave(reps,0); lp=low[st:st+120].repeat_interleave(reps,0); B=len(a)
        v=th.randn(B,10,device=dev,generator=g).sort(-1,descending=True).values[:,:4]
        u=th.rand(B,6,device=dev,generator=g)*th.special.ndtr(v[:,3:4]); x=th.zeros(B,10,device=dev)
        x.scatter_(1,lp,th.special.ndtri(u.clamp(min=1e-30))); x.scatter_(1,a,v); y=a[:,1]
        tr=(forward(*W,x[None].expand(N,-1,-1)).argmax(-1)==y).double()
        for j,k in enumerate((3,4)):
            b=mu_below(v[:,k-1:k]).expand(B,10).clone(); b.scatter_(1,a[:,:k],v[:,:k])
            d=(tr-(forward(*W,b[None].expand(N,-1,-1)).argmax(-1)==y).double()).view(N,-1,reps)
            S1[j,:,st:st+120]=d.sum(-1); S2[j,:,st:st+120]=(d*d).sum(-1)
m=(S1/reps).cpu().numpy(); vw=((S2/reps).cpu().numpy()-m**2)/(reps-1)      # noise variance of each arrangement mean
A_=arr.cpu().numpy(); cls=A_[:,0]*10+A_[:,1]; uc=np.unique(cls)
def share(mean,var,grp,size,tot):
    ug=np.unique(grp); cm=np.stack([np.bincount(grp,weights=mean[i],minlength=grp.max()+1)[ug]/size for i in range(N)])
    cv=np.stack([np.bincount(grp,weights=var[i],minlength=grp.max()+1)[ug]/size**2 for i in range(N)])
    return (cm.var(1)-cv.mean(1))/tot
print("points; median over the 64 dev networks (10th-90th percentile); zoo SDs in order mystery, seed0..4")
tot=m[1].var(1)-vw[1].mean(1); sd=100*np.sqrt(tot); sh=share(m[1],vw[1],cls,56,tot)
print(f"  true - MF4 over 5,040 arrangements: SD {np.median(sd[:64]):.3f} ({np.percentile(sd[:64],10):.3f}-{np.percentile(sd[:64],90):.3f}); (M,S) share {100*np.median(sh[:64]):.1f}%; zoo "+" ".join(f"{v:.2f}" for v in sd[64:]))
g3=A_[:,0]*100+A_[:,1]*10+A_[:,2]; u3=np.unique(g3)
gm=np.stack([np.bincount(g3,weights=m[0,i],minlength=1000)[u3]/7 for i in range(N)]); gv=np.stack([np.bincount(g3,weights=vw[0,i],minlength=1000)[u3]/49 for i in range(N)])
tot3=gm.var(1)-gv.mean(1); sd3=100*np.sqrt(tot3); sh3=share(m[0],vw[0],cls,56,tot3)
print(f"  true - MF3 over 720 MF3 arrangements: SD {np.median(sd3[:64]):.3f} ({np.percentile(sd3[:64],10):.3f}-{np.percentile(sd3[:64],90):.3f}); (M,S) share {100*np.median(sh3[:64]):.1f}%; zoo "+" ".join(f"{v:.2f}" for v in sd3[64:]))
tot0=m[0].var(1)-vw[0].mean(1)
print(f"  true - MF3 over 5,040 MF4 arrangements: SD {np.median(100*np.sqrt(tot0[:64])):.3f}")
