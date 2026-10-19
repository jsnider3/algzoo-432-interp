"""A paired reference table for scoring estimators without a GPU: for all 1,024 population networks and the 6 zoo
models, on the same 2^lc inputs, the true accuracy and the accuracy of the MF2, MF3 and MF4 mean-field anchors (top k
inputs exact, the others at mu(x_(k)) = E[x | x < x_(k)]).  Because the inputs are shared, anchor - truth (the leakage)
has a much smaller SE than either accuracy.  Saves pop1k/reference.npz (acc, mf2, mf3, mf4, se_acc, se_leak2/3/4, names).
Usage: python z40.py [log2 inputs, default 20]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import topk_mf
from seeds_clocks import load as zload
dev='cuda'; lc=int(sys.argv[1]) if len(sys.argv)>1 else 20
P,_=load('pop1k/seed0.npz')
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
names=np.array([f"pop{i}" for i in range(N-6)]+["mystery"]+[f"seed{i}" for i in range(5)])
K=[2,3,4]; S=th.zeros(N,1+len(K),dtype=th.float64,device=dev); D=th.zeros(N,len(K),2,dtype=th.float64,device=dev)   # sums of correct; sums of d, d^2 for d = anchor - true
g=th.Generator().manual_seed(20261019); B=2**14
with th.no_grad():
    for i in range(0,2**lc,B):
        x=th.randn(B,10,generator=g).to(dev); y=target(x)
        rt=(forward(*W,x[None].expand(N,-1,-1)).argmax(-1)==y[None]).double(); S[:,0]+=rt.sum(1)
        for j,k in enumerate(K):
            rv=(forward(*W,topk_mf(x,k)[None].expand(N,-1,-1)).argmax(-1)==y[None]).double(); S[:,j+1]+=rv.sum(1)
            d=rv-rt; D[:,j,0]+=d.sum(1); D[:,j,1]+=(d*d).sum(1)
n=2**lc; A=(S/n).cpu().numpy(); Dm=(D[...,0]/n).cpu().numpy(); Dv=(D[...,1]/n).cpu().numpy()-Dm**2
se_acc=np.sqrt(A[:,0]*(1-A[:,0])/n); se_leak=np.sqrt(Dv/n)
np.savez('pop1k/reference.npz',acc=A[:,0],mf2=A[:,1],mf3=A[:,2],mf4=A[:,3],se_acc=se_acc,se_leak2=se_leak[:,0],se_leak3=se_leak[:,1],se_leak4=se_leak[:,2],names=names,n=n)
a=A[pop,0]; pq=a*(1-a)
print(f"{N-6} population networks + 6 zoo, 2^{lc} shared inputs (seed 20261019); points")
print(f"  true accuracy: mean {100*a.mean():.3f}, SD {100*a.std():.3f}, percentiles 5/50/95 {' '.join(f'{100*v:.2f}' for v in np.percentile(a,[5,50,95]))}; mean p(1-p) {pq.mean():.5f}; SE {100*se_acc[pop].mean():.4f}")
print(f"  agreement with acc_2e20: RMS diff {100*np.sqrt(np.mean((a-np.load('pop1k/seed0.npz')['acc_2e20'])**2)):.4f}")
for j,k in enumerate(K):
    l=100*(A[pop,j+1]-a)
    print(f"  MF{k} leakage (anchor - truth): mean {l.mean():+.3f}  SD {l.std():.3f}  RMS {np.sqrt((l**2).mean()):.3f}  SE per network {100*se_leak[pop,j].mean():.4f}"
          f"  zoo "+" ".join(f"{100*(A[-6+z,j+1]-A[-6+z,0]):+.3f}" for z in range(6)))
print("  zoo true accuracy: "+"  ".join(f"{names[-6+z]} {A[-6+z,0]:.5f} ({100*se_acc[-6+z]:.4f})" for z in range(6)))
