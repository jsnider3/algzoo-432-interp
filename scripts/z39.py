"""A different decomposition: by events instead of by the top-k values.  Each step's relative rank among the inputs so
far is uniform and independent, so the sequence of events (new max, new 2nd max, neither) has a product law.  The event
anchor keeps every record input exact (each new max or new 2nd max, at its value) and puts each "neither" input at its
conditional mean below the running 2nd max at that time, mu(S_{t-1}) = E[x | x < S_{t-1}] (at t = 1, 2 there is no
"neither").  Variants: also keep the last k "neither" inputs exact.
Measured: the anchor's accuracy against the network's true accuracy, on 2^22 sampled inputs per network (paired), and
the number of exact inputs.  Compare the top-4 anchor (MF4): bias +0.30, RMS 0.42 points; MF3: RMS 0.57 (REPORT 7.16).
Every 16th population network + the 6 zoo models.  Usage: python scripts/z39.py [log2 inputs, default 22]
"""
import sys, numpy as np, torch as th
from batch_train import forward, target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; lc=int(sys.argv[1]) if len(sys.argv)>1 else 22
P,_=load('pop1k/seed0.npz'); P=[w[::16] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
VARS=[('event anchor',0),('+ last neither exact',1),('+ last 2 neither exact',2)]
S=np.zeros((N,len(VARS)+1)); nexact=np.zeros(len(VARS)); g=th.Generator().manual_seed(163); n=0
with th.no_grad():
    for i in range(0,2**lc,2**16):
        x=th.randn(2**16,10,generator=g).to(dev); y=target(x); B=len(x)
        m1=th.full((B,),-1e9,device=dev); m2=th.full((B,),-1e9,device=dev); neither=th.zeros(B,10,dtype=th.bool,device=dev); mu=th.zeros(B,10,device=dev)
        for t in range(10):
            xt=x[:,t]; nm=xt>m1; n2=(~nm)&(xt>m2); ne=~(nm|n2)
            neither[:,t]=ne; mu[:,t]=th.where(ne,mu_below(m2.clamp(min=-8)[:,None])[:,0],0.)
            m2=th.where(nm,m1,th.where(n2,xt,m2)); m1=th.where(nm,xt,m1)
        tr=(forward(*W,x[None].expand(N,-1,-1)).argmax(-1)==y[None]).float().sum(1)
        S[:,0]+=tr.cpu().numpy()
        for v,(lab,keep) in enumerate(VARS):
            ne=neither.clone()
            if keep:
                idx=th.cumsum(ne.flip(1).long(),1).flip(1)                                   # rank of each neither step from the end
                ne&=idx>keep
            X=th.where(ne,mu,x); nexact[v]+=(~ne).float().sum().item()
            S[:,v+1]+=(forward(*W,X[None].expand(N,-1,-1)).argmax(-1)==y[None]).float().sum(1).cpu().numpy()
        n+=B
acc=S[:,0]/n
print(f"{N-6} population networks (every 16th) + 6 zoo; event decomposition anchors on 2^{lc} sampled inputs (paired); points")
print(f"  true accuracy: pop median {np.median(acc[pop]):.4f}")
for v,(lab,keep) in enumerate(VARS):
    b=100*(S[:,v+1]/n-acc)
    print(f"  {lab:24s} exact inputs per sequence {nexact[v]/n:.2f}: bias mean {b[pop].mean():+.3f}  RMS {np.sqrt(np.mean(b[pop]**2)):.3f}  SD {b[pop].std():.3f}"
          f"   zoo "+" ".join(f"{z:+.3f}" for z in b[-6:]))
