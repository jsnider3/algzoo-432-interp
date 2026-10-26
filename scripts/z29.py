"""The clock-state model's core simplification: the clock and memory channels are linear.  REPORT's closed-form proposal
is a model of the selector's state built from clock eigenmodes (the usually-active neurons, run as a linear system) plus
age-dependent comparator thresholds (the switching neurons).  If the usually-active neurons never switched, removing
their ReLU would change nothing, and only the switching neurons would make linear regions: few regions, a closed form.
Test, per network: neurons active on at least a fraction f of steps (on sampled inputs) lose their ReLU (identity), the
rest keep it.  Measured on the MF4 anchor at 2^20 sampled top-4 configurations:
  bias       anchor accuracy of the partly linear network minus the network's (points)
  answers    share of configurations whose answer changes
  regions    distinct activation patterns of the remaining ReLU neurons per arrangement (40 arrangements x 2^14 draws),
             and the mass of the 10 / 100 most common
for f = 1.0 (nothing linearized), 0.99, 0.95, 0.9, 0.8.  Every 64th population network + the 6 zoo models.
Usage: python scripts/z29.py [log2 configurations, default 20]
"""
import sys, itertools, numpy as np, torch as th
from batch_train import target
from z2 import load
from z10 import mu_below
from seeds_clocks import load as zload
dev='cuda'; lc=int(sys.argv[1]) if len(sys.argv)>1 else 20
P,_=load('pop1k/seed0.npz'); P=[w[::64] for w in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]; pop=slice(0,N-6)
def run(w,X,lin,gates=False):
    h=th.zeros(len(X),16,device=dev); G=[]
    for t in range(10):
        pre=X[:,t:t+1]*w[0]+h@w[1].T; h=th.where(lin,pre,th.relu(pre))
        if gates: G.append((pre>0)&~lin)
    return (h@w[2].T, th.stack(G,1) if gates else None)
g=th.Generator().manual_seed(101)
x=th.randn(2**lc,10,generator=g).to(dev); y=target(x); o=x.argsort(-1,descending=True)
X=mu_below(x.gather(-1,o[:,3:4])).expand(len(x),10).clone(); X.scatter_(-1,o[:,:4],x.gather(-1,o[:,:4]))
xs=th.randn(2**16,10,generator=g).to(dev)                                            # ordinary inputs, for activity rates
top1=np.zeros(N); ARR=list(itertools.permutations(range(10),4)); sel=[ARR[i] for i in np.linspace(0,len(ARR)-1,40).astype(int)]
FS=[1.0,0.99,0.95,0.9,0.8]; out=np.zeros((N,len(FS),6))
with th.no_grad():
    for i in range(N):
        w=[q[i] for q in W]; h=th.zeros(len(xs),16,device=dev); act=th.zeros(16,device=dev)
        for t in range(10):
            pre=xs[:,t:t+1]*w[0]+h@w[1].T; act+=(pre>0).float().mean(0); h=th.relu(pre)
        act/=10; top1[i]=act.max().item(); I_true=(run(w,X,th.zeros(16,dtype=th.bool,device=dev))[0].argmax(-1)==y)
        for k,f in enumerate(FS):
            lin=act>=f if f<1 else th.zeros(16,dtype=th.bool,device=dev)
            L,_=run(w,X,lin); I=(L.argmax(-1)==y)
            regs=[]
            for arr in sel:
                v=th.randn(2**14*3,10,generator=g).to(dev).topk(4,-1).values[:2**14]
                Xa=mu_below(v[:,3:4]).expand(len(v),10).clone()
                for c in range(4): Xa[:,arr[c]]=v[:,c]
                _,G=run(w,Xa,lin,gates=True); G=G.reshape(len(v),-1).cpu().numpy()
                _,cnt=np.unique(np.packbits(G,axis=1),axis=0,return_counts=True); cnt=np.sort(cnt)[::-1]/len(v)
                regs.append((len(cnt),cnt[:10].sum(),cnt[:100].sum()))
            r=np.array(regs).mean(0)
            out[i,k]=[100*(I.float().mean()-I_true.float().mean()).item(),100*(I!=I_true).float().mean().item(),*r,lin.sum().item()]

print(f"{N-6} population networks (every 64th) + 6 zoo; MF4 anchor at 2^{lc} sampled configurations; neurons active on >= f of steps made linear")
for k,f in enumerate(FS):
    v=out[:,k]
    print(f"  f = {f:4.2f}: neurons made linear {np.median(v[pop,5]):.0f} (pop median; most active neuron's rate {np.median(top1[pop]):.3f});"
          f" bias RMS {np.sqrt(np.mean(v[pop,0]**2)):.3f} points (median {np.median(v[pop,0]):+.3f});"
          f" answers changed {np.median(v[pop,1]):.2f}%; regions per arrangement {np.median(v[pop,2]):.0f},"
          f" top 10 {np.median(v[pop,3]):.3f}, top 100 {np.median(v[pop,4]):.3f}")
np.save('pop1k/seed0_z29.npy',out)
