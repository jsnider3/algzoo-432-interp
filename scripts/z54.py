"""Does a population trained with the mystery model's recipe (batch_train_unif.py: Unif(-1,1) inputs, Adam 0.005,
no weight decay, x0.995 per epoch, 1,000 epochs of 100,000 sequences; four batch sizes, 64 networks each) contain
networks like the mystery model?  For each group, and for the zoo-recipe population (pop1k/seed0.npz) and the zoo:
accuracy on N(0,1) and on Unif(-1,1) inputs (2^18 each), median |logit margin| on N(0,1) inputs (correct logit minus
the best other), total weight norm, and strategy from z3's neuron roles (late pointer / event pointer / no pointer,
z2.neuron_features on z2's 4,096 Gaussian probes).  The mystery model's percentile within each group.
Usage: python scripts/z54.py
"""
import glob, numpy as np, torch as th
from batch_train import forward, target
from z2 import load, neuron_features
from z3 import roles
from seeds_clocks import load as zload
dev='cuda' if th.cuda.is_available() else 'cpu'
g=th.Generator(device=dev).manual_seed(54); XG=th.randn(2**18,10,device=dev,generator=g); XU=th.rand(2**18,10,device=dev,generator=g)*2-1
nm=list(np.load('pop1k/seed0.roles.npz')['names']); gp=th.Generator().manual_seed(11); XP=th.randn(4096,10,generator=gp)
def measure(W):
    W=[w.float() for w in W]; N=len(W[0]); out=np.zeros((N,4))
    for c in range(0,N,128):
        Wc=[w[c:c+128].to(dev) for w in W]; n=len(Wc[0])
        with th.no_grad():
            for j,X in enumerate((XG,XU)):
                ok=th.zeros(n,device=dev); mg=[]
                for b in range(0,len(X),2**15):
                    x=X[b:b+2**15]; lg=forward(*Wc,x[None].expand(n,-1,-1)); y=target(x)[None,:,None].expand(n,-1,1)
                    ok+=(lg.argmax(-1)==y[...,0]).float().sum(1)
                    if j==0: mg.append((lg.gather(-1,y)[...,0]-lg.scatter(-1,y,-1e30).max(-1).values).abs())
                out[c:c+n,j]=(ok/len(X)).cpu().numpy()
                if j==0: out[c:c+n,2]=th.cat(mg,1).median(1).values.cpu().numpy()
    out[:,3]=np.sqrt(sum((w.reshape(N,-1).double()**2).sum(1).numpy() for w in W))
    r=roles(neuron_features(W,XP),nm); strat=np.where((r==3).any(1),'late',np.where((r==2).any(1),'event','none'))
    return out,strat
groups={}
for f in sorted(glob.glob('pop_unif/b*_seed*.npz'),key=lambda s:int(s.split('/b')[1].split('_')[0])):
    d=np.load(f); groups[f"unif b{int(d['batch'])}"]=[th.tensor(d[f's1000_{k}']) for k in ('Wih','Whh','Wo')]
P,_=load('pop1k/seed0.npz'); groups['zoo recipe (1,024)']=[p.float() for p in P]
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]; Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
mz,sz=measure(Wz); names=['mystery','seed0','seed1','seed2','seed3','seed4']
print("zoo models: "+"; ".join(f"{n} acc N {100*m[0]:.2f}% U {100*m[1]:.2f}%, median |margin| {m[2]:.2f}, norm {m[3]:.1f}, {s}" for n,m,s in zip(names,mz,sz)))
print(f"\n{'group':20s} {'n':>5s}  {'acc N(0,1) median (5-95%)':28s} {'acc Unif median (5-95%)':26s} {'median |margin| (5-95%)':24s} {'norm median':>11s}  late/event/none     mystery percentile: accN accU margin norm")
for k,W in groups.items():
    m,s=measure(W); q=lambda v: f"{np.median(v):6.2f} ({np.percentile(v,5):5.2f}-{np.percentile(v,95):5.2f})"
    pct=[100*(m[:,j]<mz[0,j]).mean() for j in range(4)]
    print(f"{k:20s} {len(m):5d}  {q(100*m[:,0]):28s} {q(100*m[:,1]):26s} {q(m[:,2]):24s} {np.median(m[:,3]):11.1f}  {(s=='late').mean():.2f}/{(s=='event').mean():.2f}/{(s=='none').mean():.2f}    "
          +" ".join(f"{p:5.1f}" for p in pct)+f"   best acc N {100*m[:,0].max():.2f}%",flush=True)
    if k.startswith('unif'):
        late=s=='late'
        if late.any(): print(f"{'':20s} late-pointer networks: {late.sum()}, acc N median {100*np.median(m[late,0]):.2f}%, median |margin| median {np.median(m[late,2]):.2f}")
