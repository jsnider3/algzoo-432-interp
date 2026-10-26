# MF3 / MF4 leakage split by regime (S<0, R<0<S, R>0) and by the wrong answer's kind, all 1024 + zoo
import sys, numpy as np, torch as th
sys.path.insert(0,'.')
from batch_train import forward, target
from z2 import load
from z10 import topk_mf
from seeds_clocks import load as zload
dev='cuda'; m=int(sys.argv[1]) if len(sys.argv)>1 else 19
P,_=load('pop1k/seed0.npz'); Z=[zload(n) for n in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
W=[th.cat([a,b]).to(dev) for a,b in zip(P,Wz)]; N=W[0].shape[0]
x=th.randn(2**m,10,generator=th.Generator().manual_seed(11)).to(dev)
# acc[n, variant(true, MF3, MF4), regime]; kinds[n, variant, regime, kind(max,R,lower)] of wrong answers
acc=th.zeros(N,3,3,device=dev); kinds=th.zeros(N,3,3,3,device=dev); mass=th.zeros(3,device=dev)
with th.no_grad():
    for i in range(0,len(x),4096):
        xc=x[i:i+4096]; y=target(xc); o=xc.argsort(-1,descending=True); S=xc.gather(-1,o[:,1:2])[:,0]; R=xc.gather(-1,o[:,2:3])[:,0]
        reg=th.where(S<0,0,th.where(R<0,1,2)); 
        for r in range(3): mass[r]+=(reg==r).sum()
        for v,xv in enumerate((xc,topk_mf(xc,3),topk_mf(xc,4))):
            a=forward(*W,xv[None].expand(N,-1,-1)).argmax(-1); ok=a==y
            k=th.where(a==o[None,:,0],0,th.where(a==o[None,:,2],1,2))
            for r in range(3):
                acc[:,v,r]+=(ok&(reg==r)[None]).sum(1)
                for c in range(3): kinds[:,v,r,c]+=((~ok)&(reg==r)[None]&(k==c)).sum(1)
n=len(x); acc=acc.cpu().numpy()/n*100; kinds=kinds.cpu().numpy()/n*100; mass=mass.cpu().numpy()/n*100
pop=slice(0,N-6); names=['mystery']+[f'seed{i}' for i in range(5)]
print(f'regime mass (% of inputs): S<0 {mass[0]:.2f}, R<0<S {mass[1]:.2f}, R>0 {mass[2]:.2f};  pool 2^{m}')
for v,vn in ((1,'MF3'),(2,'MF4')):
    L=acc[:,v]-acc[:,0]                                              # leakage per regime (points of total accuracy)
    print(f'\n{vn} leakage by regime (points): population mean / SD / RMS, then zoo')
    for r,rn in enumerate(('S<0','R<0<S','R>0')):
        l=L[pop,r]; print(f'  {rn:6s} {l.mean():+7.3f} {l.std():6.3f} {np.sqrt((l**2).mean()):6.3f}   '+' '.join(f'{nm} {L[N-6+j,r]:+.3f}' for j,nm in enumerate(names)))
    l=L[pop].sum(1); print(f'  {"total":6s} {l.mean():+7.3f} {l.std():6.3f} {np.sqrt((l**2).mean()):6.3f}   '+' '.join(f'{nm} {L[N-6+j].sum():+.3f}' for j,nm in enumerate(names)))
    l=L[pop,2]; print(f'  R>0 only: pop RMS {np.sqrt((l**2).mean()):.3f}')
    D=kinds[:,0]-kinds[:,v]                                          # (true errors - variant errors) by regime and kind = leakage by kind
    for r,rn in enumerate(('S<0','R<0<S','R>0')):
        print(f'  {rn:6s} by wrong answer (true errors minus {vn} errors, pop mean): max {D[pop,r,0].mean():+.3f}  R {D[pop,r,1].mean():+.3f}  lower {D[pop,r,2].mean():+.3f}'
              f'   [true error rate in regime {(mass[r]-acc[pop,0,r].mean())/mass[r]*100:.1f}%, {vn} {(mass[r]-acc[pop,v,r].mean())/mass[r]*100:.1f}%]')
np.savez('pop1k/lw2_seed0.npz',acc=acc,kinds=kinds,mass=mass)
