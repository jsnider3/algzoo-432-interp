"""Was the 1,024-network population trained like the zoo?  Compares pop1k/seed0.npz with the zoo's own training logs
for the 16-neuron, length-10 2nd-argmax seeds 0-4 (ARC's public bucket, alg/zoo/{train,val}_2nd_argmax_16_10_1073741824_<seed>.jsonl;
one line per 1,024 steps: means over those steps of the training-batch loss, accuracy and learning rate).
  1. learning rate: the zoo's logged window means against batch_train.lr_mult averaged over the same windows;
  2. training accuracy and loss at the population's snapshots (2^20 ... 2^30 sequences): population quantiles (each
     network on 2^16 fresh N(0,1) sequences) and where each zoo seed's logged value falls among them;
  3. the population's mean training curve (pop1k/train_log.txt) against the zoo seeds' mean;
  4. logit scale (RMS of the MF4-anchor margin over z46's 1,440 one-box K = 1 probe rows) along training, with the
     zoo seeds and the mystery model (weights.npz) placed against the final population.
Usage: python z50.py <dir with train_<s>.jsonl and val_<s>.jsonl, s = 0..4>
"""
import sys, json, itertools, re, numpy as np, torch as th
from batch_train import forward, target, lr_mult
from seeds_clocks import load as zload
from scipy.special import ndtr, ndtri, betaincinv
D=sys.argv[1]; dev='cuda' if th.cuda.is_available() else 'cpu'; NSTEP=2**20
logs=[[json.loads(l) for l in open(f'{D}/train_{s}.jsonl')] for s in range(5)]
val=[[json.loads(l) for l in open(f'{D}/val_{s}.jsonl')] for s in range(5)]
step=np.array([r['step'] for r in logs[0]]); assert all([r['step'] for r in L]==list(step) for L in logs)
zacc=np.array([[r['accuracy'] for r in L] for L in logs]); zloss=np.array([[r['loss'] for r in L] for L in logs]); zlr=np.array([[r['lr'] for r in L] for L in logs])
# 1. learning rate (window means of lr after each step; the scheduler steps once per optimizer step)
ours=np.array([0.01*np.mean([lr_mult(t,NSTEP) for t in range(s-1023,s+1)]) for s in step])
ours0=np.array([0.01*np.mean([lr_mult(t,NSTEP) for t in range(s-1024,s)]) for s in step])
print(f"1. learning rate: {len(step)} windows of 1,024 steps; max relative difference zoo vs batch_train {np.max(np.abs(zlr/ours-1)):.2e} "
      f"(lr before each step: {np.max(np.abs(zlr/ours0-1)):.2e}); seeds identical to each other: {np.ptp(zlr,0).max():.1e}")
# 2. population at its snapshots
z=np.load('pop1k/seed0.npz'); snaps=sorted(int(k[1:-4]) for k in z.files if re.fullmatch(r's\d+_Wih',k))
g=th.Generator(device=dev).manual_seed(50); X=th.randn(2**16,10,device=dev,generator=g); Y=target(X)
def acc_loss(W):
    a,l=[],[]
    with th.no_grad():
        for c in range(0,1024,256):
            Wc=[w[c:c+256] for w in W]; A_=th.zeros(len(Wc[0]),device=dev); L_=th.zeros_like(A_)
            for b in range(0,len(X),8192):
                lg=forward(*Wc,X[b:b+8192][None].expand(len(Wc[0]),-1,-1)); y=Y[b:b+8192][None].expand(len(Wc[0]),-1)
                A_+=(lg.argmax(-1)==y).float().sum(1); L_+=th.nn.functional.cross_entropy(lg.transpose(1,2),y,reduction='none').sum(1)
            a.append(A_/len(X)); l.append(L_/len(X))
    return th.cat(a).cpu().numpy(), th.cat(l).cpu().numpy()
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
ARR=np.array(list(itertools.permutations(range(10),3))); QOF={0:7,7:0,1:3,3:1,2:5,5:2,4:6,6:4}
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; QREP=np.array([f[QOF[f.index(r[2])]] for f,r in zip(free,ARR)])
cen=[betaincinv(*ab,0.5) for ab in [(7,4),(1,3),(2,1)]]; fq=cen[0]; fr=fq+(1-fq)*cen[1]; fm=fr+(1-fr)*cen[2]; Q,R,M=ndtri([fq,fr,fm]); S_=ndtri(fr+(fm-fr)*.5)
P4=np.full((720,10),mu(Q)); P4[np.arange(720),ARR[:,0]]=M; P4[np.arange(720),ARR[:,1]]=S_; P4[np.arange(720),ARR[:,2]]=R; P4[np.arange(720),QREP]=Q
P4=th.tensor(P4,dtype=th.float32,device=dev); Y4=th.tensor(ARR[:,1],device=dev)
def scale(W):
    with th.no_grad():
        lg=forward(*W,P4[None].expand(len(W[0]),-1,-1)); y=Y4[None,:,None].expand(len(W[0]),-1,1)
        m=lg.gather(-1,y)[...,0]-lg.scatter(-1,y,-1e30).max(-1).values
    return m.pow(2).mean(1).sqrt().cpu().numpy()
Zw=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]; Wz=[th.tensor(np.stack([w[i] for w in Zw]),dtype=th.float32,device=dev) for i in range(3)]
print("2. per-network training accuracy and loss at the population's snapshots; zoo seed = its logged window mean at that step")
print("   step      seqs  pop acc 5/25/50/75/95%                 zoo seeds 0-4 acc (population percentile)          pop loss median  zoo loss")
sc={}
for s in snaps:
    W=[th.tensor(z[f's{s}_{k}'],dtype=th.float32,device=dev) for k in ('Wih','Whh','Wo')]; a,l=acc_loss(W); sc[s]=scale(W)
    i=int(np.where(step==s)[0][0]) if s in step else None
    if i is None: continue
    za=zacc[:,i]; pct=[100*(a<v).mean() for v in za]
    print(f"   {s:7d}  2^{int(np.log2(s*1024))}  {' '.join(f'{100*v:5.1f}' for v in np.percentile(a,[5,25,50,75,95]))}   "
          f"{' '.join(f'{100*v:5.1f}({p:3.0f})' for v,p in zip(za,pct))}   {np.median(l):.4f}   {' '.join(f'{v:.3f}' for v in zloss[:,i])}")
fin=snaps[-1]; a_fin,_=acc_loss([th.tensor(z[f's{fin}_{k}'],dtype=th.float32,device=dev) for k in ('Wih','Whh','Wo')])
zv=[np.mean([r['accuracy'] for r in V]) for V in val]; ref=np.load('pop1k/reference.npz')['acc']
print(f"   zoo final validation accuracy (its val log) {' '.join(f'{100*v:.2f}' for v in zv)}; our reference for weights_seed0-4 {' '.join(f'{100*v:.2f}' for v in ref[1025:])}")
# 3. mean curves
pl=[(int(m.group(1)),float(m.group(2)),float(m.group(3))) for m in (re.search(r'step (\d+)/\d+ mean loss ([\d.]+) .*mean acc ([\d.]+)',l) for l in open('pop1k/train_log.txt')) if m]
print("3. mean training accuracy: population (train_log, mean of 1,024) vs zoo (mean of 5 seeds, window at the same step)")
for st,lo,ac in pl:
    if st in (4096,16384,65536,262144,524288,1048576) and st in step:
        i=int(np.where(step==st)[0][0]); print(f"   step {st:7d}: population {100*ac:.2f}% loss {lo:.4f}   zoo seeds {100*zacc[:,i].mean():.2f}% loss {zloss[:,i].mean():.4f}  (range {100*zacc[:,i].min():.1f}-{100*zacc[:,i].max():.1f}%)")
# 4. logit scale
print("4. logit scale (RMS MF4-anchor margin over 1,440 probe rows), population median (5-95%) by snapshot:")
for s in snaps: print(f"   2^{int(np.log2(s*1024))} sequences: {np.median(sc[s]):6.2f} ({np.percentile(sc[s],5):.2f}-{np.percentile(sc[s],95):.2f})")
zs=scale(Wz); names=['mystery','seed0','seed1','seed2','seed3','seed4']
print("   final zoo models: "+"  ".join(f"{n} {v:.2f} ({100*(sc[fin]<v).mean():.1f}th pct)" for n,v in zip(names,zs)))
r=np.polyfit(np.log(a_fin),np.log(sc[fin]),1); print(f"   population: log scale vs log accuracy slope {r[0]:.1f}; scale predicted at the mystery's accuracy {np.exp(np.polyval(r,np.log(ref[1024]))):.2f}")
