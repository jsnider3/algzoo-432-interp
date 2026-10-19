"""Independent check of ChatGPT round 5's two-level estimator, and its per-network pieces on all 1,030 networks.
  p_hat = Q_fine^MF3(I3) + Q_coarse^MF4(F4 - I3),  F4 = clip(I4 + sum_j (E_j - I4), 0, 1)  (clamped first-order model,
  Gauss-Legendre q = 8 per lower input).  Fine part: the exact-S MF3 anchor, outer order 6 (Beta(8,3) x Beta(2,1)),
  720 S-lines per node by lines.trace_line.  Coarse part: 2 nodes per MF4 outer coordinate (Beta(7,4), Beta(1,3),
  Beta(2,1)), K = 8 conditional-CDF S midpoints, 8 balanced (R,Q) placements per (M,S) class (720 arrangements);
  50 rows per configuration, 2,304,000 forwards.
Also: the same correction against the MF4 anchor itself over all 5,040 arrangements at one outer node (n1/K8/q8), the
extreme-node screen (z42), and the coarse anchor integrals alone at one outer node (n1/K8, 2 rows per configuration).
Saves pop1k/seed0_z41.npz; prints the dev (0::16) / held-out (8::16) / zoo scores.  Usage: python z41.py
"""
import itertools, time, numpy as np, torch as th
from batch_train import forward
from z2 import load
from seeds_clocks import load as zload
from lines import trace_line
from scipy.special import ndtr, ndtri, roots_jacobi
dev='cuda'; q=8; K=8
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
Wall=[th.cat([a,b]) for a,b in zip(P,Wz)]; NT=Wall[0].shape[0]
def rule(n,a,b):
    x,w=roots_jacobi(n,b-1,a-1); return (x+1)/2, w/w.sum()
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
PAIRS=[(0,7),(7,0),(1,3),(3,1),(2,5),(5,2),(4,6),(6,4)]
ALL=np.array(list(itertools.permutations(range(10),4)))
BAL=np.array([r for r in ALL if ([x for x in range(10) if x not in r[:2]].index(r[2]),[x for x in range(10) if x not in r[:2]].index(r[3])) in PAIRS])
uq,wq=rule(q,1,1); wq_t=th.tensor(wq,dtype=th.float32,device=dev)
keys=['I4c','I3c','brk','fix','brk_g','fix_g','flag','c44','I4c1','I3c1']; F={k:np.zeros(NT) for k in keys}
def run(n,arr,mode,W,sl):
    """mode 'full': the two-level rows; 'mf4': F4 - I4 only; 'anchors': I4 and I3 rows only."""
    N=W[0].shape[0]; A=len(arr); low=np.array([[i for i in range(10) if i not in r] for r in arr])
    rules=[rule(n,7,4),rule(n,1,3),rule(n,2,1)]; rows=2 if mode=='anchors' else 2+6*q; bs=720 if mode=='anchors' else 90
    acc={k:th.zeros(N,dtype=th.float64,device=dev) for k in keys}
    for ij in itertools.product(range(n),repeat=3):
        u=[rules[t][0][j] for t,j in enumerate(ij)]; wt=np.prod([rules[t][1][j] for t,j in enumerate(ij)])
        fq=u[0]; fr=fq+(1-fq)*u[1]; fm=fr+(1-fr)*u[2]; Q,R,M=ndtri(fq),ndtri(fr),ndtri(fm)
        s=ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K); zn=ndtri(fq*uq)
        for st in range(0,A,bs):
            aa=arr[st:st+bs]; ll=low[st:st+bs]; b=len(aa); bi=np.arange(b)[:,None]; ki=np.arange(K)[None,:]
            base=np.full((b,K,10),mu(Q))
            base[bi,ki,aa[:,0,None]]=M; base[bi,ki,aa[:,1,None]]=s; base[bi,ki,aa[:,2,None]]=R; base[bi,ki,aa[:,3,None]]=Q
            X=np.repeat(base[:,:,None,:],rows,axis=2)
            if mode!='anchors':
                for j in range(6):
                    for r in range(q): X[bi,ki,1+j*q+r,ll[:,j,None]]=zn[r]
            m3=np.full((b,K,10),mu(R)); m3[bi,ki,aa[:,0,None]]=M; m3[bi,ki,aa[:,1,None]]=s; m3[bi,ki,aa[:,2,None]]=R; X[:,:,-1]=m3
            with th.no_grad():
                I=(forward(*W,th.tensor(X.reshape(-1,10),dtype=th.float32,device=dev)[None].expand(N,-1,-1)).argmax(-1).view(N,b,K,rows)==th.tensor(aa[:,1],device=dev)[None,:,None,None]).float()
            w=wt/(A*K); i0=I[...,0]; i3=I[...,-1]
            if mode=='anchors':
                acc['I4c1']+=w*i0.sum((1,2)).double(); acc['I3c1']+=w*i3.sum((1,2)).double(); continue
            top=I[...,1:1+6*q].reshape(N,b,K,6,q); d=((i0+((top*wq_t).sum(-1)-i0[...,None]).sum(-1)).clamp(0,1)-i0).double()
            if mode=='mf4': acc['c44']+=w*d.sum((1,2)); continue
            g=((top[...,-1]!=i0[...,None])|(top[...,0]!=i0[...,None])).any(-1).double()      # extreme-node screen
            acc['I4c']+=w*i0.sum((1,2)).double(); acc['I3c']+=w*i3.sum((1,2)).double()
            acc['brk']+=w*d.clamp(max=0).sum((1,2)); acc['fix']+=w*d.clamp(min=0).sum((1,2))
            acc['brk_g']+=w*(d*g).clamp(max=0).sum((1,2)); acc['fix_g']+=w*(d*g).clamp(min=0).sum((1,2)); acc['flag']+=g.sum((1,2))/(n**3*A*K)
    for k in {'full':['I4c','I3c','brk','fix','brk_g','fix_g','flag'],'mf4':['c44'],'anchors':['I4c1','I3c1']}[mode]: F[k][sl]=acc[k].cpu().numpy()
t0=time.time()
for c in range(0,NT,206):
    sl=slice(c,min(NT,c+206)); W=[w[sl].to(dev) for w in Wall]
    run(2,BAL,'full',W,sl); run(1,ALL,'mf4',W,sl); run(1,BAL,'anchors',W,sl)
    print(f"GPU part, networks {c}-{sl.stop}: {time.time()-t0:.0f}s",flush=True)
# exact-S MF3 anchor, outer order 6
ARR=np.array(list(itertools.permutations(range(10),3))); L=len(ARR); p=np.ascontiguousarray(ARR[:,1],dtype=np.int64)
ur,wr=rule(6,8,3); ug,wg=rule(6,2,1); anc=np.zeros(NT); cost=np.zeros(NT)
for i in range(NT):
    Wn=[np.ascontiguousarray(Wall[j][i].numpy(),dtype=np.float64) for j in range(3)]
    for a,wa in zip(ur,wr):
        for g_,wb in zip(ug,wg):
            b=a+(1-a)*g_; R,M=ndtri(a),ndtri(b); m=-np.exp(-R*R/2)/np.sqrt(2*np.pi)/a
            x=np.full((L,10),m); x[np.arange(L),ARR[:,0]]=M; x[np.arange(L),ARR[:,1]]=R; x[np.arange(L),ARR[:,2]]=R
            pr=np.empty(L); rem=np.arange(L); cap=512
            while len(rem):          # overflowing lines retry with a larger cap; discarded attempts are charged
                pp,cc,ov=trace_line(*Wn,np.ascontiguousarray(x[rem]),p[rem],np.full(len(rem),R),np.full(len(rem),M),p[rem],cap)
                cost[i]+=cc.sum(); ok=ov==0; pr[rem[ok]]=pp[ok]; rem=rem[~ok]; cap*=2
            anc[i]+=wa*wb*pr.mean()
print(f"exact anchor done: {time.time()-t0:.0f}s",flush=True)
np.savez('pop1k/seed0_z41.npz',anchor6=anc,anchor6_cost=cost,**F)
ref=np.load('pop1k/reference.npz'); acc=ref['acc']; pq=acc*(1-acc)
c43=F['I4c']-F['I3c']+F['brk']+F['fix']; dv=np.arange(0,1024,16); hd=np.arange(8,1024,16); zo=np.arange(1024,1030)
def line(name,est,C=None):
    e=100*(est-acc); s=f"  {name:44s}"
    for lab,ix in (('dev',dv),('held',hd)):
        s+=f" {lab} RMS {np.sqrt((e[ix]**2).mean()):.4f} mean {e[ix].mean():+.4f}"
        if C is not None: s+=f" (MSE/sampling {(e[ix]**2).mean()/(1e4*np.mean(pq[ix]/C[ix])):.1f})"
    print(s+"  zoo "+" ".join(f"{v:+.4f}" for v in e[zo]))
print("ChatGPT round 5 two-level estimator, re-implemented (points; zoo order mystery, seed0..4)")
line('free MF3 anchor (reference mf3) + correction',ref['mf3']+c43)
line('free MF4 anchor (reference mf4) + n1/5,040 corr.',ref['mf4']+F['c44'])
line('exact MF3 anchor minus reference mf3',anc-ref['mf3']+acc)
line('end to end: exact anchor + correction',anc+c43,cost+2304000)
print(f"  exact anchor MAC-forwards per network: dev {cost[dv].mean():.0f}, held {cost[hd].mean():.0f}; per MF3 node {cost[dv].mean()/36:.0f}; total dev {cost[dv].mean()+2304000:.0f}")
print(f"  coarse I3 integral minus reference mf3 (cancels in F4 - I3): dev mean {100*(F['I3c']-ref['mf3'])[dv].mean():+.3f}")
