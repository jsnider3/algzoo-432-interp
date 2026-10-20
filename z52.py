"""Independent check of ChatGPT round 6's larger-budget global estimators G(B, K): B = n^3 equal-probability outer boxes
(z46's box-centre rule) x K conditional-CDF S slices x 720 MF3 arrangements, two rows per cell (MF4 and MF3 anchors).
Here: rows built from z46's rule, the prefix trie built with integer keys (np.unique per depth), every network
evaluated on the GPU through the trie (checked against the full forward on 32 networks, and against z46's caches
where they exist), the cell features summarized per network as ChatGPT's summarize() describes (logit-scale
context; ten cell features averaged globally, per S slice and per value of each outer-box digit), 20 weight
statistics appended, and the readout fitted by z49's nested ridge / RBF search (8 outer folds, i mod 8 and one random
permutation; zoo from a model refitted on all 1,024).  Cost: ChatGPT's formula (prefix MACs and scalar ops, 70 ops
per cell for the features when B > 1, a 5,000-op allowance, and an RBF readout over 1,024 supports).
Saves pop1k/seed0_z52_b<n>_k<K>.npz (summary features, ChatGPT's cache layout).  Usage: python z52.py [n,K ...]
"""
import sys, itertools, time, numpy as np, torch as th
from batch_train import forward
from z2 import load
from seeds_clocks import load as zload
from scipy.special import ndtr, ndtri, betaincinv
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
dev='cuda' if th.cuda.is_available() else 'cpu'
DES=[tuple(map(int,a.split(','))) for a in sys.argv[1:]] or [(1,8),(2,2),(2,8),(3,4),(3,8),(4,8),(5,8)]
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
Wall=[th.cat([a,b]) for a,b in zip(P,Wz)]; NT=Wall[0].shape[0]; acc=np.load('pop1k/reference.npz')['acc']; pq=acc*(1-acc); pop=np.arange(1024)
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
ARR=np.array(list(itertools.permutations(range(10),3))); A=len(ARR); QOF={0:7,7:0,1:3,3:1,2:5,5:2,4:6,6:4}
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; QREP=np.array([f[QOF[f.index(r[2])]] for f,r in zip(free,ARR)])
def rows(NB,K):
    out=[]; ai=np.arange(A)[:,None]; ki=np.arange(K)[None,:]
    for b in range(NB**3):
        half=[(b//NB**i)%NB for i in range(3)]; cen=[betaincinv(*ab,(h+.5)/NB) for ab,h in zip([(7,4),(1,3),(2,1)],half)]
        fq=cen[0]; fr=fq+(1-fq)*cen[1]; fm=fr+(1-fr)*cen[2]; Q,R,M=ndtri([fq,fr,fm]); s=ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K)
        X=np.empty((A,K,2,10)); X[:,:,0]=mu(Q); X[:,:,1]=mu(R)
        for r in (0,1): X[ai,ki,r,ARR[:,0,None]]=M; X[ai,ki,r,ARR[:,1,None]]=s; X[ai,ki,r,ARR[:,2,None]]=R
        X[ai,ki,0,QREP[:,None]]=Q; out.append(X.astype(np.float32).reshape(-1,10))
    return np.concatenate(out)                     # row 2c = MF4, 2c+1 = MF3 of cell c = (b*720 + a)*K + k
def trie(x):
    vals,tok=np.unique(x,return_inverse=True); tok=tok.reshape(x.shape); par=np.zeros(len(x),np.int64); lev=[]
    for t in range(10):
        key=par*len(vals)+tok[:,t]; u,par=np.unique(key,return_inverse=True); lev.append((u//len(vals),vals[u%len(vals)]))
    return lev,par,len(vals)
def stats(W):
    wi,Am,O=[w.numpy().astype(float) for w in W]; wi=wi.reshape(len(wi),-1)
    F=np.stack([wi,np.diagonal(Am,axis1=1,axis2=2),np.log(np.linalg.norm(Am,axis=2)+1e-9),np.log(np.linalg.norm(Am,axis=1)+1e-9),np.log(np.linalg.norm(O,axis=1)+1e-9)],2)
    return np.concatenate([F.mean(1),F.std(1),F.min(1),F.max(1)],1)
WS=stats(Wall)
GAM=[.003,.01,.03,.1,.3,1.]; ALP=np.logspace(-5,3,9)
def fold_preds(x,tr,va):
    m=x[tr].mean(0); sd=np.maximum(x[tr].std(0),1e-6); a,b=(x[tr]-m)/sd,(x[va]-m)/sd; y=acc[tr]-acc[tr].mean(); out=[]
    ev,V=eigh(a.T@a); q=V.T@(a.T@y); out.append(((b@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None]*len(tr))))
    D=cdist(a,a,'sqeuclidean'); Dv=cdist(b,a,'sqeuclidean')
    for g in GAM:
        ev,V=eigh(np.exp(-g/x.shape[1]*D)); q=V.T@y; out.append(((np.exp(-g/x.shape[1]*Dv)@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None])))
    return acc[tr].mean()+np.concatenate(out,1)
def nested(x,fold):
    pred=np.zeros(1024)
    for f in range(8):
        tr,va=pop[fold!=f],pop[fold==f]; inner=(np.arange(len(tr))*4//len(tr))[np.argsort(np.argsort(tr*7919%1024))]
        loss=sum(((fold_preds(x,tr[inner!=j],tr[inner==j])-acc[tr[inner==j],None])**2).sum(0) for j in range(4))
        pred[va]=fold_preds(x,tr,va)[:,int(loss.argmin())]
    loss=sum(((fold_preds(x,pop[fold!=f],pop[fold==f])-acc[pop[fold==f],None])**2).sum(0) for f in range(8))
    return pred, fold_preds(x,pop,np.arange(1024,NT))[:,int(loss.argmin())]
LEDGER={(1,8):(4196.37,.3809),(2,2):(8792.17,.3559),(2,8):(33386.88,.2666),(3,4):(57062.76,.2478),(3,8):(112333.23,.2273),(4,8):(266027.59,.2164),(5,8):(514895.74,.2122)}
print("design   cells   rows  network FE (ChatGPT)  total FE   RMS i mod 8 | perm   mean   MSE/s   ChatGPT RMS   zoo errors (mystery, seed0-4)",flush=True)
for NB,K in DES:
    t0=time.time(); x=rows(NB,K); n=len(x); C=n//2; lev,leaf,nv=trie(x); cnt=[len(p) for p,_ in lev]
    mac=256*sum(cnt[:-1])+16*nv+160*cnt[-1]; oth=16*sum(cnt[1:])+16*sum(cnt)+20*cnt[-1]; net=(mac+oth)/2880
    L=[(th.tensor(p,device=dev),th.tensor(v,device=dev)) for p,v in lev]; lf=th.tensor(leaf,device=dev)
    y=th.tensor(np.repeat(np.tile(np.repeat(ARR[:,1],K),NB**3),2),device=dev)
    feats=[]; ch=max(1,int(2.5e8//(n*16*4)))
    cache={(1,8):'pop1k/seed0_z46_cells_b1.npz',(2,8):'pop1k/seed0_z46_cells.npz'}.get((NB,K)); cz=np.load(cache) if cache else None; bad=0; tbad=0
    for c in range(0,NT,ch):
        sl=slice(c,min(NT,c+ch)); W=[w[sl].to(dev) for w in Wall]; N=W[0].shape[0]
        with th.no_grad():
            h=None
            for t,(p,v) in enumerate(L):
                inp=v[None,:,None]*W[0][:,None,:]; h=th.relu(inp) if t==0 else th.relu(th.bmm(h,W[1].transpose(1,2))[:,p]+inp)
            lg=th.bmm(h,W[2].transpose(1,2))[:,lf]; del h
            yy=y[None,:,None].expand(N,-1,1); ok=(lg.argmax(-1)==yy[...,0]); mg=lg.gather(-1,yy)[...,0]-lg.scatter(-1,yy,-1e30).max(-1).values
            if c<32:   # trie vs full forward
                full=th.cat([forward(*W,th.tensor(x[st:st+2**17],device=dev)[None].expand(N,-1,-1)).argmax(-1) for st in range(0,n,2**17)],1)
                tbad+=int(((full==y[None])!=ok).sum())
            i4,i3=ok[:,0::2].double(),ok[:,1::2].double(); m4,m3=mg[:,0::2].double(),mg[:,1::2].double()
            if cz is not None: bad+=int((i4.cpu().numpy()!=cz['I4'][sl]).sum()+(i3.cpu().numpy()!=cz['I3'][sl]).sum())
            s=m4.pow(2).mean(1,keepdim=True).sqrt(); a,b=m4/s,m3/s
            f=th.stack([i4,i3,a,b,a.abs(),b.abs(),a/(.1+a.abs()),b/(.1+b.abs()),a/(.03+a.abs()),b/(.03+b.abs())],-1).view(N,NB,NB,NB,A,K,10)
            out=[s,s.log(),m4.mean(1,keepdim=True),m3.mean(1,keepdim=True),f.mean((1,2,3,4,5))]+[f.mean((1,2,3,4))[:,k] for k in range(K)]
            if NB>1:   # digit j of the box index b = d0 + NB d1 + NB^2 d2 sits at view axis 3 - j
                for j in range(3): out+=[f.select(3-j,v).mean(tuple(range(1,5))) for v in range(NB)]
            feats.append(th.cat(out,1).cpu().numpy())
    Fs=np.concatenate(feats); np.savez(f'pop1k/seed0_z52_b{NB}_k{K}.npz',NB=NB,K=K,summary=Fs)
    X=np.c_[WS,Fs]; d=X.shape[1]; tot=net+((40+30*(NB>1))*C+5000+2*1024*d+34*1024+2*d+1+2)/2880
    r=[]; zo=None
    for fold in (pop%8,np.random.default_rng(100).permutation(1024)%8):
        p,z_=nested(X,fold); r.append(p); zo=zo if zo is not None else z_
    e=100*(r[0]-acc[:1024]); rms=[100*np.sqrt(((p-acc[:1024])**2).mean()) for p in r]; ratio=(rms[0]/100)**2/np.mean(pq[:1024]/tot)
    print(f"G({NB**3},{K}) {C:7d} {n:7d}  {net:10.2f} ({LEDGER[(NB,K)][0]-((40+30*(NB>1))*C+5000+2*1024*d+34*1024+2*d+3)/2880:9.2f})  {tot:9.2f}  {rms[0]:.4f} | {rms[1]:.4f}  {e.mean():+.4f}  {ratio:6.3f}   {LEDGER[(NB,K)][1]:.4f}   "
          +" ".join(f"{v:+.3f}" for v in 100*(zo-acc[1024:]))+f"   [trie vs full forward bit mismatches (32 nets) {tbad}; vs z46 cache {bad if cz is not None else 'n/a'}; {time.time()-t0:.0f}s]",flush=True)
