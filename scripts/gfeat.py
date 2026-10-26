"""Shared pieces of ChatGPT round 6's global estimator G(B, K), for any set of networks (from z52.py, which checked
them against the round-6 caches): the probe rows, the prefix trie, the per-network cell summary, the 20 raw weight
statistics, the cost formula, and z49's nested ridge / RBF readout generalized to any training set and labels.
"""
import itertools, numpy as np, torch as th
from scipy.special import ndtr, ndtri, betaincinv
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
dev='cuda' if th.cuda.is_available() else 'cpu'

def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
ARR=np.array(list(itertools.permutations(range(10),3))); A=len(ARR); QOF={0:7,7:0,1:3,3:1,2:5,5:2,4:6,6:4}
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; QREP=np.array([f[QOF[f.index(r[2])]] for f,r in zip(free,ARR)])

def rows(NB,K):
    """row 2c = MF4, 2c+1 = MF3 of cell c = (b*720 + a)*K + k"""
    out=[]; ai=np.arange(A)[:,None]; ki=np.arange(K)[None,:]
    for b in range(NB**3):
        half=[(b//NB**i)%NB for i in range(3)]; cen=[betaincinv(*ab,(h+.5)/NB) for ab,h in zip([(7,4),(1,3),(2,1)],half)]
        fq=cen[0]; fr=fq+(1-fq)*cen[1]; fm=fr+(1-fr)*cen[2]; Q,R,M=ndtri([fq,fr,fm]); s=ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K)
        X=np.empty((A,K,2,10)); X[:,:,0]=mu(Q); X[:,:,1]=mu(R)
        for r in (0,1): X[ai,ki,r,ARR[:,0,None]]=M; X[ai,ki,r,ARR[:,1,None]]=s; X[ai,ki,r,ARR[:,2,None]]=R
        X[ai,ki,0,QREP[:,None]]=Q; out.append(X.astype(np.float32).reshape(-1,10))
    return np.concatenate(out)

def trie(x):
    vals,tok=np.unique(x,return_inverse=True); tok=tok.reshape(x.shape); par=np.zeros(len(x),np.int64); lev=[]
    for t in range(10):
        key=par*len(vals)+tok[:,t]; u,par=np.unique(key,return_inverse=True); lev.append((u//len(vals),vals[u%len(vals)]))
    return lev,par,len(vals)

def stats(wi,Am,O):
    """20 raw weight statistics: mean, SD, min, max over hidden units of W_ih, diag W_hh, log row / column norms of
    W_hh, log column norms of W_o (numpy arrays (N,16), (N,16,16), (N,10,16))"""
    F=np.stack([wi,np.diagonal(Am,axis1=1,axis2=2),np.log(np.linalg.norm(Am,axis=2)+1e-9),np.log(np.linalg.norm(Am,axis=1)+1e-9),np.log(np.linalg.norm(O,axis=1)+1e-9)],2)
    return np.concatenate([F.mean(1),F.std(1),F.min(1),F.max(1)],1)

def summary(W,NB,K,cells=False):
    """W = [Wih (N,16), Whh (N,16,16), Wo (N,10,16)] torch float32.  Returns the per-network summary (N, d), the
    network-row cost in FE (prefix MACs and scalar ops / 2880) and the number of cells; with cells=True also the
    cell arrays I4, I3 (bool) and m4, m3 (float32), each (N, C)."""
    x=rows(NB,K); n=len(x); C=n//2; lev,leaf,nv=trie(x); cnt=[len(p) for p,_ in lev]
    net=(256*sum(cnt[:-1])+16*nv+160*cnt[-1]+16*sum(cnt[1:])+16*sum(cnt)+20*cnt[-1])/2880
    L=[(th.tensor(p,device=dev),th.tensor(v,device=dev)) for p,v in lev]; lf=th.tensor(leaf,device=dev)
    y=th.tensor(np.repeat(np.tile(np.repeat(ARR[:,1],K),NB**3),2),device=dev)
    NT=W[0].shape[0]; feats=[]; cl=[]; ch=max(1,int(2.5e8//(n*16*4)))
    for c in range(0,NT,ch):
        sl=slice(c,min(NT,c+ch)); Wc=[w[sl].to(dev).float() for w in W]; N=Wc[0].shape[0]
        with th.no_grad():
            h=None
            for t,(p,v) in enumerate(L):
                inp=v[None,:,None]*Wc[0][:,None,:]; h=th.relu(inp) if t==0 else th.relu(th.bmm(h,Wc[1].transpose(1,2))[:,p]+inp)
            lg=th.bmm(h,Wc[2].transpose(1,2))[:,lf]; del h
            yy=y[None,:,None].expand(N,-1,1); ok=(lg.argmax(-1)==yy[...,0]); mg=lg.gather(-1,yy)[...,0]-lg.scatter(-1,yy,-1e30).max(-1).values
            i4,i3=ok[:,0::2].double(),ok[:,1::2].double(); m4,m3=mg[:,0::2].double(),mg[:,1::2].double()
            if cells: cl.append([i4.bool().cpu().numpy(),i3.bool().cpu().numpy(),m4.float().cpu().numpy(),m3.float().cpu().numpy()])
            s=m4.pow(2).mean(1,keepdim=True).sqrt(); a,b=m4/s,m3/s
            f=th.stack([i4,i3,a,b,a.abs(),b.abs(),a/(.1+a.abs()),b/(.1+b.abs()),a/(.03+a.abs()),b/(.03+b.abs())],-1).view(N,NB,NB,NB,A,K,10)
            out=[s,s.log(),m4.mean(1,keepdim=True),m3.mean(1,keepdim=True),f.mean((1,2,3,4,5))]+[f.mean((1,2,3,4))[:,k] for k in range(K)]
            if NB>1:
                for j in range(3): out+=[f.select(3-j,v).mean(tuple(range(1,5))) for v in range(NB)]
            feats.append(th.cat(out,1).cpu().numpy())
    F=np.concatenate(feats)
    if cells: return F,net,C,[np.concatenate([c[i] for c in cl]) for i in range(4)]
    return F,net,C

def cost(net,C,NB,d,n_sup):
    """ChatGPT's total: network rows + 40 (70 when B > 1) ops per cell + 5,000-op allowance + RBF readout over n_sup
    supports with d features"""
    return net+((40+30*(NB>1))*C+5000+2*n_sup*d+34*n_sup+2*d+3)/2880

GAM=[.003,.01,.03,.1,.3,1.]; ALP=np.logspace(-5,3,9)
def fold_preds(x,y,tr,va):
    """predictions on va for every (family, gamma, alpha), trained on tr: ridge, then RBF per gamma"""
    m=x[tr].mean(0); sd=np.maximum(x[tr].std(0),1e-6); a,b=(x[tr]-m)/sd,(x[va]-m)/sd; yt=y[tr]-y[tr].mean(); out=[]
    ev,V=eigh(a.T@a); q=V.T@(a.T@yt); out.append(((b@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None]*len(tr))))
    D=cdist(a,a,'sqeuclidean'); Dv=cdist(b,a,'sqeuclidean')
    for g in GAM:
        ev,V=eigh(np.exp(-g/x.shape[1]*D)); q=V.T@yt; out.append(((np.exp(-g/x.shape[1]*Dv)@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None])))
    return y[tr].mean()+np.concatenate(out,1)

def pick(x,y,tr,nf=4,salt=7919):
    """hyperparameter column chosen by nf-fold CV inside tr"""
    inner=(np.arange(len(tr))*nf//len(tr))[np.argsort(np.argsort(tr*salt%max(len(y),1)))]
    loss=sum(((fold_preds(x,y,tr[inner!=j],tr[inner==j])-y[tr[inner==j],None])**2).sum(0) for j in range(nf))
    return int(loss.argmin())

def crossfit(x,y,idx,fold):
    """nested cross-fitted predictions for networks idx (fold labels per idx entry, 8 folds)"""
    pred=np.zeros(len(idx))
    for f in range(fold.max()+1):
        tr,va=idx[fold!=f],idx[fold==f]; pred[fold==f]=fold_preds(x,y,tr,va)[:,pick(x,y,tr)]
    return pred

def fit_predict(x,y,tr,te):
    """hyperparameters by 8-fold CV on tr, refit on all of tr, predict te"""
    fold=np.arange(len(tr))%8; loss=sum(((fold_preds(x,y,tr[fold!=f],tr[fold==f])-y[tr[fold==f],None])**2).sum(0) for f in range(8))
    return fold_preds(x,y,tr,te)[:,int(loss.argmin())]
