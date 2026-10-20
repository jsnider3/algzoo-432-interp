"""Deterministic cached margin summaries; every model choice uses training folds.
Run from the extracted project: python /path/aggregates.py --recipe unif --K 1
This script never evaluates sampled inputs or accesses noisy weight replicas.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

GAM = [.003, .01, .03, .1, .3, 1.]
ALP = np.logspace(-5, 3, 9)

def extra_features(c, K):
    """Return tails (15K columns), temporal groups (24K), and scalar-op bounds."""
    m4, m3 = c['m4'].astype(float), c['m3'].astype(float)
    s = np.sqrt((m4*m4).mean(1, keepdims=True))
    a, b = m4/s, m3/s
    i4, i3 = c['I4'], c['I3']
    tail = [v < t for v in (a, b) for t in [-.1, -.03, .01, .03, .1, .3]]
    tail += [abs(a-b), (a-b)**2, i4 != i3]
    T = np.stack(tail, -1).reshape(len(a), 720, K, -1).mean(1).reshape(len(a), -1)
    M, S, R = c['arr'].T
    masks = [M == 9, S == 9, M == S+1, M > S, R > S, abs(M-S) >= 5]
    vals = np.stack([i4, i3, a/(.1+abs(a)), b/(.1+abs(b))], -1)
    vals = vals.reshape(len(a), 720, K, 4)
    H = np.concatenate([vals[:, mask].mean(1).reshape(len(a), -1) for mask in masks], 1)
    # Baseline already computes a,b and the two softsigns. Costs include accumulation.
    ops = {'tails':32*720*K+15*K, 'groups':4*K*sum(m.sum() for m in masks)+24*K}
    return T, H, ops

def load(recipe, K):
    if recipe == 'unif':
        c = np.load(f'pop_unif/cells_k{K}.npz')
        base, y = np.c_[c['stats'], c['F']], c['acc']
        net = float(c['net_fe'])
    else:
        c = np.load(f'pop1k/seed0_z46_cells_b1_k{K}.npz')
        p = np.load('pop1k/seed0.npz'); snap = max(int(k[1:-4]) for k in p.files if k.startswith('s') and k.endswith('_Wih'))
        released = [np.load('weights.npz')]+[np.load(f'weights_seed{i}.npz') for i in range(5)]
        keys = ['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight']
        W = [np.concatenate([p[f's{snap}_{key}'],np.stack([z[name][:,0] if j == 0 else z[name] for z in released])]).astype(float)
             for j,(key,name) in enumerate(zip(['Wih','Whh','Wo'],keys))]
        wi,A,O=W
        wf=np.stack([wi,np.diagonal(A,axis1=1,axis2=2),np.log(np.linalg.norm(A,axis=2)+1e-9),np.log(np.linalg.norm(A,axis=1)+1e-9),np.log(np.linalg.norm(O,axis=1)+1e-9)],2)
        st=np.concatenate([wf.mean(1),wf.std(1),wf.min(1),wf.max(1)],1)
        m4,m3 = c['m4'].astype(float),c['m3'].astype(float)
        s = np.sqrt((m4*m4).mean(1,keepdims=True)); a,b = m4/s,m3/s
        f = np.stack([c['I4'],c['I3'],a,b,abs(a),abs(b),a/(.1+abs(a)),b/(.1+abs(b)),a/(.03+abs(a)),b/(.03+abs(b))],-1)
        F = np.c_[s,np.log(s),m4.mean(1),m3.mean(1),f.mean(1),f.reshape(len(a),720,K,10).mean(1).reshape(len(a),-1)]
        base, y = np.c_[st,F],np.load('pop1k/reference.npz')['acc']
        net = float(np.load(f'pop_unif/cells_k{K}.npz')['net_fe'])
    T,H,ops = extra_features(c,K)
    return base,T,H,y,net,ops

def predictions(X, extras, y, tr, va, modes):
    """Ridge + RBF grids with block distances and training-only standardization."""
    def distances(Z):
        mean = Z[tr].mean(0); sd = np.maximum(Z[tr].std(0),1e-6)
        A,B = (Z[tr]-mean)/sd,(Z[va]-mean)/sd
        return A,B,cdist(A,A,'sqeuclidean')/Z.shape[1],cdist(B,A,'sqeuclidean')/Z.shape[1]
    A,B,D,V = distances(X); yt = y[tr]-y[tr].mean()
    E = {k: distances(z)[2:] for k,z in extras.items()}
    out, names = [], []
    ev,U = eigh(A.T@A); q=U.T@(A.T@yt)
    out.append(((B@U)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None]*len(tr))))
    names += [('base','ridge',0.,float(al)) for al in ALP]
    for mode,weight in modes:
        Dt,Dv = (D,V) if mode == 'base' else (D+weight*E[mode][0],V+weight*E[mode][1])
        for g in GAM:
            ev,U=eigh(np.exp(-g*Dt)); q=U.T@yt
            out.append(((np.exp(-g*Dv)@U)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None])))
            names += [(mode,float(weight),float(g),float(al)) for al in ALP]
    return y[tr].mean()+np.concatenate(out,1),names

def evaluate(X, extras, y, modes, kinds=None):
    idx=np.arange(1024); fold=np.random.default_rng(0).permutation(1024)%8
    pred={k:np.zeros(len(y)) for k in (kinds or ['base','tails','groups','both','select'])}
    picks={k:[] for k in pred}; outs=[]
    for outer in range(9):
        tr = idx if outer == 8 else idx[fold != outer]
        va = np.arange(1024,len(y)) if outer == 8 else idx[fold == outer]
        inner = np.arange(len(tr))%8 if outer == 8 else (np.arange(len(tr))*4//len(tr))[np.argsort(np.argsort(tr*7919%1024))]
        loss=None
        for j in range(inner.max()+1):
            pr,names=predictions(X,extras,y,tr[inner!=j],tr[inner==j],modes)
            err=((pr-y[tr[inner==j],None])**2).sum(0)
            loss=err if loss is None else loss+err
        pr,names=predictions(X,extras,y,tr,va,modes)
        for kind in pred:
            eligible = np.array([kind == 'select' or nm[0] in ('base',kind) for nm in names])
            choice=np.where(eligible)[0][np.argmin(loss[eligible])]
            pred[kind][va]=pr[:,choice]; picks[kind].append(names[choice])
        print('outer',outer,'choices', {k:picks[k][-1] for k in pred},flush=True)
    return pred,picks

def main():
    p=argparse.ArgumentParser(); p.add_argument('--recipe',choices=['unif','zoo'],default='unif')
    p.add_argument('--K',type=int,default=1); p.add_argument('--out',default='/workspace/scratch/0aa5efd25118/round7_work')
    a=p.parse_args(); started=time.time()
    X,T,H,y,net,ops=load(a.recipe,a.K)
    extras={'tails':T,'groups':H,'both':np.c_[T,H]}
    modes=[('base',0.)]+[(k,w) for k in extras for w in [.1,1.]]
    pred,picks=evaluate(X,extras,y,modes)
    results={}
    for kind in pred:
        d=X.shape[1]+(extras[kind].shape[1] if kind in extras else 0 if kind == 'base' else T.shape[1]+H.shape[1])
        add = 0 if kind == 'base' else ops['tails'] if kind == 'tails' else ops['groups'] if kind == 'groups' else sum(ops.values())
        # Per-block scaling adds at most d multiplies; all features charged even if baseline selected.
        cost=net+(40*720*a.K+5000+2*1024*d+34*1024+2*d+3+add+d)/2880
        groups={'crossfit':np.arange(1024)}
        groups.update({'b128':np.arange(1024,1088),'b64':np.arange(1088,1152),'mystery':np.array([1152])} if a.recipe=='unif' else {'mystery':np.array([1024]),'seeds':np.arange(1025,1030)})
        summary={}
        for name,idx in groups.items():
            error=pred[kind][idx]-y[idx]
            summary[name]={'rms_points':float(100*np.sqrt(np.mean(error**2))),'mean_points':float(100*error.mean()),'mse_ratio':float(np.mean(error**2)*cost/np.mean(y[idx]*(1-y[idx])))}
            if len(idx)<=6: summary[name]['errors_points']=(100*error).tolist()
        results[kind]={'d':d,'rows':1440*a.K,'net_fe':net,'add_feature_ops':int(add),'total_fe':cost,'summary':summary,'picks':picks[kind]}
        print(kind,json.dumps(results[kind]),flush=True)
    out=Path(a.out); out.mkdir(exist_ok=True)
    stem=out/f'aggregates_{a.recipe}_k{a.K}'
    np.savez_compressed(str(stem)+'.npz',**pred,y=y,T=T,H=H,X=X)
    Path(str(stem)+'.json').write_text(json.dumps({'recipe':a.recipe,'K':a.K,'runtime_seconds':time.time()-started,'results':results},indent=2))
    print('runtime_seconds',time.time()-started,flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=1): main()
