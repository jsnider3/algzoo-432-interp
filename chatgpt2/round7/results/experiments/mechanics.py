"""Deterministic clock/pointer features. Run from the archive directory.

python /path/to/mechanics.py --recipes unif zoo --ks 1 2 4
No torch, target sampling, or fitting on external networks is used.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

GAM = [.003, .01, .03, .1, .3, 1.]
ALP = np.logspace(-5, 3, 9)

def weights(recipe):
    def read(path, prefix):
        d = np.load(path)
        return [d[prefix+k].astype(float) for k in ('Wih','Whh','Wo')]
    def one(path):
        d = np.load(path)
        return [d['rnn.weight_ih_l0'][:,0],d['rnn.weight_hh_l0'],d['linear.weight']]
    if recipe == 'unif':
        ww = [read('pop_unif/'+p+'.npz','s1000_') for p in
              ('b128_n1024_seed4','b128_seed0','b64_seed5')]
        ww.append([x[None] for x in one('weights.npz')])
    else:
        ww = [read('pop1k/seed0.npz','s1048576_')]
        ww += [[x[None] for x in one(p)] for p in
               ['weights.npz']+[f'weights_seed{j}.npz' for j in range(5)]]
    return [np.concatenate([w[j] for w in ww]) for j in range(3)]

def impulse_features(wi,A,O):
    """16 basis-pointer summaries plus 10 signed input-pulse summaries.

    Logits are normalized independently by age and injected direction; all
    summaries survive positive diagonal hidden-unit and global logit scaling.
    Basis impulses test a possible memory state, not its reachability.
    """
    n = len(wi)
    h = np.concatenate([np.broadcast_to(np.eye(16),(n,16,16)),
                        np.maximum(wi,0)[:,:,None],np.maximum(-wi,0)[:,:,None]],2)
    L = []
    for age in range(10):
        L.append(O @ h)
        if age != 9: h = np.maximum(A @ h,0)
    L = np.stack(L,1)                           # network, age, class, impulse
    lc = L-L.mean(2,keepdims=True)
    amp = np.sqrt((lc*lc).mean(2))
    scale = np.maximum(amp,1e-12)
    out = []
    allmean = []
    for lag in (0,1):
        a = np.arange(10-lag); target = 9-a-lag
        valid = L[:,a].copy(); correct = valid.argmax(2)==target[None,:,None]
        tgt = valid[:,a,target,:].copy()
        valid[:,a,target,:] = -np.inf
        margin = (tgt-valid.max(2))/scale[:,a]
        mean = margin[:,:,:16].mean(1)
        rate = correct[:,:,:16].mean(1)
        allmean.append(mean)
        out += [mean.max(1),np.sort(mean,axis=1)[:,-4:].mean(1),
                rate.max(1),np.sort(rate,axis=1)[:,-4:].mean(1),
                margin[:,:,:16].min(1).max(1),
                margin[:,5:,:16].mean(1).max(1)]
        if lag == 0:
            pulse = [mean for mean in
                     (margin[:,:,16:].mean(1),correct[:,:,16:].mean(1),
                      margin[:,5:,16:].mean(1))]
    # Choose a basis pointer using only that network's mean normalized margins.
    best = np.stack(allmean,1).reshape(n,-1).argmax(1)%16
    growth = np.diff(np.log(np.maximum(amp,1e-12)),axis=1)
    selected = growth[np.arange(n),:,best]
    out += [selected.mean(1),selected.std(1),
            growth[:,:,:16].mean((1,2)),growth[:,:,:16].std((1,2))]
    pulse += [growth[:,:,16:].mean(1),growth[:,:,16:].std(1)]
    return np.stack(out,1),np.concatenate(pulse,1)

def pulse_features(wi,A,O):
    """10 features, only two deterministic signed pulses; budget 6 FE.

    For each sign: mean age margin, age alignment fraction, late-age margin,
    and mean/SD of successive log output-amplitude changes. Age alignment
    tests how the memory decodes a pulse, not second-argmax task correctness.
    """
    h=np.maximum(np.stack([wi,-wi],2),0); ll=[]
    for age in range(10):
        ll.append(O@h)
        if age<9: h=np.maximum(A@h,0)
    L=np.stack(ll,1); lc=L-L.mean(2,keepdims=True)
    amp=np.maximum(np.sqrt((lc*lc).mean(2)),1e-12)
    age=np.arange(10); tgt=9-age
    aligned=L.argmax(2)==tgt[None,:,None]
    target=L[:,age,tgt,:].copy(); L[:,age,tgt,:]=-np.inf
    margin=(target-L.max(2))/amp
    growth=np.diff(np.log(amp),axis=1)
    return np.concatenate([margin.mean(1),aligned.mean(1),margin[:,5:].mean(1),
                           growth.mean(1),growth.std(1)],1)

def features(recipe,k,W):
    path = (f'pop_unif/cells_k{k}.npz' if recipe=='unif' else
            f'pop1k/seed0_z46_cells_b1_k{k}.npz')
    d = np.load(path)
    if recipe == 'unif': return np.c_[d['stats'],d['F']],d['acc']
    wi,A,O = W
    ff = np.stack([wi,np.diagonal(A,axis1=1,axis2=2),
                   np.log(np.linalg.norm(A,axis=2)+1e-9),
                   np.log(np.linalg.norm(A,axis=1)+1e-9),
                   np.log(np.linalg.norm(O,axis=1)+1e-9)],2)
    st = np.concatenate([ff.mean(1),ff.std(1),ff.min(1),ff.max(1)],1)
    m4,m3 = d['m4'].astype(float),d['m3'].astype(float)
    s = np.sqrt((m4*m4).mean(1,keepdims=True)); a,b = m4/s,m3/s
    f = np.stack([d['I4'],d['I3'],a,b,abs(a),abs(b),a/(.1+abs(a)),
                  b/(.1+abs(b)),a/(.03+abs(a)),b/(.03+abs(b))],2)
    return np.c_[st,s,np.log(s),m4.mean(1),m3.mean(1),f.mean(1),
                  f.reshape(len(f),720,k,10).mean(1).reshape(len(f),-1)],np.load('pop1k/reference.npz')['acc']

def fold_preds(x,y,tr,va):
    mu=x[tr].mean(0); sd=np.maximum(x[tr].std(0),1e-6)
    a,b=(x[tr]-mu)/sd,(x[va]-mu)/sd; ym=y[tr].mean(); yt=y[tr]-ym
    ev,V=eigh(a.T@a); q=V.T@(a.T@yt)
    out=[((b@V)*q)@(1/(np.maximum(ev,0)[:,None]+ALP[None]*len(tr)))]
    D,Dv=cdist(a,a,'sqeuclidean'),cdist(b,a,'sqeuclidean')
    for g in GAM:
        ev,V=eigh(np.exp(-g/x.shape[1]*D)); q=V.T@yt
        out.append(((np.exp(-g/x.shape[1]*Dv)@V)*q)@
                   (1/(np.maximum(ev,0)[:,None]+ALP[None])))
    return ym+np.concatenate(out,1)

def nested(x,y):
    idx=np.arange(1024); folds=np.random.default_rng(0).permutation(1024)%8
    p=np.zeros(1024); pick=[]
    for f in range(8):
        tr,va=idx[folds!=f],idx[folds==f]
        inner=(np.arange(len(tr))*4//len(tr))[np.argsort(np.argsort(tr*7919%1024))]
        loss=sum(((fold_preds(x,y,tr[inner!=j],tr[inner==j])-
                   y[tr[inner==j],None])**2).sum(0) for j in range(4))
        j=int(loss.argmin()); pick.append(j)
        p[va]=fold_preds(x,y,tr,va)[:,j]
    # Refit hyperparameters chosen entirely on the 1024 training networks.
    loss=sum(((fold_preds(x,y,idx[idx%8!=f],idx[idx%8==f])-
               y[idx[idx%8==f],None])**2).sum(0) for f in range(8))
    j=int(loss.argmin())
    extra=fold_preds(x,y,idx,np.arange(1024,len(y)))[:,j]
    return np.r_[p,extra],pick,j

def describe(pred,y,cost):
    e=pred-y
    return {'rms':float(100*np.sqrt(np.mean(e*e))),
            'mean':float(100*e.mean()),'ratio':float(np.mean(e*e)*cost/np.mean(y*(1-y)))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--recipes',nargs='+',default=['unif','zoo'])
    p.add_argument('--ks',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--sets',nargs='+',default=['basis','pulse'])
    p.add_argument('--output-json',default='mechanics_results.json')
    args=p.parse_args(); dest=Path(__file__).parent
    outpath=dest/args.output_json
    records=json.loads(outpath.read_text()) if outpath.exists() else []
    with threadpool_limits(limits=1):
      for recipe in args.recipes:
        W=weights(recipe); base,pulse=impulse_features(*W)
        assert np.allclose(pulse,pulse_features(*W),atol=1e-12,rtol=1e-12)
        np.savez(dest/f'mechanics_features_{recipe}.npz',basis=base,pulse=pulse)
        for k in args.ks:
          X,y=features(recipe,k,W)
          for label in args.sets:
            extra={'basis':base,'pulse':pulse,'both':np.c_[base,pulse],
                   'baseline':np.empty((len(X),0)),'statspulse':pulse,
                   'weights':np.empty((len(X),0))}[label]
            low=label in ('statspulse','weights')
            xx=np.c_[X[:,:20] if low else X,extra]
            start=time.time(); pr,pick,full=nested(xx,y)
            # Provisional explicit ledger; the notes provide exact accounting.
            added={'basis':35.,'pulse':6.,'both':41.,'baseline':0.,'statspulse':6.,'weights':0.}[label]
            net=0 if low else 494*k+69.4
            total=net+(40*720*k*(not low)+5000+2*1024*xx.shape[1]+34*1024+2*xx.shape[1]+3)/2880+added
            r={'recipe':recipe,'k':k,'set':label,'d':xx.shape[1],'added_fe':added,
               'fe':total,'picks':pick,'full_pick':full,'seconds':time.time()-start,
               'crossfit':describe(pr[:1024],y[:1024],total)}
            if recipe=='unif':
                r.update(b128=describe(pr[1024:1088],y[1024:1088],total),
                         b64=describe(pr[1088:1152],y[1088:1152],total),
                         mystery_error=float(100*(pr[-1]-y[-1])))
            else: r['external_errors']=list(100*(pr[1024:]-y[1024:]))
            records=[v for v in records if (v['recipe'],v['k'],v['set'])!=(recipe,k,label)]
            records.append(r); print(json.dumps(r),flush=True)
            np.savez(dest/f'mechanics_pred_{recipe}_{k}_{label}.npz',pred=pr,y=y)
            outpath.write_text(json.dumps(records,indent=2))

if __name__=='__main__': main()
