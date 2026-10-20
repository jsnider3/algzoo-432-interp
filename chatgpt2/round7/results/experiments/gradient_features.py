"""Prespecified gradients on 34 event arrangements; no target sampling.
Run from the supplied project directory. Needs audit_check.py beside this file.
"""
import json, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import audit_check as au
G=au.gfeat
OUT=Path(__file__).resolve().parent

def weights(recipe):
    names=['pop_unif/b128_n1024_seed4.npz','pop_unif/b128_seed0.npz','pop_unif/b64_seed5.npz'] if recipe=='unif' else ['pop1k/seed0.npz']
    prefix='s1000_' if recipe=='unif' else 's1048576_'
    pp=[np.load(f) for f in names]
    zz=[au.zload(f) for f in ['weights.npz']+([] if recipe=='unif' else [f'weights_seed{i}.npz' for i in range(5)])]
    return [np.concatenate([p[prefix+n] for p in pp]+[np.stack([z[i] for z in zz])]).astype(float) for i,n in enumerate(('Wih','Whh','Wo'))]

def selection(k):
    ids=[]
    for i,(m,s,r) in enumerate(G.ARR):
        free=[t for t in range(10) if t not in (m,s)]
        if (m==9 or m==s+1) and r in (free[0],free[-1]): ids.append(i)
    slices=[0] if k==1 else [0,k-1]
    return np.array([i*k+j for i in ids for j in slices]),np.array(slices)

def margin_grad(w,x,y):
    """Signed target-versus-best-rival margin and the active ReLU Jacobian."""
    wi,A,O=w; n=len(wi); h=np.zeros((n,len(x),16)); gates=[]
    for t in range(10):
        pre=x[None,:,t,None]*wi[:,None,:]
        if t: pre=pre+h@A.transpose(0,2,1)
        gates.append(pre>0); h=np.maximum(pre,0)
    logits=h@O.transpose(0,2,1)
    target=logits[:,np.arange(len(x)),y].copy()
    logits[:,np.arange(len(x)),y]=-np.inf
    rival=logits.argmax(2)  # NumPy argmax preserves the lowest-index tie rule.
    margin=target-np.take_along_axis(logits,rival[...,None],2)[...,0]
    adj=O[:,y,:]-O[np.arange(n)[:,None],rival,:]
    grad=np.zeros((n,len(x),10))
    for t in range(9,-1,-1):
        adj=adj*gates[t]; grad[:,:,t]=np.sum(adj*wi[:,None,:],axis=2)
        if t: adj=adj@A
    return margin,grad

def extract(w,k,s):
    """Differentiates actual anchor gates and active output rival; eight features/slice."""
    ids,sl=selection(k); x=G.rows(1,k)[2*ids].astype(float)
    margin,grad=margin_grad(w,x,G.ARR[ids//k,1])
    lower=x<0; grad*=lower[None]
    q=float(np.sort(x[0])[-4]); mu=float(G.mu(q))
    sigma=np.sqrt(1+q*mu-mu*mu)
    z=grad*(sigma/np.maximum(s,1e-12))[:,None,None]
    l2=np.sqrt(np.sum(z*z,axis=2)); a=margin/np.maximum(s,1e-12)[:,None]
    pair=lower[:,:-1]&lower[:,1:]
    both=(z[:,:,:-1]+z[:,:,1:])*pair[None]
    diff=(z[:,:,:-1]-z[:,:,1:])*pair[None]
    f=np.stack([l2,np.sum(abs(z),axis=2)/np.sqrt(6),
                abs(z.sum(2))/np.sqrt(6),np.sqrt(np.sum(z[:,:,8:]**2,axis=2)),
                np.sqrt(np.sum(both*both,axis=2)/5),
                np.sqrt(np.sum(diff*diff,axis=2)/5),
                (a>0)&(a<=l2),l2/(abs(a)+l2+1e-12)],axis=2)
    out=np.concatenate([f[:,ids%k==j,:].mean(1) for j in sl],axis=1)
    return out,dict(rows=len(x),dims=out.shape[1],backward_fe=len(x)*2640/2880,
                    summary_fe=(len(x)*160+out.shape[1])/2880,
                    # Includes explicit gate comparisons in this NumPy version.
                    standalone_forward_fe=len(x)*3108/2880,
                    q=q,sigma=sigma)

def main(k=4):
    for recipe in ('unif','zoo'):
        start=time.time(); w=weights(recipe)
        d=np.load(f'pop_unif/cells_k{k}.npz' if recipe=='unif' else f'pop1k/seed0_z46_cells_b1_k{k}.npz')
        f=au.summarize(d,k); st=d['stats'] if recipe=='unif' else G.stats(*w)
        y=d['acc'] if recipe=='unif' else np.load('pop1k/reference.npz')['acc']
        b=np.c_[st,f]; s=f[:,0]
        feat,meta=extract(w,k,s); x=np.c_[b,feat]
        base=next(r for r in json.loads((OUT/'audit_baselines.json').read_text()) if r['recipe']==recipe and r['k']==k)
        cost=base['cost']+meta['backward_fe']+meta['summary_fe']+(2*1024+2)*feat.shape[1]/2880
        p=G.crossfit(x[:1024],y[:1024],au.pop,au.fold)
        ex=G.fit_predict(x,y,au.pop,np.arange(1024,len(y)))
        row=dict(recipe=recipe,k=k,meta=meta,total_fe=cost,standalone_total_fe=cost+meta['standalone_forward_fe'],
                 population=au.score(p,y[:1024],cost),baseline=base['population'])
        if recipe=='unif':
            row.update(b128=au.score(ex[:64],y[1024:1088],cost),b64=au.score(ex[64:128],y[1088:1152],cost),mystery_error=100*float(ex[-1]-y[-1]))
        else: row['released_errors']=(100*(ex-y[1024:])).tolist()
        for group in ('population','b128','b64'):
            if group in row:
                row[group]['standalone_ratio']=row[group]['ratio']*row['standalone_total_fe']/cost
        row['seconds']=time.time()-start
        np.savez(OUT/f'gradients_{recipe}_k{k}.npz',features=feat,pred=p,ext=ex)
        (OUT/f'gradients_{recipe}_k{k}.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)

if __name__=='__main__':
    import sys
    with threadpool_limits(limits=1): main(int(sys.argv[1]) if len(sys.argv)>1 else 4)
