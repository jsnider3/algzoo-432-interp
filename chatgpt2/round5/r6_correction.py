"""Deterministic two-level MF3/MF4 correction; RNG is confined to measure().
Run from the repository, with OPENBLAS_NUM_THREADS=2. All forward rows charged.
Examples: python r6_correction.py --measure 256
 python r6_correction.py --split dev --k 3 --n 2 --K 16 --q 8 --keep 8
 python r6_correction.py --split held --k 4 --n 2 --K 8 --q 8 --keep 8
"""
import argparse, itertools, json, os, pathlib, time
import numpy as np
from scipy.special import ndtr, ndtri, roots_jacobi

OUT = pathlib.Path('r6_results'); OUT.mkdir(exist_ok=True)

def save_npz(path, **data):
    """Atomic replacement also keeps any open lazy NPZ reader valid."""
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**data)
    tmp.replace(path)

def load(split='dev'):
    ids = np.r_[np.arange(8 if split == 'held' else 0, 1024, 16), 1024:1030]
    p = np.load('pop1k/seed0.npz')
    z = [np.load(f) for f in ['weights.npz']+[f'weights_seed{i}.npz' for i in range(5)]]
    zz = [np.stack([v[key][:, 0] if j == 0 else v[key] for v in z]) for j, key in
          enumerate(['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'])]
    w = [np.concatenate([p['s1048576_'+key][ids[:-6]], b]).astype(np.float32) for key, b in zip(['Wih','Whh','Wo'],zz)]
    return ids, w

def predict(w, x):
    h = np.zeros((len(w[0]), len(x), 16), dtype=w[0].dtype)
    for t in range(10): h = np.maximum(h @ w[1].transpose(0,2,1) + x[None,:,t,None]*w[0][:,None], 0)
    return (h @ w[2].transpose(0,2,1)).argmax(-1)

def rule(n, a, b):
    x, w = roots_jacobi(n, b-1, a-1)
    return (x+1)/2, w/w.sum()

def mu(v): return -np.exp(-np.asarray(v)**2/2)/np.sqrt(2*np.pi)/ndtr(v)

def arrangements(k, keep):
    a = np.array(list(itertools.permutations(range(10), k)))
    if keep == 0 or keep == (8 if k == 3 else 56): return a
    chosen = []
    for row in a:
        f = [x for x in range(10) if x not in row[:2]]
        if k == 3: yes = f.index(row[2]) in {1:[3],2:[1,6],4:[0,2,5,7]}[keep]
        else:
            pairs = [(0,7),(7,0),(1,3),(3,1),(2,5),(5,2),(4,6),(6,4)]
            yes = (f.index(row[2]),f.index(row[3])) in pairs[:keep]
        if yes: chosen.append(row)
    return np.array(chosen)

def grid(w, k, n, K, q, keep):
    """Returns per-arrangement integrals: anchor3, anchor-k, corrected model.
    CDF Gauss-Jacobi outer rule, S midpoints, q Gauss-Legendre lower nodes.
    For k=4 the separately evaluated MF3 anchor allows MF4-model minus MF3.
    """
    arr = arrangements(k, keep); A = len(arr); N = len(w[0]); low = np.array([[i for i in range(10) if i not in a] for a in arr])
    rules = [rule(n,8,3),rule(n,2,1)] if k == 3 else [rule(n,7,4),rule(n,1,3),rule(n,2,1)]
    uq, wq = rule(q,1,1); out = np.zeros((N,A,3)); rows = 1+(10-k)*q+(k==4)
    block = max(1,4096//(K*rows)); count = 0
    for ij in itertools.product(range(n), repeat=len(rules)):
        u = [rules[t][0][j] for t,j in enumerate(ij)]; wt = np.prod([rules[t][1][j] for t,j in enumerate(ij)])
        if k == 3: fr = u[0]; fm = fr+(1-fr)*u[1]; vals = [ndtri(fm),None,ndtri(fr)]
        else: fq=u[0]; fr=fq+(1-fq)*u[1]; fm=fr+(1-fr)*u[2]; vals=[ndtri(fm),None,ndtri(fr),ndtri(fq)]
        s = ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K); zn = ndtri(ndtr(vals[-1])*uq)
        for start in range(0,A,block):
            aa = arr[start:start+block]; ll = low[start:start+block]; b=len(aa)
            base = np.full((b,K,10),mu(vals[-1]),np.float32)
            for j,v in enumerate(vals): base[np.arange(b)[:,None],np.arange(K),aa[:,j,None]] = s if j==1 else v
            X = np.repeat(base[:,:,None,:],rows,axis=2)
            for j in range(10-k): X[np.arange(b)[:,None,None],np.arange(K)[None,:,None],1+j*q+np.arange(q),ll[:,j,None,None]] = zn
            if k == 4:
                X[:,:,-1] = mu(vals[2])
                for j in range(3): X[np.arange(b)[:,None],np.arange(K),-1,aa[:,j,None]] = s if j==1 else vals[j]
            I = (predict(w,X.reshape(-1,10)).reshape(N,b,K,rows)==aa[None,:,None,None,1]).astype(float)
            i0=I[...,0]; ej=(I[...,1:1+(10-k)*q].reshape(N,b,K,10-k,q)*wq).sum(-1)
            model=np.clip(i0+(ej-i0[...,None]).sum(-1),0,1)
            out[:,start:start+b,0]+=wt*(I[...,-1] if k==4 else i0).mean(-1)
            out[:,start:start+b,1]+=wt*i0.mean(-1); out[:,start:start+b,2]+=wt*model.mean(-1)
            count += b*K*rows
    return arr,out,count

def score(ids, est, rows, target='acc'):
    p=np.load('pop1k/reference.npz'); e=100*(est-p[target][ids]); pop=e[:-6]
    samp=100*np.sqrt(np.mean(p['acc'][ids[:-6]]*(1-p['acc'][ids[:-6]]))/(rows*1.01))
    return dict(forward_rows=int(rows),cost=rows*1.01,rms=float(np.sqrt(np.mean(pop**2))),mean=float(pop.mean()),
                sampling_rms=float(samp),mse_ratio=float(np.mean(pop**2)/samp**2),zoo=e[-6:].tolist())

def measure(reps):
    """Balanced independent measurement of true-minus-anchor arrangement means.
    This function never constructs an estimator; finite measurement noise is subtracted.
    """
    ids,w=load(); arr=arrangements(4,0); N=len(ids); A=len(arr); rng=np.random.default_rng(6007)
    sums=np.zeros((2,N,A)); squares=np.zeros_like(sums); parts=np.zeros((4,2,N,A))
    block=max(1,4096//reps); t0=time.time()
    for st in range(0,A,block):
        aa=arr[st:st+block]; b=len(aa); a=np.repeat(aa,reps,axis=0); B=len(a)
        v=np.sort(rng.standard_normal((B,10)),axis=1)[:,::-1][:,:4]
        lp=np.array([[p for p in range(10) if p not in row] for row in a]); x=np.zeros((B,10),np.float32)
        x[np.arange(B)[:,None],lp]=ndtri(ndtr(v[:,3,None])*rng.random((B,6)))
        x[np.arange(B)[:,None],a]=v; y=a[:,1]; truth=predict(w,x)==y
        for j,k in enumerate((3,4)):
            base=np.repeat(mu(v[:,k-1,None]),10,axis=1).astype(np.float32); base[np.arange(B)[:,None],a[:,:k]]=v[:,:k]
            d=(truth.astype(np.int8)-(predict(w,base)==y)).reshape(N,b,reps)
            sums[j,:,st:st+b]=d.sum(-1); squares[j,:,st:st+b]=(d*d).sum(-1)
            for h,pp in enumerate(np.array_split(np.arange(reps),4)): parts[h,j,:,st:st+b]=d[...,pp].sum(-1)
        if st%(block*80)==0: print('measurement',st,A,round(time.time()-t0,1),flush=True)
    result={}
    for j,k in enumerate((3,4)):
        groups=np.array([tuple(row[:k]) for row in arr]); _,inv=np.unique(groups,axis=0,return_inverse=True)
        G=inv.max()+1; s=np.zeros((N,G)); ss=np.zeros_like(s); nr=np.bincount(inv)*reps
        for i in range(N): s[i]=np.bincount(inv,weights=sums[j,i]); ss[i]=np.bincount(inv,weights=squares[j,i])
        means=s/nr; rawnoise=(squares[j]-sums[j]**2/reps)/(reps-1)/reps
        noise=np.stack([np.bincount(inv,weights=v) for v in rawnoise])/(nr/reps)**2
        var=means.var(1)-(1-1/G)*noise.mean(1)
        result[f'mf{k}']=dict(sd_points=(100*np.sqrt(np.maximum(var,0))).tolist(),raw_sd_points=(100*means.std(1)).tolist(),
            per_arrangement_se_points=(100*np.sqrt(noise.mean(1))).tolist(),correction_mean_points=(100*means.mean(1)).tolist(),arrangements=int(G),draws_per_arrangement=int(nr[0]))
    save_npz(OUT/'correction_measurement.npz',ids=ids,arr=arr,sums=sums,squares=squares,parts=parts,reps=reps)
    (OUT/'correction_measurement.json').write_text(json.dumps(result,indent=2)); arrangement_report()

def arrangement_report():
    """Describe the measurement only; noise-corrected subset errors are not estimates."""
    z=np.load(OUT/'correction_measurement.npz'); a=z['arr']; reps=int(z['reps']); result={}
    for j,k,label in [(0,3,'mf3'),(1,4,'mf4'),(0,4,'mf4_arr_minus_mf3')]:
        arr,inv=np.unique(a[:,:k],axis=0,return_inverse=True); G=len(arr); N=len(z['ids']); nr=np.bincount(inv)*reps
        s=np.stack([np.bincount(inv,weights=v,minlength=G) for v in z['sums'][j]])
        ss=np.stack([np.bincount(inv,weights=v,minlength=G) for v in z['squares'][j]])
        m=s/nr; rawnoise=(z['squares'][j]-z['sums'][j]**2/reps)/(reps-1)/reps
        noise=np.stack([np.bincount(inv,weights=v) for v in rawnoise])/(nr/reps)**2
        var=m.var(1)-(1-1/G)*noise.mean(1)
        r=dict(sd_points=(100*np.sqrt(np.maximum(var,0))).tolist(),classes={},subsets={})
        ms=10*arr[:,0]+arr[:,1]; rel=ms.copy()
        for col in range(2,k): rel=4*rel+2*(arr[:,col]>arr[:,0])+(arr[:,col]>arr[:,1])
        for class_label,gg in [('MS',ms),('MS_relative_order',rel)]:
            _,ii=np.unique(gg,return_inverse=True); cnt=np.bincount(ii); cm=np.stack([np.bincount(ii,weights=v)/cnt for v in m])
            residual=((m-cm[:,ii])**2).mean(1)-(noise*(1-1/cnt[ii])).mean(1)
            r['classes'][class_label]=dict(count=len(cnt),variance_share=(1-residual/var).tolist(),residual_sd_points=(100*np.sqrt(np.maximum(residual,0))).tolist())
        for keep in ([1,2,4] if k==3 else [4,8]):
            aa={tuple(row) for row in arrangements(k,keep)}; mask=np.array([tuple(row) in aa for row in arr]); w=mask/mask.sum()-1/G
            e=100*(m@w); nv=10000*(noise@(w*w)); rms=np.sqrt(max(np.mean((e*e-nv)[:-6]),0))
            r['subsets'][str(keep)]=dict(arrangements=int(mask.sum()),noise_corrected_rms=float(rms),raw_rms=float(np.sqrt(np.mean(e[:-6]**2))),
                mean=float(e[:-6].mean()),per_network_measurement_se=float(np.sqrt(nv[:-6].mean())),zoo=e[-6:].tolist(),zoo_se=np.sqrt(nv[-6:]).tolist())
        result[label]=r
    (OUT/'correction_arrangement_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:dict(median_sd_points=float(np.median(v['sd_points'][:-6])),subsets=v['subsets']) for k,v in result.items()},indent=2))

def reports(split):
    """Read fixed subset rules from already evaluated arrangement integrals.
    Charge only rows required when that declared rule is deployed, plus actual anchor cost.
    """
    ref=np.load('pop1k/reference.npz'); corrections={}; output=[]
    for cfg in json.load(open(OUT/'correction_frozen.json'))['settings']:
        k,n,K,q,keep=[cfg[a] for a in ['k','n','K','q','keep']]
        stem=f'correction_{split}_k{k}_n{n}_K{K}_q{q}_keep'
        source=OUT/(stem+str(8 if k==3 or n==2 else 56)+'.npz')
        if not source.exists(): continue
        z=np.load(source); aa={tuple(x) for x in arrangements(k,keep)}; mask=np.array([tuple(x) in aa for x in z['arr']])
        ids=z['ids']; v=z['integrals'][:,mask]; rows=int(z['rows'])*mask.sum()//len(mask); es=v.mean(1); delta=es[:,2]-es[:,0]; delta4=es[:,2]-es[:,1]
        name=stem+str(keep); corrections[name]=(ids,delta,rows)
        save_npz(OUT/(name+'.npz'),ids=ids,arr=z['arr'][mask],integrals=v,rows=rows,delta=delta,delta4=delta4)
        rr=dict(settings=cfg,same_grid=score(ids,es[:,2],rows),correction_with_reference_anchor=score(ids,ref['mf3'][ids]+delta,rows))
        if k==4: rr['correction_with_reference_mf4']=score(ids,ref['mf4'][ids]+delta4,rows)
        (OUT/(name+'.json')).write_text(json.dumps(rr,indent=2))
    for af in ['anchors','exact_anchor']:
        f=OUT/(af+'_'+split+'.json')
        if not f.exists(): continue
        for a in json.load(open(f)):
            if a['n'] not in (3,4,6): continue
            for name,(ids,delta,rows) in corrections.items():
                assert np.array_equal(ids,a['ids'])
                costs=np.asarray(a.get('costs_per_network',np.full(len(ids),a['cost'])))+rows*1.01
                e=100*(np.array(a['estimates'])+delta-ref['acc'][ids]); pq=ref['acc'][ids]*(1-ref['acc'][ids])
                sm=100*np.sqrt(np.mean((pq/costs)[:-6])); rms=float(np.sqrt(np.mean(e[:-6]**2)))
                output.append(dict(anchor=af,n=a['n'],K=a.get('K'),correction=name,cost=float(costs[:-6].mean()),cost_is_lower_bound=False,
                    cost_min=float(costs[:-6].min()),cost_max=float(costs[:-6].max()),rms=rms,mean=float(e[:-6].mean()),
                    sampling_rms=float(sm),mse_ratio=float(rms*rms/sm**2),zoo=e[-6:].tolist(),ids=ids.tolist(),errors_pp=e.tolist()))
                output[-1]['exact_anchor_omits_scalar_work']=a.get('cost_is_lower_bound',False)
                if a.get('cost_is_lower_bound',False):
                    mac=np.asarray(a['costs_per_network'])+rows
                    ms=100*np.sqrt(np.mean((pq/mac)[:-6]))
                    output[-1].update(mac_only_cost_lower_bound=float(mac[:-6].mean()),
                        sampling_rms_at_mac_cost=float(ms),mse_ratio_at_mac_cost=float(rms*rms/ms**2))
    (OUT/('correction_combinations_'+split+'.json')).write_text(json.dumps(output,indent=2))
    print(split,'combinations saved:',len(output))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--measure',type=int,default=0); p.add_argument('--report',action='store_true'); p.add_argument('--split',default='dev',choices=['dev','held'])
    for a,d in [('k',3),('n',1),('K',8),('q',4),('keep',8)]: p.add_argument('--'+a,type=int,default=d)
    a=p.parse_args()
    if a.measure: measure(a.measure); return
    if a.report: reports(a.split); return
    ids,w=load(a.split); t=time.time(); arr,v,rows=grid(w,a.k,a.n,a.K,a.q,a.keep)
    est=v.mean(1); ref=np.load('pop1k/reference.npz'); delta=est[:,2]-est[:,0]
    report=dict(settings=vars(a),seconds=time.time()-t,arrangements=len(arr),same_grid=score(ids,est[:,2],rows),
                correction_with_reference_anchor=score(ids,ref['mf3'][ids]+delta,rows))
    delta4=est[:,2]-est[:,1]
    if a.k==4: report['correction_with_reference_mf4']=score(ids,ref['mf4'][ids]+delta4,rows)
    name=f'correction_{a.split}_k{a.k}_n{a.n}_K{a.K}_q{a.q}_keep{a.keep}'
    save_npz(OUT/(name+'.npz'),ids=ids,arr=arr,integrals=v,rows=rows,delta=delta,delta4=delta4)
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
