"""Round 7: prespecified sparse designs and compiled/inducing-point readouts.

Run from the supplied project: python /path/to/cost_experiments.py . OUTDIR
Only the first 1024 labels enter fitting. No target input sampling is used.
"""
import sys, json, time, itertools
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import ndtr, ndtri, betaincinv
from threadpoolctl import threadpool_limits

GAM = [.003, .01, .03, .1, .3, 1.]
ALP = np.logspace(-5, 3, 9)
ARR = np.array(list(itertools.permutations(range(10), 3)))
FREE = [[i for i in range(10) if i not in a[:2]] for a in ARR]
RRANK = np.array([f.index(a[2]) for f, a in zip(FREE, ARR)])
PAIR = {0: 7, 7: 0, 1: 3, 3: 1, 2: 5, 5: 2, 4: 6, 6: 4}
QREP = np.array([f[PAIR[r]] for f, r in zip(FREE, RRANK)])
FOLD = np.random.default_rng(0).permutation(1024) % 8

def weights(path):
    z = np.load(path)
    if 'rnn.weight_ih_l0' in z:
        return [z['rnn.weight_ih_l0'].reshape(16).astype(float),
                z['rnn.weight_hh_l0'].astype(float),z['linear.weight'].astype(float)]
    if 'Wih' in z:
        return [z[k].astype(float) for k in ('Wih', 'Whh', 'Wo')]
    t = max(int(k[1:-4]) for k in z.files if k.startswith('s') and k.endswith('_Wih'))
    return [z[f's{t}_{k}'].astype(float) for k in ('Wih', 'Whh', 'Wo')]

def stats(wi, wh, wo):
    f = np.stack([wi, np.diagonal(wh, axis1=1, axis2=2),
                  np.log(np.linalg.norm(wh, axis=2) + 1e-9),
                  np.log(np.linalg.norm(wh, axis=1) + 1e-9),
                  np.log(np.linalg.norm(wo, axis=1) + 1e-9)], 2)
    return np.concatenate([f.mean(1), f.std(1), f.min(1), f.max(1)], 1)

def load(root, population, k):
    if population == 'unif':
        z = np.load(root / f'pop_unif/cells_k{k}.npz')
        return z, z['stats'], z['acc'].astype(float)
    z = np.load(root / f'pop1k/seed0_z46_cells_b1_k{k}.npz')
    p = weights(root / 'pop1k/seed0.npz')
    zz = [weights(root / name) for name in ['weights.npz'] + [f'weights_seed{i}.npz' for i in range(5)]]
    # Released models store the original PyTorch key names.
    w = [np.concatenate([p[j], np.stack([a[j] for a in zz])]) for j in range(3)]
    return z, stats(*w), np.load(root / 'pop1k/reference.npz')['acc'].astype(float)

def design(k, ar, kind='both'):
    fq = betaincinv(7, 4, .5)
    fr = fq + (1-fq) * betaincinv(1, 3, .5)
    fm = fr + (1-fr) * betaincinv(2, 1, .5)
    q, r, m = ndtri([fq, fr, fm])
    s = ndtri(fr + (fm-fr)*(np.arange(k)+.5)/k)
    mu = lambda v: -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
    out = []
    for a in ar:
        for si in s:
            for typ in ([4,3] if kind == 'both' else [int(kind)]):
                row = np.full(10, mu(q if typ == 4 else r))
                row[ARR[a]] = [m, si, r]
                if typ == 4:
                    row[QREP[a]] = q
                out.append(row)
    return np.array(out, dtype=np.float32)

def trie_plan(x):
    vals, tok = np.unique(x, return_inverse=True)
    tok = tok.reshape(x.shape)
    par, levels = np.zeros(len(x), dtype=int), []
    for t in range(10):
        key, par = np.unique(par*len(vals)+tok[:,t], return_inverse=True)
        levels.append((key//len(vals), key%len(vals)))
    return vals, levels, par

def row_cost(x):
    vals, levels, _ = trie_plan(x)
    n = np.array([len(p) for p, _ in levels])
    mac = 256*n[:-1].sum()+16*len(vals)+160*n[-1]
    other = 16*n[1:].sum()+16*n.sum()+20*n[-1]
    return float((mac+other)/2880)

def features(z, st, k, ar, kind='both'):
    ci = (ar[:,None]*k+np.arange(k)).ravel()
    if kind != 'both':
        ma, ia = z[f'm{kind}'][:,ci].astype(float), z[f'I{kind}'][:,ci]
        s = np.maximum(np.sqrt((ma*ma).mean(1,keepdims=True)),1e-15)
        a = ma/s
        f = np.stack([ia,a,abs(a),a/(.1+abs(a)),a/(.03+abs(a))],-1)
        ctx = [s,np.log(s),ma.mean(1,keepdims=True)]
    else:
        m4, m3 = [z[name][:, ci].astype(float) for name in ('m4','m3')]
        i4, i3 = [z[name][:, ci] for name in ('I4','I3')]
        s = np.maximum(np.sqrt((m4*m4).mean(1,keepdims=True)),1e-15)
        a, b = m4/s, m3/s
        f = np.stack([i4,i3,a,b,abs(a),abs(b),a/(.1+abs(a)),b/(.1+abs(b)),a/(.03+abs(a)),b/(.03+abs(b))],-1)
        ctx = [s,np.log(s),m4.mean(1,keepdims=True),m3.mean(1,keepdims=True)]
    ff = f.reshape(len(st),len(ar),k,-1)
    return np.concatenate([st]+ctx+[f.mean(1)]+[ff[:,:,i].mean(1) for i in range(k)],1)

def predictions(x, y, tr, te, supports=0):
    """Candidate predictions. Fit work is offline; compiled query is O(supports*d)."""
    mu, sd = x[tr].mean(0), np.maximum(x[tr].std(0),1e-6)
    a, b = (x[tr]-mu)/sd, (x[te]-mu)/sd
    ym = y[tr].mean()
    yc = y[tr]-ym
    ev, v = eigh(a.T@a)
    coef = v@((v.T@(a.T@yc))[:,None]/(np.maximum(ev,0)[:,None]+len(tr)*ALP))
    ans = [b@coef]
    for g in GAM:
        if supports:
            ix = np.random.default_rng(1701).permutation(len(tr))[:min(supports,len(tr))]
            centers = a[ix]
            km = np.exp(-g/x.shape[1]*cdist(centers,centers,'sqeuclidean'))
            e, u = eigh(km)
            keep = e > max(1e-12,e[-1]*1e-12)
            whitening = u[:,keep]/np.sqrt(e[keep])
            ka = np.exp(-g/x.shape[1]*cdist(a,centers,'sqeuclidean'))
            phi = ka@whitening
            e, v = eigh(phi.T@phi)
            coef = whitening@v@((v.T@(phi.T@yc))[:,None]/(np.maximum(e,0)[:,None]+ALP))
        else:
            centers = a
            ka = np.exp(-g/x.shape[1]*cdist(a,a,'sqeuclidean'))
            e, v = eigh(ka)
            coef = v@((v.T@yc)[:,None]/(np.maximum(e,0)[:,None]+ALP))
        kb = np.exp(-g/x.shape[1]*cdist(b,centers,'sqeuclidean'))
        ans.append(kb@coef)
    return ym+np.concatenate(ans,1)

def evaluate(x, y, supports=0):
    pop = np.arange(1024)
    out = np.empty(len(y)); outer, picks = [], []
    for f in range(8):
        tr, te = pop[FOLD!=f], pop[FOLD==f]
        inner = (np.arange(len(tr))*4//len(tr))[np.argsort(np.argsort(tr*7919%1024))]
        loss = sum(((predictions(x,y,tr[inner!=j],tr[inner==j],supports)-y[tr[inner==j],None])**2).sum(0) for j in range(4))
        p = predictions(x,y,tr,te,supports)
        best = int(loss.argmin()); picks.append(best)
        out[te] = p[:,best]; outer.append((te,p))
    loss = sum(((p-y[te,None])**2).sum(0) for te,p in outer)
    best = int(loss.argmin())
    out[1024:] = predictions(x,y,pop,np.arange(1024,len(y)),supports)[:,best]
    return out, picks, best

def metrics(p, y, total, population):
    groups = [('crossfit',slice(0,1024))]
    groups += [('b128',slice(1024,1088)),('b64',slice(1088,1152)),('mystery',slice(1152,1153))] if population=='unif' else [('mystery',slice(1024,1025)),('seeds',slice(1025,1030))]
    out = {}
    for name, sl in groups:
        e = p[sl]-y[sl]; mse = (e*e).mean()
        out[name] = dict(rms=100*float(np.sqrt(mse)),mean=100*float(e.mean()),ratio=float(mse*total/(y[sl]*(1-y[sl])).mean()))
        if name in ('seeds','mystery'): out[name]['errors'] = (100*e).tolist()
    return out

def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True,exist_ok=True)
    # Fixed structural designs: all ordered (M,S) pairs, with R in free ranks 0 and 7.
    designs = [(1,'ends','both',0),(2,'ends','both',0),(4,'ends','both',0),
               (2,'all','4',0),(4,'all','4',0),(4,'all','3',0),
               (1,'all','both',128),(2,'all','both',128),(4,'all','both',128)]
    if len(sys.argv)>3:
        designs = [designs[int(i)] for i in sys.argv[3].split(',')]
    for popname in ('unif','zoo'):
        for k, subset, kind, supports in designs:
            name = f'{popname}_k{k}_{subset}_mf{kind}_sup{supports or 1024}'
            if (out/f'{name}.json').exists(): continue
            start = time.monotonic()
            z, st, y = load(root,popname,k)
            ar = np.where(np.isin(RRANK,[0,7]))[0] if subset=='ends' else np.arange(720)
            x = features(z,st,k,ar,kind)
            rows = design(k,ar,kind)
            net = row_cost(rows); cells = len(ar)*k
            # Conservative feature extraction, with context, norm/log and standardization allowance.
            feature_fe = ((40 if kind=='both' else 22)*cells+5000)/2880
            readout_fe = (2*(supports or 1024)*x.shape[1]+34*(supports or 1024)+2*x.shape[1]+3)/2880
            total = net+feature_fe+readout_fe
            pred,picks,best = evaluate(x,y,supports)
            result = dict(name=name,population=popname,k=k,subset=subset,kind=kind,supports=supports or 1024,
                          rows=len(rows),cells=cells,dimensions=x.shape[1],net_fe=net,feature_fe=feature_fe,
                          readout_fe=readout_fe,total_fe=total,outer_picks=picks,refit_pick=best,
                          metrics=metrics(pred,y,total,popname),seconds=time.monotonic()-start)
            np.savez_compressed(out/f'{name}.npz',pred=pred,y=y,X=x)
            (out/f'{name}.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
