"""Reproduce fixed-protocol G baselines; run from the supplied project directory."""
import ast, json, sys, time, types, itertools
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import ndtr, ndtri, betaincinv
from seeds_clocks import load as zload

# Load only the unchanged NumPy/SciPy definitions; torch is absent here.
tree=ast.parse(Path('scripts/gfeat.py').read_text())
keep={'mu','rows','trie','stats','cost','fold_preds','pick','crossfit','fit_predict'}
ns=dict(np=np,itertools=itertools,eigh=eigh,cdist=cdist,ndtr=ndtr,ndtri=ndtri,betaincinv=betaincinv)
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in keep:
        exec(compile(ast.Module(body=[node],type_ignores=[]),'gfeat.py','exec'),ns)
    elif isinstance(node,(ast.Assign,ast.AnnAssign)):
        names={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        if names and 'dev' not in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'gfeat.py','exec'),ns)
gfeat=types.SimpleNamespace(**ns)

OUT = Path(__file__).resolve().parent
pop = np.arange(1024)
fold = np.random.default_rng(0).permutation(1024) % 8

def summarize(d, k):
    m4, m3 = d['m4'].astype(float), d['m3'].astype(float)
    s = np.sqrt(np.mean(m4*m4, axis=1, keepdims=True))
    a, b = m4/s, m3/s
    f = np.stack([d['I4'], d['I3'], a, b, abs(a), abs(b),
                  a/(.1+abs(a)), b/(.1+abs(b)),
                  a/(.03+abs(a)), b/(.03+abs(b))], axis=-1)
    return np.c_[s, np.log(s), m4.mean(1), m3.mean(1), f.mean(1),
                 f.reshape(len(f), 720, k, 10).mean(1).reshape(len(f), -1)]

def score(pred, target, cost):
    e = pred-target
    return dict(n=len(e), rms=100*np.sqrt(np.mean(e*e)), mean=100*e.mean(),
                ratio=float(np.mean(e*e)*cost/np.mean(target*(1-target))))

def main():
    results=[]
    pd=np.load('pop1k/seed0.npz')
    p=[pd[f's1048576_{n}'] for n in ('Wih','Whh','Wo')]
    z=[zload(f) for f in ['weights.npz']+[f'weights_seed{i}.npz' for i in range(5)]]
    w=[np.concatenate([a.astype(float), np.stack([b[i] for b in z])]) for i,a in enumerate(p)]
    ws=gfeat.stats(*w)
    for recipe in ('unif','zoo'):
        for k in (1,2,4):
            start=time.time()
            path=f'pop_unif/cells_k{k}.npz' if recipe=='unif' else f'pop1k/seed0_z46_cells_b1_k{k}.npz'
            d=np.load(path)
            f=summarize(d,k)
            x=np.c_[d['stats'] if recipe=='unif' else ws, f]
            y=d['acc'] if recipe=='unif' else np.load('pop1k/reference.npz')['acc']
            lev,_,nv=gfeat.trie(gfeat.rows(1,k)); cnt=[len(p) for p,_ in lev]
            net=(256*sum(cnt[:-1])+16*nv+160*cnt[-1]+16*sum(cnt[1:])+16*sum(cnt)+20*cnt[-1])/2880
            cost=gfeat.cost(net,720*k,1,x.shape[1],1024)
            pred=gfeat.crossfit(x[:1024],y[:1024],pop,fold)
            ext=gfeat.fit_predict(x,y,pop,np.arange(1024,len(y)))
            row=dict(recipe=recipe,k=k,d=x.shape[1],cost=float(cost),net=float(net),
                     population=score(pred,y[:1024],cost),prefix_counts=cnt,
                     input_product_cost_gap=16*(sum(cnt)-nv)/2880,
                     cache_summary_diff=float(np.max(abs(d['F']-f))) if recipe=='unif' else None)
            if recipe=='unif':
                row.update(b128=score(ext[:64],y[1024:1088],cost),
                           b64=score(ext[64:128],y[1088:1152],cost),
                           mystery_error=100*float(ext[-1]-y[-1]))
            else:
                row['released_errors']=(100*(ext-y[1024:])).tolist()
            row['seconds']=time.time()-start
            results.append(row)
            np.savez(OUT/f'audit_{recipe}_k{k}.npz',pred=pred,ext=ext,y=y)
            (OUT/'audit_baselines.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
