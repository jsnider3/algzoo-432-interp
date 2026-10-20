"""Nested learning curve for fixed MF4 + pulses, K=2, 128 supports.

Run next to cost_experiments.py:
  python learning_curve.py PATH/TO/unif_k2_all_mf4_p1_sup128.npz
The adjacent JSON supplies the constant deployed cost. No test labels enter
subset selection, standardization, hyperparameter choice, or fitting.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cost_experiments import predictions, FOLD

def score(p, y, cost):
    e=p-y; mse=float(np.mean(e*e))
    return dict(rms=100*np.sqrt(mse),mean=100*float(e.mean()),
                ratio=mse*cost/float(np.mean(y*(1-y))))

def main(path):
    src=Path(path); d=np.load(src); x,y=d['X'],d['y']
    cost=json.loads(src.with_suffix('.json').read_text())['total_fe']
    pop=np.arange(1024); external=np.arange(1024,len(y))
    priority=np.random.default_rng(202607).permutation(1024)
    records=[]; allpred=[]; started=time.monotonic()
    for nt,nfinal in zip((224,448,672,896),(256,512,768,1024)):
        tic=time.monotonic(); cf=np.empty(1024); picks=[]; outer=[]
        for f in range(8):
            eligible=priority[FOLD[priority]!=f]
            tr=np.sort(eligible[:nt]); te=pop[FOLD==f]
            inner=(np.arange(nt)*4//nt)[np.argsort(np.argsort(tr*7919%1024))]
            loss=sum(((predictions(x,y,tr[inner!=j],tr[inner==j],128)
                       -y[tr[inner==j],None])**2).sum(0) for j in range(4))
            best=int(loss.argmin()); candidates=predictions(x,y,tr,te,128)
            cf[te]=candidates[:,best]; picks.append(best); outer.append((te,candidates))
        # Fixed nested subsets of the 1024 population. Refit parameters come
        # from eight training-only folds on that subset, preserving FOLD IDs.
        tr=np.sort(priority[:nfinal])
        if nfinal==1024:
            loss=sum(((p-y[te,None])**2).sum(0) for te,p in outer)
        else:
            loss=sum(((predictions(x,y,tr[FOLD[tr]!=f],tr[FOLD[tr]==f],128)
                       -y[tr[FOLD[tr]==f],None])**2).sum(0) for f in range(8))
        best=int(loss.argmin()); extra=predictions(x,y,tr,external,128)[:,best]
        pp=np.r_[cf,extra]; allpred.append(pp)
        r=dict(outer_train_count=nt,external_train_count=nfinal,total_fe=cost,
               crossfit=score(cf,y[:1024],cost),outer_picks=picks,refit_pick=best,
               b128=score(pp[1024:1088],y[1024:1088],cost),
               b64=score(pp[1088:1152],y[1088:1152],cost),
               mystery_error=100*float(pp[-1]-y[-1]),seconds=time.monotonic()-tic)
        records.append(r); print(json.dumps(r),flush=True)
    out=Path(__file__).parent
    result=dict(method='MF4 + pulses, K=2, 128 supports',source=str(src),
                priority_seed=202607,outer_folds='rng0 permutation(1024) % 8',
                inner_folds='4, stable IDs and denominator1024',
                supports=128,features=x.shape[1],total_fe=cost,records=records,
                total_seconds=time.monotonic()-started)
    (out/'learning_curve.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'learning_curve.npz',pred=np.array(allpred),y=y,
                        priority=priority,outer_train=[224,448,672,896],
                        external_train=[256,512,768,1024])

if __name__=='__main__':
    with threadpool_limits(limits=1): main(sys.argv[1])
