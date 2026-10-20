"""Follow-on cost experiments; choices motivated by initial population-CV results.

No external labels choose a method. Every RBF/ridge parameter is nested.
Run: python combinations.py PROJECT OUTPUT
"""
import sys,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cost_experiments import load,features,design,row_cost,evaluate,metrics,RRANK

def main():
    root,out=Path(sys.argv[1]),Path(sys.argv[2]); out.mkdir(exist_ok=True,parents=True)
    source=Path(__file__).parent
    # First three compare one-anchor designs; next three add signed pulses to
    # each end-position design. Last three add pulses to the full MF4 design.
    variants=[(1,'all','4',False),(1,'all','3',False),(2,'all','3',False)]
    variants += [(k,'ends','both',True) for k in (1,2,4)]
    variants += [(k,'all','4',True) for k in (1,2,4)]
    for pop in ('unif','zoo'):
        pulse=np.load(source/f'mechanics_pulse_{pop}.npz')['pulse']
        for k,subset,kind,addpulse in variants:
            name=f'{pop}_k{k}_{subset}_mf{kind}_p{int(addpulse)}_sup128'
            if (out/f'{name}.json').exists(): continue
            t=time.monotonic();z,st,y=load(root,pop,k)
            ar=np.where(np.isin(RRANK,[0,7]))[0] if subset=='ends' else np.arange(720)
            x=features(z,st,k,ar,kind)
            if addpulse: x=np.c_[x,pulse]
            rows=design(k,ar,kind);net=row_cost(rows);cells=len(ar)*k;d=x.shape[1]
            other=((40 if kind=='both' else 22)*cells+5000)/2880+6*addpulse
            readout=(2*128*d+34*128+2*d+3)/2880
            total=net+other+readout
            pred,picks,best=evaluate(x,y,128)
            result=dict(name=name,population=pop,k=k,subset=subset,kind=kind,pulse=addpulse,
                        supports=128,rows=len(rows),cells=cells,dimensions=d,net_fe=net,
                        feature_fe=other,readout_fe=readout,total_fe=total,
                        outer_picks=picks,refit_pick=best,metrics=metrics(pred,y,total,pop),seconds=time.monotonic()-t)
            np.savez_compressed(out/f'{name}.npz',pred=pred,y=y,X=x)
            (out/f'{name}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
