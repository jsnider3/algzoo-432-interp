"""More S slices with only MF4; fixed deterministic rows, CPU prefix sharing.
Run from project: python /path/extended.py PROJECT OUTPUT [K ...]
"""
import sys,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cost_experiments import load,features,design,row_cost,trie_plan,evaluate,metrics,ARR
from mechanics import weights

def mf4_cells(w, k):
    rows=design(k,np.arange(720),'4');vals,levels,leaf=trie_plan(rows)
    target=np.repeat(ARR[:,1],k);bits=[];margins=[]
    for start in range(0,len(w[0]),24):
        wi,wh,wo=[a[start:start+24].astype(np.float32) for a in w]
        inp=wi[:,None,:]*vals[None,:,None];h=None
        for parent,token in levels:
            h=np.maximum(inp[:,token],0) if h is None else np.maximum((h@wh.transpose(0,2,1))[:,parent]+inp[:,token],0)
        lg=(h@wo.transpose(0,2,1))[:,leaf]
        bits.append(lg.argmax(2)==target[None])
        chosen=lg[:,np.arange(len(target)),target].copy()
        lg[:,np.arange(len(target)),target]=-np.inf
        margins.append(chosen-lg.max(2))
    return {'I4':np.concatenate(bits),'m4':np.concatenate(margins)}

def temporal4(z,k):
    m=z['m4'].astype(float);s=np.maximum(np.sqrt((m*m).mean(1,keepdims=True)),1e-15)
    a=m/s;v=np.stack([z['I4'],a/(.1+abs(a))],-1).reshape(len(m),720,k,2)
    M,S,R=ARR.T
    masks=[M==9,S==9,M==S+1,M>S,R>S,abs(M-S)>=5]
    h=np.concatenate([v[:,mask].mean(1).reshape(len(m),-1) for mask in masks],1)
    # If called separately from features, include normalizing and softsign again.
    ops=6*720*k+2*k*sum(int(mask.sum()) for mask in masks)+12*k
    return h,ops

def main():
    root,out=Path(sys.argv[1]),Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    ks=list(map(int,sys.argv[3:])) or [8,16]
    for pop in ('unif','zoo'):
        w=weights(pop)
        _,st,y=load(root,pop,4)
        pulse=np.load(Path(__file__).parent/f'mechanics_pulse_{pop}.npz')['pulse']
        for k in ks:
            start=time.monotonic();cache=out/f'cells_{pop}_k{k}.npz'
            if cache.exists():z=dict(np.load(cache))
            elif k<=4:
                zz,_,_=load(root,pop,k);z={name:zz[name] for name in ('m4','I4')}
            else:
                z=mf4_cells(w,k);np.savez_compressed(cache,**z)
            generated=time.monotonic()-start
            x=features(z,st,k,np.arange(720),'4');h,hops=temporal4(z,k)
            for groups in (False,True):
                name=f'{pop}_k{k}_mf4_pulse_h{int(groups)}_sup128'
                if (out/f'{name}.json').exists():continue
                began=time.monotonic();xx=np.c_[x,pulse,h] if groups else np.c_[x,pulse]
                cells=720*k;d=xx.shape[1];net=row_cost(design(k,np.arange(720),'4'))
                oth=(22*cells+5000+(hops if groups else 0))/2880+6
                readout=(2*128*d+34*128+2*d+3)/2880;total=net+oth+readout
                pred,picks,best=evaluate(xx,y,128)
                result=dict(name=name,population=pop,k=k,pulse=True,groups=groups,dimensions=d,rows=cells,
                            cells=cells,supports=128,net_fe=net,feature_fe=oth,readout_fe=readout,total_fe=total,
                            outer_picks=picks,refit_pick=best,metrics=metrics(pred,y,total,pop),
                            generation_seconds=generated,fit_seconds=time.monotonic()-began)
                np.savez_compressed(out/f'{name}.npz',X=xx,y=y,pred=pred)
                (out/f'{name}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
