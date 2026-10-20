"""Run after mechanics pulse caches exist; choices remain nested within folds."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from aggregates import load,evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',default='unif');p.add_argument('--K',type=int,default=1);p.add_argument('--mf4',action='store_true')
    a=p.parse_args();start=time.time();out=Path(__file__).parent
    X,T,H,y,net,ops=load(a.recipe,a.K)
    if a.mf4:
        # Retain only MF4 context and channels; its RMS normalization is unchanged.
        X=np.c_[X[:,:23],X[:,24:].reshape(len(X),a.K+1,10)[:,:,[0,2,4,6,8]].reshape(len(X),-1)]
        H=H.reshape(len(H),6,a.K,4)[:,:,:,[0,2]].reshape(len(H),-1)
        net=264.5*a.K+37.9  # exact prefix-trie count for all 720 MF4 arrangements
        ops['groups']//=2
    P=np.load(out/f'mechanics_pulse_{a.recipe}.npz')['pulse']
    X=np.c_[X,P]
    pred,picks=evaluate(X,{'groups':H},y,[('base',0.),('groups',.1),('groups',1.)],['base','groups'])
    results={}
    for kind,pr in pred.items():
        d=X.shape[1]+(H.shape[1] if kind=='groups' else 0)
        add=6*2880+(ops['groups'] if kind=='groups' else 0)
        cost=net+((22 if a.mf4 else 40)*720*a.K+5000+2*1024*d+34*1024+2*d+3+add+d)/2880
        groups={'crossfit':np.arange(1024)}
        groups.update({'b128':np.arange(1024,1088),'b64':np.arange(1088,1152),'mystery':np.array([1152])} if a.recipe=='unif' else {'mystery':np.array([1024]),'seeds':np.arange(1025,1030)})
        summary={}
        for name,idx in groups.items():
            err=pr[idx]-y[idx]
            summary[name]={'rms_points':float(100*np.sqrt(np.mean(err**2))),'mean_points':float(100*err.mean()),'mse_ratio':float(np.mean(err**2)*cost/np.mean(y[idx]*(1-y[idx])))}
            if len(idx)<=6:summary[name]['errors_points']=(100*err).tolist()
        nr=(720 if a.mf4 else 1440)*a.K
        results[kind]={'d':d,'rows':nr+2,'anchor_rows':nr,'pulse_rows':2,'net_fe':net,'add_feature_ops':int(add),'total_fe':cost,'summary':summary,'picks':picks[kind]}
        print(kind,json.dumps(results[kind]),flush=True)
    stem=out/f'aggregates_{"mf4pulse" if a.mf4 else "pulse"}_{a.recipe}_k{a.K}'
    np.savez_compressed(str(stem)+'.npz',**pred,y=y,X=X,H=H)
    Path(str(stem)+'.json').write_text(json.dumps({'recipe':a.recipe,'K':a.K,'runtime_seconds':time.time()-start,'results':results},indent=2))
    print('runtime_seconds',time.time()-start,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
