import numpy as np
from scipy.stats import norm
from seeds_clocks import load, run, T
def f_and_mf(x,Wih,Whh,Wo):
    o=np.argsort(-x,1); p2=o[:,1]; R=-np.sort(-x,1)[:,2]
    f=((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==p2)
    xm=x.copy(); mu=-norm.pdf(R)/norm.cdf(R); low=np.ones_like(x,bool); n=np.arange(len(x))
    low[n,o[:,0]]=low[n,o[:,1]]=low[n,o[:,2]]=False; xm[low]=np.repeat(mu[:,None],10,1)[low]
    mf=((run(xm,Wih,Whh)[:,-1]@Wo.T).argmax(1)==p2)
    return f.astype(float), mf.astype(float)
for name,path,EMF in [("mystery","weights.npz",0.9559),("seed 2","weights_seed2.npz",0.9498),("seed 0","weights_seed0.npz",0.9115)]:
    Wih,Whh,Wo=load(path); rng=np.random.default_rng(1)
    xt=rng.standard_normal((2000000,T)); ft,mt=f_and_mf(xt,Wih,Whh,Wo); true=ft.mean()
    print(f"{name}: true {true:.4f}; E[MF] by quadrature {EMF:.4f} vs sampled {mt.mean():.4f}; P(f != MF) {np.mean(ft!=mt):.4f}; corr {np.corrcoef(ft,mt)[0,1]:.3f}")
    for budget in [2000,8000]:
        nv=[]; cv=[]
        for _ in range(400):
            x=rng.standard_normal((budget,T)); f,_=f_and_mf(x[:0],Wih,Whh,Wo) if False else (None,None)
            fb,_m=f_and_mf(x,Wih,Whh,Wo); nv.append(fb.mean())                    # naive uses `budget` evals (MF part unused)
            xh=x[:budget//2]; fh,mh=f_and_mf(xh,Wih,Whh,Wo); cv.append((fh-mh).mean()+EMF)  # CV uses budget/2 samples x 2 evals
        nv=np.array(nv); cv=np.array(cv)
        print(f"   budget {budget} evals: naive MSE {np.mean((nv-true)**2):.2e} | control-variate MSE {np.mean((cv-true)**2):.2e} (bias {cv.mean()-true:+.4f}) -> {np.mean((nv-true)**2)/np.mean((cv-true)**2):.1f}x")
