import numpy as np
from seeds_clocks import load, run, T
from scipy.special import expit
from scipy.optimize import minimize
rng=np.random.default_rng(91)
def estimate(path, nprobe=20000):
    Wih,Whh,Wo=load(path)
    f=lambda x:(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)
    # local probe: last input placed near the current 2nd max (u = (x9 - S)/M in [-0.15,0.15]); P(model picks slot 9)
    x=rng.standard_normal((nprobe,T)); srt=-np.sort(-x[:,:9],1); M,S=srt[:,0],srt[:,1]
    u=rng.uniform(-0.15,0.15,nprobe); x[:,9]=S+u*M; ok=(S>0); y=(f(x)==9)
    nll=lambda p: -np.mean(np.where(y[ok], np.log(expit((u[ok]-p[0])/p[1])+1e-12), np.log(1-expit((u[ok]-p[0])/p[1])+1e-12)))
    b,eps=minimize(nll,[0,0.02],method='Nelder-Mead').x
    # blind spot rate: error when S<0 (small probe: rejection sample S<0 cases)
    xs=rng.standard_normal((400000,T)); xs=xs[(-np.sort(-xs,1))[:,1]<0][:2000]
    eneg=np.mean(f(xs)!=np.argsort(xs,1)[:,-2])
    # model-free integration over Gaussian order statistics
    z=rng.standard_normal((4_000_000,T)); s=-np.sort(-z,1); Mz,Sz,Rz=s[:,0],s[:,1],s[:,2]; pos=Sz>0
    g23=(Sz-Rz)/Mz; g12=(Mz-Sz)/Mz
    # P(correct) for a pairwise comparison with gap g: logistic resolution eps, bias b
    p23=expit((g23-b)/eps); p12=expit((g12-b)/eps)
    acc=np.mean(np.where(pos, p23*p12, 1-eneg))
    return acc, eps, b, eneg, nprobe+2000
for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz"),("seed 0","weights_seed0.npz")]:
    Wih,Whh,Wo=load(path); x=np.random.default_rng(5).standard_normal((3000000,T)); true=np.mean((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2])
    acc,eps,b,eneg,cost=estimate(path)
    sd=np.sqrt(true*(1-true)/cost)
    print(f"{name:8s} true {true:.4f} | estimate {acc:.4f} (error {acc-true:+.4f}) using eps={eps:.3f}·M, bias {b:+.3f}, blind-spot err {eneg:.2f}, {cost} model evals | sampling sd at same cost {sd:.4f}")
