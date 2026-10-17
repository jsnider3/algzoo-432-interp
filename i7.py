import numpy as np
from seeds_clocks import load, run, T
# coarse strata: blind spot + 10 normalized-gap bins on the decisive pair (model-free assignment)
GB=np.array([0,.01,.02,.035,.05,.075,.1,.15,.2,.3,10])
def strat(x):
    s=-np.sort(-x,1); M,S,R=s[:,0],s[:,1],s[:,2]; g=np.minimum(M-S,S-R)/np.abs(M)
    return np.where(S<0,0,1+np.clip(np.digitize(g,GB)-1,0,9))
K=11; rng=np.random.default_rng(111)
W=np.bincount(strat(rng.standard_normal((20_000_000,T))),minlength=K)/20_000_000
mid=np.r_[0,(GB[:-1]+np.minimum(GB[1:],0.5))/2]
prior=np.r_[0.75, 0.4*np.exp(-mid[1:]/0.03)+0.01]          # generic mechanism prior, same for every model
alloc=W*np.sqrt(prior*(1-prior)); alloc/=alloc.sum()
print("strata weights", W.round(3)); print("allocation   ", alloc.round(3))
pool={}
def draw(h,n):
    buf=pool.get(h,np.zeros((0,T)))
    while len(buf)<n:
        z=rng.standard_normal((200000,T)); z=z[strat(z)==h]; buf=np.concatenate([buf,z])
    pool[h]=buf[n:]; return buf[:n]
if __name__=="__main__":
    for name,path in [("mystery","weights.npz"),("seed 2","weights_seed2.npz"),("seed 3","weights_seed3.npz"),("seed 0","weights_seed0.npz"),("seed 1","weights_seed1.npz")]:
        Wih,Whh,Wo=load(path); f=lambda x:(run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==np.argsort(x,1)[:,-2]
        xt=np.random.default_rng(5).standard_normal((3000000,T)); true=np.mean(f(xt))
        for budget in [2000,8000]:
            nv=[np.mean(f(rng.standard_normal((budget,T)))) for _ in range(200)]
            st=[]
            for _ in range(200):
                est=0
                for h in range(K):
                    n=max(3,int(round(alloc[h]*budget))); est+=W[h]*np.mean(f(draw(h,n)))
                st.append(est)
            mn=np.mean((np.array(nv)-true)**2); ms=np.mean((np.array(st)-true)**2)
            print(f"{name:8s} budget {budget}: naive MSE {mn:.2e} | pilot-free mechanism-stratified MSE {ms:.2e} (bias {np.mean(st)-true:+.4f}) -> {mn/ms:.1f}x")
