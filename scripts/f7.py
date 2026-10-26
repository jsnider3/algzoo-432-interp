import numpy as np
from seeds_clocks import load, T
def jac(x,Wih,Whh,Wo):
    N=len(x); h=np.zeros((N,16)); dh=np.zeros((N,16,T)); W=[]
    for t in range(T):
        pre=x[:,t:t+1]*Wih+h@Whh.T; m=(pre>0).astype(float)
        dp=np.einsum('ij,njk->nik',Whh,dh); dp[:,:,t]+=Wih
        W.append(np.linalg.norm(Wih*m,axis=1))   # immediate write strength of x_t
        dh=dp*m[:,:,None]; h=pre*m
    return np.einsum('oh,nht->not',Wo,dh), np.stack(W,1)
if __name__=="__main__":
    for name,path in [("seed 0","weights_seed0.npz"),("seed 4","weights_seed4.npz"),("mystery","weights.npz"),("seed 2","weights_seed2.npz")]:
        Wih,Whh,Wo=load(path); x=np.random.default_rng(61).standard_normal((60000,T)); J,Wr=jac(x,Wih,Whh,Wo)
        o=np.argsort(-x,1); n=np.arange(len(x)); p1,p2,p3=o[:,0],o[:,1],o[:,2]
        # own-slot sensitivity d logit_i / d x_i, centered across logits
        Jc=J-J.mean(1,keepdims=True)
        own=lambda p: Jc[n,p,p].mean()
        # write strength when x_t is a new running max vs not
        run_max=np.maximum.accumulate(x,1); newmax=np.c_[np.ones((len(x),1),bool), x[:,1:]>run_max[:,:-1]]
        print(f"{name:8s} d logit_i/d x_i (centered) at: max slot {own(p1):6.1f}  2nd slot {own(p2):6.1f}  3rd slot {own(p3):6.1f} | input-write norm: new max {Wr[newmax].mean():5.1f}  not new max {Wr[~newmax].mean():5.1f}")
