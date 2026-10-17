import numpy as np, itertools
perms=np.array(list(itertools.permutations(range(10),3)))
# symbol per position: 0=mu, 1=M, 2=S, 3=R
seqs=np.zeros((720,10),int)
for i,p in enumerate(perms): seqs[i,p[0]]=1; seqs[i,p[1]]=2; seqs[i,p[2]]=3
levels=[]; node_of_prefix={(): 0}; parents_prev=None
leaf_index=None
prev_nodes=[()]
for d in range(10):
    nodes={}; par=[]; sym=[]
    for s in seqs:
        pre=tuple(s[:d+1])
        if pre not in nodes:
            nodes[pre]=len(nodes); par.append(prev_index[tuple(s[:d])] if d>0 else 0); sym.append(s[d])
    levels.append((np.array(par),np.array(sym)))
    prev_index=nodes
leaf=np.array([prev_index[tuple(s)] for s in seqs])
n_steps=sum(len(p) for p,_ in levels)
def cached_logits(vals, Wih, Whh, Wo):
    """vals: (nl,4) values for symbols [mu,M,S,R]; returns logits (nl,720,10)"""
    nl=len(vals); h=np.zeros((nl,1,16))
    for par,sym in levels:
        x=vals[:,sym]                                   # (nl,n_d)
        h=np.maximum(x[:,:,None]*Wih[None,None,:]+h[:,par,:]@Whh.T,0)
    return h[:,leaf,:]@Wo.T
if __name__=="__main__":
    from seeds_clocks import load
    from l3 import logits
    from scipy.stats import norm
    Wih,Whh,Wo=load("weights.npz"); rng=np.random.default_rng(0)
    R=rng.normal(size=50); S=R+rng.exponential(.5,50); M=S+rng.exponential(.5,50); mu=-norm.pdf(R)/norm.cdf(R)
    vals=np.c_[mu,M,S,R]; Lc=cached_logits(vals,Wih,Whh,Wo)
    X=np.repeat(mu[:,None,None],10,2).repeat(720,1); ii=np.arange(50)[:,None]; jj=np.arange(720)[None,:]
    X[ii,jj,perms[None,:,0]]=M[:,None]; X[ii,jj,perms[None,:,1]]=S[:,None]; X[ii,jj,perms[None,:,2]]=R[:,None]
    Lu=logits(X.reshape(-1,10),Wih,Whh,Wo).reshape(50,720,10)
    print("max |cached - uncached| logits:", np.abs(Lc-Lu).max(), "| steps per line:", n_steps, "vs", 7200, f"({7200/n_steps:.2f}x)")
