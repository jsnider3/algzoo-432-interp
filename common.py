import os, numpy as np
w = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz"))
Wih = w['rnn.weight_ih_l0'][:,0].astype(np.float64)
Whh = w['rnn.weight_hh_l0'].astype(np.float64)
Wo  = w['linear.weight'].astype(np.float64)
H, T = 16, 10
def run(x, Wih=Wih, Whh=Whh, Wo=Wo, hook=None):
    """x: (N,T). returns logits (N,T), hidden states (N,T,H) (state after reading x_t)"""
    N = x.shape[0]; h = np.zeros((N,H)); hs = []
    for t in range(x.shape[1]):
        pre = x[:,t:t+1]*Wih + h@Whh.T
        h = np.maximum(pre, 0)
        if hook is not None: h = hook(t, h, pre)
        hs.append(h)
    hs = np.stack(hs, 1)
    return h@Wo.T, hs
def target(x): return np.argsort(x, -1)[:, -2]
def acc(logits, x): return (logits.argmax(-1) == target(x)).mean()
rng = np.random.default_rng(0)
def sample(N=200000, seed=0): return np.random.default_rng(seed).standard_normal((N, T))
def run_jac(x):
    """returns logits, hs, masks (N,T,H), jacobian of logits wrt x (N,T_out,T_in). exact: logits = J x"""
    N = x.shape[0]; h = np.zeros((N,H)); dh = np.zeros((N,H,T)); hs=[]; ms=[]
    for t in range(T):
        pre = x[:,t:t+1]*Wih + h@Whh.T
        m = (pre>0).astype(float)
        dpre = np.einsum('ij,njk->nik', Whh, dh); dpre[:,:,t] += Wih
        dh = dpre*m[:,:,None]; h = pre*m; hs.append(h); ms.append(m)
    return h@Wo.T, np.stack(hs,1), np.stack(ms,1), np.einsum('oh,nht->not', Wo, dh)
