import sys, numpy as np, torch as th
from alg_zoo import DistRNN
from seeds_clocks import load, run, T
th.set_num_threads(2)
A=load("ckpt/s3_e0.npz"); B=load("ckpt/s4_e0.npz")
xe=np.random.default_rng(801).standard_normal((60000,T))
def delay1_neither(Wih,Whh,n):
    hs=run(xe,Wih,Whh); ne=[]; ev=[]
    for t in range(3,T):
        e=t-1; P=xe[:,:e]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]; xv=xe[:,e]; f=hs[:,t,n]>0
        ne.append(f[xv<S].mean()); ev.append(f[(xv>S)&(xv<M)].mean())
    return np.mean(ne), np.mean(ev)
def train(W, ds, steps, snaps=()):
    m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(W[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(W[1],dtype=th.float32),'linear.weight':th.tensor(W[2],dtype=th.float32)})
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(ds); out={}
    for s in range(steps):
        mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
        if s+1 in snaps: out[s+1]=(m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float).copy(), m.rnn.weight_hh_l0.detach().numpy().astype(float).copy())
    return out
if __name__=="__main__":
    full=[B[0],A[1],A[2]]; base=[B[0],B[1],B[2]]
    for nm,W in [("full s3 Whh+Wo",full),("pure s4",base)]:
        for ds in [91,95]:
            out=train(W,ds,2**15,snaps=(2048,4096,6144,8192,12288,16384,32768))
            print(nm, ds, {k:tuple(round(v,2) for v in delay1_neither(*w,13)) for k,w in out.items()}, flush=True)
