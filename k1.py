import numpy as np, torch as th, sys, time
from alg_zoo import DistRNN
from seeds_clocks import load, run, T
xe=np.random.default_rng(201).standard_normal((100000,T)); ye=np.argsort(xe,1)[:,-2]
def best_pointer_sel(Wih,Whh):
    hs=run(xe,Wih,Whh); best=(9,None)
    for dl in [0,1]:
        r={'n2':[], 'ne':[]}
        rates=np.zeros((2,16))
        for t in range(3,T):
            e=t-dl; P=xe[:,:e]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]; x_=xe[:,e]; f=hs[:,t]>0
            rates[0]+=f[(x_>S)&(x_<M)].mean(0)/(T-3); rates[1]+=f[x_<S].mean(0)/(T-3)
        for n in range(16):
            if rates[0,n]>=0.8 and rates[1,n]<best[0]: best=(rates[1,n],(n,dl,rates[0,n]))
    return best
def clean_err(Wih,Whh,Wo):
    s=-np.sort(-xe,1); M,S,R=s[:,0],s[:,1],s[:,2]; c=(S>0)&((M-S)>.1*M)&((S-R)>.1*M)
    p=(run(xe,Wih,Whh)[:,-1]@Wo.T).argmax(1); return (p==ye).mean(), (p!=ye)[c].mean()
def finetune(path, data_seed, steps=2**14, start_step=2**14):
    Wih,Whh,Wo=load(path); m=DistRNN(hidden_size=16,seq_len=10)
    sd={'rnn.weight_ih_l0':th.tensor(Wih[:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(Whh,dtype=th.float32),'linear.weight':th.tensor(Wo,dtype=th.float32)}
    m.load_state_dict(sd); th.manual_seed(data_seed)
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); n_steps=2**20
    lr_at=lambda s: min((s-start_step)/500,1)*max(0.01**(2*s/n_steps),0.01)
    for i in range(steps):
        s=start_step+i
        for g in opt.param_groups: g['lr']=0.01*lr_at(s)
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
    return W
if __name__=="__main__":
    for s in [2,0]:
        path=f"ckpt/s{s}_e24.npz"; W0=load(path)
        a,ce=clean_err(*W0); bp=best_pointer_sel(W0[0],W0[1])
        print(f"\nseed {s} @2^24 start: acc {a:.3f} clean-err {ce:.4f} | most selective pointer: neither-rate {bp[0]:.2f} {bp[1]}")
        W25=load(f"ckpt/s{s}_e25.npz"); a,ce=clean_err(*W25); bp=best_pointer_sel(W25[0],W25[1])
        print(f"  original run @2^25:   acc {a:.3f} clean-err {ce:.4f} | most selective pointer: neither-rate {bp[0]:.2f} {bp[1]}")
        for ds in [11,12,13]:
            t0=time.time(); W=finetune(path,ds); a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
            print(f"  continued (data seed {ds}) +2^24 seqs: acc {a:.3f} clean-err {ce:.4f} | most selective pointer: neither-rate {bp[0]:.2f} {bp[1]}  [{time.time()-t0:.0f}s]", flush=True)
