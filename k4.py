import numpy as np, torch as th, time
from alg_zoo import DistRNN
from k1 import clean_err, best_pointer_sel, load
def train_from(path, data_seed, steps, warmup):
    Wih,Whh,Wo=load(path); m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(Wih[:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(Whh,dtype=th.float32),'linear.weight':th.tensor(Wo,dtype=th.float32)})
    th.manual_seed(data_seed); opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); n_steps=2**20
    for s in range(steps):
        mult=min(max(s,1)/warmup,1)*max(0.01**(2*s/n_steps),0.01)
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    return [m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
if __name__=="__main__":
    for s in [0,1,4]:
        for warm in [2**14,500]:
            t0=time.time(); W=train_from(f"ckpt/s{s}_e0.npz",41,steps=2**15,warmup=warm); a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
            print(f"seed {s} init, warmup {warm:5d} steps, 2^25 seqs: acc {a:.3f} clean-err {ce:.4f} | most selective pointer neither-rate {bp[0]:.2f} {bp[1][:2] if bp[1] else None} [{time.time()-t0:.0f}s]", flush=True)
