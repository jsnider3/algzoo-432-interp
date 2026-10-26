import numpy as np, torch as th, time
from alg_zoo import DistRNN
from k1 import clean_err, best_pointer_sel, load
def finetune_faithful(path, data_seed, steps, start_step):
    Wih,Whh,Wo=load(path); m=DistRNN(hidden_size=16,seq_len=10)
    m.load_state_dict({'rnn.weight_ih_l0':th.tensor(Wih[:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(Whh,dtype=th.float32),'linear.weight':th.tensor(Wo,dtype=th.float32)})
    th.manual_seed(data_seed); opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); n_steps=2**20
    for i in range(steps):
        s=start_step+i; mult=min(s/2**14,1)*max(0.01**(2*s/n_steps),0.01)   # the original schedule, incl. warmup
        for g in opt.param_groups: g['lr']=0.01*mult
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
    return [m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
if __name__=="__main__":
    for s in [0,2]:
        for ds in [31,32,33]:
            t0=time.time(); W=finetune_faithful(f"ckpt/s{s}_e20.npz",ds,steps=2**15,start_step=2**10); a,ce=clean_err(*W); bp=best_pointer_sel(W[0],W[1])
            print(f"[original lr schedule] seed {s} from 2^20 (data seed {ds}) +2^25 seqs: acc {a:.3f} clean-err {ce:.4f} | most selective pointer neither-rate {bp[0]:.2f} {bp[1][:2] if bp[1] else None} [{time.time()-t0:.0f}s]", flush=True)
