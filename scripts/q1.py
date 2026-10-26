import sys, numpy as np, torch as th
from alg_zoo import DistRNN
from m1 import A, B
from k1 import best_pointer_sel
th.set_num_threads(1)
for ro in [int(a) for a in sys.argv[1].split(',')]:
    th.manual_seed(6000+ro); Wo_rand=DistRNN(hidden_size=16,seq_len=10).linear.weight.detach().numpy().astype(float)
    for whh_name,Whh in [("s3",A[1]),("s4",B[1])]:
        m=DistRNN(hidden_size=16,seq_len=10)
        m.load_state_dict({'rnn.weight_ih_l0':th.tensor(B[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(Whh,dtype=th.float32),'linear.weight':th.tensor(Wo_rand,dtype=th.float32)})
        opt=th.optim.AdamW([m.rnn.weight_ih_l0,m.rnn.weight_hh_l0],lr=0.01,weight_decay=0.001); th.manual_seed(91)
        for s in range(2**15):
            mult=min(max(s,1)/2**14,1)*max(0.01**(2*s/2**20),0.01)
            for g in opt.param_groups: g['lr']=0.01*mult
            x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
        bp=best_pointer_sel(m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float))
        print(f"readout {ro} (frozen) + Whh {whh_name}: pointer {bp[0]:.2f} switch={bp[0]<0.4}",flush=True)
