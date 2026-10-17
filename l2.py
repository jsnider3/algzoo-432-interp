import sys, json, numpy as np, torch as th
from alg_zoo import DistRNN
from seeds_clocks import load, run, T
from k1 import best_pointer_sel
th.set_num_threads(1)
xe=np.random.default_rng(401).standard_normal((60000,T))
def neuron_stats(Wih,Whh,n):
    hs=run(xe,Wih,Whh); ne=[]; fits=[]
    for t in range(3,T):
        P=xe[:,:t]; s=-np.sort(-P,1); M,S=s[:,0],s[:,1]; x_=xe[:,t]; ne.append((hs[:,t,n]>0)[x_<S].mean())
        if t==7:
            thr=-(hs[:,t-1]@Whh[n])/(Wih[n] if abs(Wih[n])>1e-6 else 1e-6); A=np.c_[M,S,np.ones(len(xe))]; c,*_=np.linalg.lstsq(A,thr,rcond=None)
            fits=[float(c[0]),float(c[1]),float(1-((A@c-thr)**2).mean()/thr.var())]
    return float(np.mean(ne)), fits
def train(m, data_seed, start, steps, kick=False, snaps=()):
    opt=th.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.001); th.manual_seed(data_seed); out={}
    for i in range(steps):
        s=start+i
        warm = min((i+1)/500,1) if kick else min(max(s,1)/2**14,1)
        for g in opt.param_groups: g['lr']=0.01*warm*max(0.01**(2*s/2**20),0.01)
        x=th.randn(1024,10); loss=th.nn.functional.cross_entropy(m(x),x.argsort(-1)[:,-2]); opt.zero_grad(); loss.backward(); opt.step()
        if s+1 in snaps: out[s+1]=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float).copy(), m.rnn.weight_hh_l0.detach().numpy().astype(float).copy()]
    return out
mode=sys.argv[1]
if mode=="fresh":
    for init in [int(a) for a in sys.argv[2].split(',')]:
        th.manual_seed(3000+init); m=DistRNN(hidden_size=16,seq_len=10)
        snaps=train(m,8000+init,0,2**15,snaps=(1024,2048,4096,8192,2**15))
        Wf=snaps[2**15]; bp=best_pointer_sel(*Wf); sw=bool(bp[0]<0.4)
        lead=int(np.argmax(snaps[4096][0]))     # leading input-weight neuron at step 4096
        rec=dict(init=init,switch=sw,final_pointer=bp[1][0] if bp[1] else -1,lead4096=lead)
        for k in (1024,2048,4096,8192,2**15):
            ne,f=neuron_stats(*snaps[k],lead); rec[f"lead_neither@{k}"]=round(ne,3); rec[f"lead_thrS@{k}"]=round(f[1],2); rec[f"lead_thrM@{k}"]=round(f[0],2); rec[f"lead_R2@{k}"]=round(f[2],2)
        print(json.dumps(rec),flush=True)
else:
    W=load("ckpt/s0_e22.npz")
    for kick in [False,True]:
        m=DistRNN(hidden_size=16,seq_len=10)
        m.load_state_dict({'rnn.weight_ih_l0':th.tensor(W[0][:,None],dtype=th.float32),'rnn.weight_hh_l0':th.tensor(W[1],dtype=th.float32),'linear.weight':th.tensor(W[2],dtype=th.float32)})
        st=2**12; snaps=train(m,21,st,2**14,kick=kick,snaps=tuple(st+k for k in (256,1024,2048,4096,8192,2**14)))
        fin=snaps[st+2**14]; bp=best_pointer_sel(*fin); n=bp[1][0] if bp[1] else 0
        if bp[0]>=0.4: n=int(np.argmax(fin[0]))
        print(f"seed0 from 2^22, kick={kick}: final pointer neither-rate {bp[0]:.2f}; tracking neuron n{n}:",flush=True)
        for k,Wk in snaps.items():
            ne,f=neuron_stats(*Wk,n); print(f"   step {k}: n{n} Wih {Wk[0][n]:+.2f} neither {ne:.2f} thr≈{f[0]:+.2f}M{f[1]:+.2f}S (R2 {f[2]:.2f})",flush=True)
