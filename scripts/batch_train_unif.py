"""Train many 2nd-argmax RNNs at once with the recipe of the AlgZoo mystery model (example_2nd_argmax), as given in
Zihao Chen's notebook linked from ARC's AlgZoo post (refs/RNN_2nd_Argmax_notebook.ipynb):
  inputs Unif(-1,1)^10, target argsort[-2]; 16 hidden ReLU neurons, no biases, PyTorch default init;
  Adam(lr=0.005, weight_decay=1e-14); ExponentialLR(gamma=0.995) stepped once per epoch;
  1,000 epochs of 100,000 fresh sequences (steps per epoch = 100,000 // batch); cross-entropy on the final logits.
The notebook never defines its batch size, so it is an argument here.  Each model has its own init and data stream;
Adam acts elementwise and the total loss is a sum of per-model mean losses, so one optimizer over the stacked
tensors is N independent runs (checked by --check against nn.RNN + nn.Linear).
Saves init_*, s<epoch>_* snapshots (epochs 10, 30, 100, 300 and 1,000) and acc_gauss / acc_unif on 2^20 inputs.
Usage: python scripts/batch_train_unif.py --n 64 --batch 128 --seed 0 --out pop_unif/b128_seed0.npz
       python scripts/batch_train_unif.py --check | --bench
"""
import argparse, os, time, numpy as np, torch as th
from batch_train import forward, target, init_params, accuracy

def step_loss(Wih, Whh, Wo, x):
    L=forward(Wih,Whh,Wo,x); T=L.shape[-1]
    return th.nn.functional.cross_entropy(L.reshape(-1,T), target(x).reshape(-1), reduction='sum')/x.shape[1]

def train(params, batch, epochs=1000, epoch_seqs=100_000, seed=0, snaps=(10,30,100,300), log_every=10, ckpt_path=None, eval_x=None):
    params=[p.detach().clone().requires_grad_(True) for p in params]; N,T=params[2].shape[:2]; dev=params[0].device
    opt=th.optim.Adam(params, lr=0.005, weight_decay=1e-14); sched=th.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    gen=th.Generator(device=dev); gen.manual_seed(seed); out={}; start=0; spe=epoch_seqs//batch
    if ckpt_path and os.path.exists(ckpt_path):
        st=th.load(ckpt_path, weights_only=False)
        with th.no_grad():
            for p,q in zip(params,st['params']): p.copy_(q)
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); gen.set_state(st['gen']); out=st['out']; start=st['epoch']
        print(f"resumed after epoch {start}", flush=True)
    t0=time.time()
    for ep in range(start,epochs):
        tot=0.
        for _ in range(spe):
            x=th.rand(N,batch,T,generator=gen,device=dev)*2-1
            loss=step_loss(*params,x); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot+=loss.detach()
        sched.step(); e=ep+1
        if e in snaps: out[e]=[p.detach().cpu().numpy().copy() for p in params]
        if log_every and e%log_every==0:
            msg=f"epoch {e}/{epochs} mean loss {tot.item()/spe/N:.4f} lr {sched.get_last_lr()[0]:.2e} [{time.time()-t0:.0f}s]"
            if eval_x is not None: msg+=f" mean acc (Gaussian) {accuracy(params,eval_x).mean():.4f}"
            print(msg, flush=True)
        if ckpt_path and (e%25==0 or e==epochs):
            th.save(dict(params=[p.detach() for p in params], opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(), out=out, epoch=e), ckpt_path+".tmp")
            os.replace(ckpt_path+".tmp", ckpt_path)
    out[epochs]=[p.detach().cpu().numpy().copy() for p in params]
    return out

def check():
    """one batched model vs nn.RNN + nn.Linear under Adam on uniform data, 300 steps"""
    th.manual_seed(0); dev="cuda"; H,T,B=16,10,128
    rnn=th.nn.RNN(1,H,nonlinearity="relu",bias=False,batch_first=True).to(dev); lin=th.nn.Linear(H,T,bias=False).to(dev)
    P=[rnn.weight_ih_l0.detach()[:,0][None].clone(), rnn.weight_hh_l0.detach()[None].clone(), lin.weight.detach()[None].clone()]
    P=[p.requires_grad_(True) for p in P]
    o1=th.optim.Adam(list(rnn.parameters())+list(lin.parameters()),lr=0.005,weight_decay=1e-14); o2=th.optim.Adam(P,lr=0.005,weight_decay=1e-14)
    g=th.Generator(device=dev); g.manual_seed(1)
    for s in range(300):
        x=th.rand(B,T,generator=g,device=dev)*2-1; y=x.sort(-1)[1][:,-2]
        l1=th.nn.functional.cross_entropy(lin(rnn(x[...,None])[1][0]),y); o1.zero_grad(); l1.backward(); o1.step()
        l2=step_loss(*P,x[None]); o2.zero_grad(); l2.backward(); o2.step()
    d=max((rnn.weight_hh_l0-P[1][0]).abs().max().item(), (lin.weight-P[2][0]).abs().max().item(), (rnn.weight_ih_l0[:,0]-P[0][0]).abs().max().item())
    print(f"after 300 Adam steps: loss nn.RNN {l1.item():.6f} vs batched {l2.item():.6f}; max |weight diff| {d:.2e}")

def bench():
    for n,b in ((64,64),(64,128),(64,256),(64,1024),(256,128),(256,1024)):
        P=init_params(n); g=th.Generator(device='cuda'); g.manual_seed(0); th.cuda.synchronize(); t0=time.time()
        train(P,b,epochs=1,epoch_seqs=b*100,log_every=0); th.cuda.synchronize(); dt=(time.time()-t0)/100
        print(f"N={n:4d} batch {b:5d}: {dt*1e3:5.1f} ms/step -> 1,000 epochs ({1000*(100_000//b):,} steps): {dt*1000*(100_000//b)/3600:5.2f} h", flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--check",action="store_true"); ap.add_argument("--bench",action="store_true")
    ap.add_argument("--n",type=int,default=64); ap.add_argument("--batch",type=int,default=128); ap.add_argument("--epochs",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--out",default=None); a=ap.parse_args()
    if a.check: check()
    elif a.bench: bench()
    else:
        g=th.Generator(device="cuda"); g.manual_seed(20_000+a.seed); P0=init_params(a.n,gen=g)
        ex=th.randn(2**14,10,generator=g,device="cuda")
        dst=a.out or f"pop_unif/b{a.batch}_seed{a.seed}.npz"; os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        out=train(P0,a.batch,a.epochs,seed=a.seed,eval_x=ex,ckpt_path=dst+".resume.pt")
        P=[th.tensor(p,device="cuda") for p in out[a.epochs]]
        xg=th.randn(2**20,10,generator=g,device="cuda"); xu=th.rand(2**20,10,generator=g,device="cuda")*2-1
        ag,au=accuracy(P,xg),accuracy(P,xu)
        save={f"init_{k}":p.cpu().numpy() for k,p in zip(["Wih","Whh","Wo"],P0)}
        for st,W in out.items():
            for k,p in zip(["Wih","Whh","Wo"],W): save[f"s{st}_{k}"]=p
        np.savez(dst, acc_gauss=ag, acc_unif=au, batch=a.batch, **save)
        print(f"saved {dst}; accuracy on 2^20 inputs, Gaussian: mean {ag.mean():.4f} (5/50/95% {np.percentile(ag,[5,50,95]).round(4)}); "
              f"uniform: mean {au.mean():.4f} (5/50/95% {np.percentile(au,[5,50,95]).round(4)})")
