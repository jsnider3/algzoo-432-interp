"""Train many AlgZoo-style 2nd-argmax RNNs at once on one GPU.

Each model is an independent DistRNN (one-layer ReLU RNN, no biases, input size 1, readout to seq_len logits),
with its own PyTorch-default initialization and its own data stream. AlgZoo's recipe (alg_zoo/training.py) is
reproduced: AdamW(lr=0.04/sqrt(H), weight_decay=0.001), batch 1024, 2^14 warmup steps, exponential decay to 0.01x
over the first half of training, cross-entropy on argsort[-2]. Because AdamW and weight decay act elementwise and
the total loss is a sum of per-model mean losses, one optimizer over the stacked tensors is exactly N independent
runs.

Usage:
  python scripts/batch_train.py --bench                       # throughput for a few population sizes
  python scripts/batch_train.py --check                       # batched step == nn.RNN step, same init and data
  python scripts/batch_train.py --n 1024 --log2-seqs 30 --seed 0 --compile --out pop1k/seed0.npz   # resumes if interrupted
"""
import argparse, math, os, time, numpy as np, torch as th

def lr_mult(step, n_steps, warmup=2**14, min_mult=0.01):
    return min(step/warmup, 1)*max(min_mult**(2*step/n_steps), min_mult)

def init_params(n, H=16, T=10, gen=None, device="cuda"):
    """PyTorch defaults: nn.RNN weights ~ U(-1/sqrt(H), 1/sqrt(H)); nn.Linear (fan_in H) uses the same bound."""
    b=1/math.sqrt(H); u=lambda *s: (th.rand(*s,generator=gen,device=device)*2-1)*b
    return [u(n,H), u(n,H,H), u(n,T,H)]

def forward(Wih, Whh, Wo, x):
    """x: (N,B,T) -> logits (N,B,T_out)"""
    h=th.zeros(x.shape[0],x.shape[1],Whh.shape[1],device=x.device,dtype=x.dtype); WhhT=Whh.transpose(1,2)
    for t in range(x.shape[2]):
        h=th.relu(th.baddbmm(x[:,:,t,None]*Wih[:,None,:], h, WhhT))
    return th.bmm(h, Wo.transpose(1,2))

def target(x):
    """= argsort(x)[..., -2], computed as the argmax after masking out the argmax (much faster on GPU)"""
    return x.scatter(-1, x.argmax(-1,keepdim=True), float('-inf')).argmax(-1)

def step_loss(Wih, Whh, Wo, x):
    """sum over models of each model's mean cross-entropy"""
    L=forward(Wih,Whh,Wo,x); T=L.shape[-1]
    return th.nn.functional.cross_entropy(L.reshape(-1,T), target(x).reshape(-1), reduction='sum')/x.shape[1]

def train(params, log2_seqs=30, batch=1024, H=16, seed=0, snaps=(), log_every=2**14, eval_x=None,
          ckpt_path=None, ckpt_every=2**15, compile=False, stop_after=None):
    """params: list of (N,...) tensors. Returns {step: [np arrays]} for steps in snaps and the final step.
    With ckpt_path, a resumable state (weights, AdamW moments, schedule, data RNG, snapshots) is saved every
    ckpt_every steps, and an existing one is resumed from."""
    params=[p.detach().clone().requires_grad_(True) for p in params]
    N,T=params[2].shape[:2]; n_steps=2**log2_seqs//batch; dev=params[0].device
    opt=th.optim.AdamW(params, lr=0.04/math.sqrt(H), weight_decay=0.001)
    sched=th.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_mult(s,n_steps))
    gen=th.Generator(device=dev); gen.manual_seed(seed); out={}; start=1
    if ckpt_path and os.path.exists(ckpt_path):
        st=th.load(ckpt_path, weights_only=False)
        with th.no_grad():
            for p,q in zip(params,st['params']): p.copy_(q)
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); gen.set_state(st['gen'].cpu() if hasattr(st['gen'],'cpu') else st['gen'])
        out=st['out']; start=st['step']+1; print(f"resumed from step {st['step']}", flush=True)
    f=th.compile(step_loss) if compile else step_loss; t0=time.time()
    for step in range(start,n_steps+1):
        x=th.randn(N,batch,T,generator=gen,device=dev)
        loss=f(*params,x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step in snaps: out[step]=[p.detach().cpu().numpy().copy() for p in params]
        if log_every and step%log_every==0:
            msg=f"step {step}/{n_steps} mean loss {loss.item()/N:.4f} [{time.time()-t0:.0f}s]"
            if eval_x is not None: msg+=f" mean acc {accuracy(params,eval_x).mean():.4f}"
            print(msg, flush=True)
        if ckpt_path and (step%ckpt_every==0 or step==n_steps):
            th.save(dict(params=[p.detach() for p in params], opt=opt.state_dict(), sched=sched.state_dict(),
                         gen=gen.get_state(), out=out, step=step), ckpt_path+".tmp")
            os.replace(ckpt_path+".tmp", ckpt_path)
        if stop_after and step==stop_after: return out
    out[n_steps]=[p.detach().cpu().numpy().copy() for p in params]
    return out

@th.no_grad()
def accuracy(params, x):
    """x: (B,T) shared eval inputs -> (N,) accuracies; chunked to keep N x chunk at 2^24"""
    N=params[0].shape[0]; ok=th.zeros(N,device=x.device); chunk=max(1,2**24//N)
    for i in range(0,len(x),chunk):
        xc=x[i:i+chunk]; L=forward(*params, xc[None].expand(N,-1,-1))
        ok+=(L.argmax(-1)==target(xc)[None]).float().sum(1)
    return (ok/len(x)).cpu().numpy()

def check():
    """one batched model vs nn.RNN + nn.Linear, same init and data, 200 AdamW steps"""
    th.manual_seed(0); dev="cuda"; H,T,B=16,10,1024
    rnn=th.nn.RNN(1,H,nonlinearity="relu",bias=False,batch_first=True).to(dev); lin=th.nn.Linear(H,T,bias=False).to(dev)
    P=[rnn.weight_ih_l0.detach()[:,0][None].clone(), rnn.weight_hh_l0.detach()[None].clone(), lin.weight.detach()[None].clone()]
    P=[p.requires_grad_(True) for p in P]
    o1=th.optim.AdamW(list(rnn.parameters())+list(lin.parameters()),lr=0.01,weight_decay=0.001); o2=th.optim.AdamW(P,lr=0.01,weight_decay=0.001)
    g=th.Generator(device=dev); g.manual_seed(1)
    for s in range(200):
        x=th.randn(B,T,generator=g,device=dev); y=x.argsort(-1)[:,-2]
        l1=th.nn.functional.cross_entropy(lin(rnn(x[...,None])[1][0]),y); o1.zero_grad(); l1.backward(); o1.step()
        l2=th.nn.functional.cross_entropy(forward(*P,x[None])[0],y); o2.zero_grad(); l2.backward(); o2.step()
    d=max((rnn.weight_hh_l0-P[1][0]).abs().max().item(), (lin.weight-P[2][0]).abs().max().item(), (rnn.weight_ih_l0[:,0]-P[0][0]).abs().max().item())
    print(f"after 200 steps: loss nn.RNN {l1.item():.6f} vs batched {l2.item():.6f}; max |weight diff| {d:.2e}")
    x=th.randn(64,4096,10,device=dev); print("fast target == argsort target:", bool((target(x)==x.argsort(-1)[...,-2]).all()))
    # interrupted-and-resumed run == uninterrupted run; compiled step == eager step
    import tempfile; P0=init_params(8); ck=os.path.join(tempfile.mkdtemp(),"r.pt")
    A=train(P0,log2_seqs=19,seed=3,log_every=0)[512]
    train(P0,log2_seqs=19,seed=3,log_every=0,ckpt_path=ck,ckpt_every=256,stop_after=256)
    B=train(P0,log2_seqs=19,seed=3,log_every=0,ckpt_path=ck,ckpt_every=256)[512]
    C=train(P0,log2_seqs=19,seed=3,log_every=0,compile=True)[512]
    print("resume: max |diff|", max(np.abs(a-b).max() for a,b in zip(A,B)), "| compiled: max |diff|", max(np.abs(a-c).max() for a,c in zip(A,C)))

def bench(sizes=(1024,2048), steps=64, compile=False):
    for n in sizes:
        P=init_params(n)
        if compile: train(P, log2_seqs=13, log_every=0, compile=True)      # warm up the compile cache
        th.cuda.synchronize(); t0=time.time()
        k=int(math.log2(steps)); train(P, log2_seqs=10+k, log_every=0, compile=compile); th.cuda.synchronize(); dt=(time.time()-t0)/2**k
        full=dt*2**20/3600
        print(f"N={n:5d}: {dt*1e3:6.1f} ms/step -> one 2^30-sequence run of all {n}: {full:5.1f} h; "
              f"10,000 models at 2^30: {full*10000/n:6.1f} h, at 2^27: {full*10000/n/8:5.1f} h | peak mem {th.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        th.cuda.reset_peak_memory_stats()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--bench",action="store_true"); ap.add_argument("--check",action="store_true"); ap.add_argument("--compile",action="store_true")
    ap.add_argument("--n",type=int,default=1024); ap.add_argument("--log2-seqs",type=int,default=30)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--out",default=None)
    a=ap.parse_args()
    if a.check: check()
    elif a.bench: bench(compile=a.compile)
    else:
        g=th.Generator(device="cuda"); g.manual_seed(10_000+a.seed); P0=init_params(a.n,gen=g)
        ex=th.randn(2**16,10,generator=g,device="cuda")
        n_steps=2**a.log2_seqs//1024; snaps=tuple(2**k//1024 for k in range(20,a.log2_seqs,2))
        dst=a.out or f"pop_seed{a.seed}.npz"; os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        out=train(P0,a.log2_seqs,seed=a.seed,snaps=snaps,eval_x=ex,log_every=2**12,ckpt_path=dst+".resume.pt",compile=a.compile)
        P=[th.tensor(p,device="cuda") for p in out[n_steps]]
        xe=th.randn(2**20,10,generator=g,device="cuda"); acc=accuracy(P,xe)
        save={f"init_{k}":p.cpu().numpy() for k,p in zip(["Wih","Whh","Wo"],P0)}
        for st,W in out.items():
            for k,p in zip(["Wih","Whh","Wo"],W): save[f"s{st}_{k}"]=p
        np.savez(dst, acc_2e20=acc, **save)
        print(f"saved; accuracy on 2^20 inputs: mean {acc.mean():.4f}, quantiles 5/50/95% {np.percentile(acc,[5,50,95]).round(4)}")
