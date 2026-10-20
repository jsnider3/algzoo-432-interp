"""Learned correction ("learned quadrature") on z46's cells.  A small MLP g maps a cell's deterministic features to
its true accuracy; a network's estimate is the mean of g over its 46,080 cells (all equally likely), so the only
per-network work is z46's 2 forward rows per cell (92,160 forwards).
  cell features: box (3 bits), S slice, positions of M, S, R and the representative Q (embeddings); the MF4 and MF3
    anchor bits I4, I3 and logit margins m4, m3, raw and divided by the network's own margin scale;
  network context (optional): the network's mean I4, mean I3, mean |m4|, mean |m3|, share of I4 != I3, margin scale.
Trained on the 896 networks outside dev (0::16) and held out (8::16), with soft binary cross-entropy against z46's
sampled cell accuracies.  Scored on dev, held out and the zoo, none seen in training.
With z46 --screen data: --screen adds the 12 screening rows to every cell (flip bits vs I4 and margins; 14 rows per
cell), and their network-level flip rates to the context; --screen-ctx uses screening rows on every 16th cell only,
for the context (2 + 12/16 rows per cell).
--boxes=n / --k=K read z46's --boxes=n / K data.  --weights adds free weight statistics to the context (per
neuron: W_ih, the self-loop, log norms of its recurrent row and column and of its readout column; mean/SD/min/max).
--hidden=a,b,... sets the MLP's hidden widths (default 128,128,64).  Inference cost is counted: per cell, the
network-dependent multiply-adds (8 continuous inputs x first width, then each later layer; the embeddings' and the
context's first-layer terms are precomputed per cell and per network), at 2,880 per forward.
--cv: 8-fold cross-fitting over all 1,024 networks (fold = i mod 8) instead of the dev/held split; zoo predictions
are averaged over the 8 fold models.  --seed=s sets the training seed.
Usage: python z47.py [epochs, default 3] [--no-context] [--screen | --screen-ctx] [--boxes=n]
"""
import sys, time, numpy as np, torch as th
args=[a for a in sys.argv[1:] if not a.startswith('--')]; ctx='--no-context' not in sys.argv
SCR='--screen' in sys.argv; SCX='--screen-ctx' in sys.argv
dev='cuda' if th.cuda.is_available() else 'cpu'; EP=int(args[0]) if args else 3; CV='--cv' in sys.argv; SEED=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--seed=')),47)); th.manual_seed(SEED)
NBa=next((a.split('=')[1] for a in sys.argv if a.startswith('--boxes=')),'2'); BS=('' if NBa=='2' else f'_b{NBa}')+next((f"_k{a.split('=')[1]}" for a in sys.argv if a.startswith('--k=') and a!='--k=8'),''); WTS='--weights' in sys.argv; HID=[int(v) for v in next((a.split('=')[1] for a in sys.argv if a.startswith('--hidden=')),'128,128,64').split(',')]
D=np.load(f"pop1k/seed0_z46_cells{'_screen' if SCR or SCX else ''}{BS}.npz"); NB=int(D['NB']) if 'NB' in D.files else 2; K=int(D['K']); ARR=D['arr']; QREP=D['qrep']; NT,C=D['I4'].shape
ref=np.load('pop1k/reference.npz'); acc=ref['acc']; pq=(acc[:1024]*(1-acc[:1024])).mean()
g=lambda k,dt=th.float32: th.tensor(D[k]).to(dev,dt)
I4,I3,m4,m3,tgt=g('I4'),g('I3'),g('m4'),g('m3'),g('tgt')
c=np.arange(C); b=c//(720*K); NBOX=NB**3; a=(c//K)%720; k=c%K
cat=th.tensor(np.stack([b%NB,(b//NB)%NB,(b//NB**2)%NB,k,ARR[a,0],ARR[a,1],ARR[a,2],QREP[a]],1),device=dev)       # (C, 8)
scale=th.sqrt((m4**2).mean(1,keepdim=True))
cx=th.stack([I4.mean(1),I3.mean(1),m4.abs().mean(1)/10,m3.abs().mean(1)/10,(I4!=I3).float().mean(1),scale[:,0]/10],1)
if SCR or SCX:
    fl=th.tensor(D['fl']).to(dev); sub=th.arange(C,device=dev) if SCR else th.arange(0,C,16,device=dev)
    flip=(fl[:,sub]!=I4[:,sub,None].to(th.uint8)).float(); r4=I4[:,sub,None]
    big,small=flip[...,1::2],flip[...,0::2]                                           # rows ordered (j, small/big)
    st=[(big*r4).sum((1,2))/r4.sum((1,2))/6,(small*r4).sum((1,2))/r4.sum((1,2))/6,(big*(1-r4)).sum((1,2))/(1-r4).sum((1,2))/6,(small*(1-r4)).sum((1,2))/(1-r4).sum((1,2))/6]
    cx=th.cat([cx,th.stack(st,1)],1)
    if SCR: fm=th.tensor(D['fm']).to(dev)
    else: del fl
if WTS:
    from z2 import load as pload
    from seeds_clocks import load as zload
    Pw,_=pload('pop1k/seed0.npz'); Zw=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
    Wi=np.r_[Pw[0].numpy(),np.stack([z[0] for z in Zw])]; Wh=np.r_[Pw[1].numpy(),np.stack([z[1] for z in Zw])]; Wo_=np.r_[Pw[2].numpy(),np.stack([z[2] for z in Zw])]
    pn=np.stack([Wi,np.diagonal(Wh,axis1=1,axis2=2),np.log(np.linalg.norm(Wh,axis=2)+1e-9),np.log(np.linalg.norm(Wh,axis=1)+1e-9),np.log(np.linalg.norm(Wo_,axis=1)+1e-9)],2)
    cx=th.cat([cx,th.tensor(np.concatenate([pn.mean(1),pn.std(1),pn.min(1),pn.max(1)],1),dtype=th.float32,device=dev)],1)
cx=(cx-cx[:1024].mean(0))/cx[:1024].std(0)
idx=np.arange(1030); test=np.r_[np.arange(0,1024,16),np.arange(8,1024,16)]; train=np.setdiff1d(np.arange(1024),test)
class G(th.nn.Module):
    def __init__(s):
        super().__init__(); s.emb=th.nn.ModuleList([th.nn.Embedding(n,4) for n in (NB,NB,NB,K,10,10,10,10)])
        d=[32+8+(cx.shape[1] if ctx else 0)+(24 if SCR else 0)]+HID; L_=[]
        for i in range(len(HID)): L_+=[th.nn.Linear(d[i],d[i+1]),th.nn.ReLU()]
        s.net=th.nn.Sequential(*L_,th.nn.Linear(d[-1],1))
    def forward(s,n,cc):
        e=th.cat([m(cat[cc,i]) for i,m in enumerate(s.emb)],1)
        f4,f3=m4[n,cc],m3[n,cc]; sc=scale[n,0]
        x=[e,I4[n,cc,None],I3[n,cc,None],(f4/sc)[:,None],(f3/sc)[:,None],(f4/sc).abs()[:,None],(f3/sc).abs()[:,None],f4.clamp(-20,20)[:,None]/10,f3.clamp(-20,20)[:,None]/10]
        if ctx: x.append(cx[n])
        if SCR: x+=[(fl[n,cc]!=I4[n,cc,None].to(th.uint8)).float(),(fm[n,cc].float()/sc[:,None]).clamp(-5,5)]
        return s.net(th.cat(x,1))[:,0]
def train_model(train,verbose=True):
    model=G().to(dev); opt=th.optim.Adam(model.parameters(),lr=2e-3); B=65536; steps=max(EP*len(train)*C//B,1500)      # at least 1,500 steps for small cell counts
    sched=th.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=steps); trn=th.tensor(train,device=dev); t0=time.time()
    for st in range(steps):
        n=trn[th.randint(len(train),(B,),device=dev)]; cc=th.randint(C,(B,),device=dev)
        loss=th.nn.functional.binary_cross_entropy_with_logits(model(n,cc),tgt[n,cc]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and st%(steps//6)==0: print(f"  step {st}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s",flush=True)
    return model.eval()
def predict(model,nets):
    with th.no_grad(): return np.array([th.sigmoid(model(th.full((C,),i,device=dev),th.arange(C,device=dev))).double().mean().item() for i in nets])
if CV:
    est=np.zeros(NT); zoo=np.zeros(6); pop=np.arange(1024)
    for f in range(8):
        m_=train_model(pop[pop%8!=f],False); est[pop[pop%8==f]]=predict(m_,pop[pop%8==f]); zoo+=predict(m_,range(1024,1030))/8
        print(f"  fold {f} done",flush=True)
    est[1024:]=zoo; e=100*(est-acc); rows=(14*C if SCR else 2*C+12*(C//16) if SCX else 2*C)
    mac=(8+(24 if SCR else 0))*HID[0]+HID[0]+sum(HID[i]*HID[i+1] for i in range(len(HID)-1))+HID[-1]+16      # per cell, incl. margins and the sigmoid
    cost=rows+C*mac/2880; samp=100*np.sqrt(pq/cost)
    print(f"  hidden {HID}: inference {mac} multiply-adds per cell = {C*mac/2880:.0f} forward-equivalents; total {cost:.0f}")
    print(f"{C} cells, {rows} forward rows + inference = {cost:.0f} forward-equivalents per network; sampling RMS {samp:.4f}; weights {'on' if WTS else 'off'}; seed {SEED}; 8-fold cross-fitting")
    for lab,ix in (('all 1,024',pop),('dev 0::16',np.arange(0,1024,16)),('held 8::16',np.arange(8,1024,16))):
        print(f"  {lab:11s} RMS {np.sqrt((e[ix]**2).mean()):.4f} mean {e[ix].mean():+.4f}  MSE/sampling {np.mean(e[ix]**2)/samp**2:.2f}")
    print(f"  zoo errors "+" ".join(f"{v:+.3f}" for v in e[1024:])+f"  RMS {np.sqrt((e[1024:]**2).mean()):.3f}  MSE/sampling {np.mean(e[1024:]**2)/samp**2:.2f}")
    np.save(f"pop1k/seed0_z47cv{BS}{'_w' if WTS else ''}{'' if HID==[128,128,64] else '_h'+'-'.join(map(str,HID))}_s{SEED}_est.npy",est); sys.exit(0)
model=train_model(train); est=np.zeros(NT); cb=th.tensor(b,device=dev); ck=th.tensor(k,device=dev); cms=th.tensor(ARR[a,0]*10+ARR[a,1],device=dev); cR=th.tensor(ARR[a,2],device=dev)
RB=np.zeros((NT,NBOX)); RK=np.zeros((NT,K)); RMS_=np.zeros((NT,100)); RI=np.zeros((NT,4))       # residual g - t by box, slice, (M,S), (I4,I3)
with th.no_grad():
    for i in range(NT):
        nn_=th.full((C,),i,device=dev); pr=th.sigmoid(model(nn_,th.arange(C,device=dev))).double(); est[i]=pr.mean().item()
        r=(pr-tgt[i].double())/C
        RB[i]=th.zeros(NBOX,device=dev,dtype=th.float64).index_add_(0,cb,r).cpu().numpy(); RK[i]=th.zeros(K,device=dev,dtype=th.float64).index_add_(0,ck,r).cpu().numpy()
        RMS_[i]=th.zeros(100,device=dev,dtype=th.float64).index_add_(0,cms,r).cpu().numpy()
        RI[i]=th.zeros(4,device=dev,dtype=th.float64).index_add_(0,(I4[i]*2+I3[i]).long(),r).cpu().numpy()
cost=(14*C if SCR else 2*C+12*(C//16) if SCX else 2*C); samp=100*np.sqrt(pq/cost); dv,hd,zo=np.arange(0,1024,16),np.arange(8,1024,16),np.arange(1024,1030)
def show(name,e):
    print(f"  {name:44s} dev RMS {np.sqrt((e[dv]**2).mean()):.4f} mean {e[dv].mean():+.4f}  held RMS {np.sqrt((e[hd]**2).mean()):.4f} mean {e[hd].mean():+.4f}"
          f"  MSE/sampling {np.mean(e[test]**2)/samp**2:6.1f}  zoo "+" ".join(f"{v:+.3f}" for v in e[zo]))
print(f"{C} cells, {cost} forwards per network; sampling RMS at that cost {samp:.4f} points; context {'on' if ctx else 'off'}; screening {'per cell' if SCR else 'context only' if SCX else 'none'}; weights {'on' if WTS else 'off'}; {EP} epochs")
mI4,mI3,mt=(v.double().mean(1).cpu().numpy() for v in (I4,I3,tgt))
X=np.c_[np.ones(NT),mI4,mI3]; bb=np.linalg.lstsq(X[train],acc[train],rcond=None)[0]
show('oracle: mean sampled cell accuracy (check)',100*(mt-acc))
show('uncalibrated mean I4 over cells',100*(mI4-acc))
show('network-level calibration of mean I4, I3',100*(X@bb-acc))
show('learned cell model',100*(est-acc))
Xl=np.c_[np.ones(NT),est]; bl=np.linalg.lstsq(Xl[train],acc[train],rcond=None)[0]
show('learned cell model, then network-level calibration',100*(Xl@bl-acc))
print("  where the learned model's network-level error sits (test networks; residual g - sampled cell accuracy; points):")
tv=lambda R: 100*np.sqrt((R[test]**2).mean(0))
tot=100*R_tot if False else None
e_t=100*np.r_[RB.sum(1)][test]; print(f"    total residual RMS {np.sqrt((e_t**2).mean()):.4f} (includes cell-target noise)")
print("    by box (digits Q,R,M piece, low to high): "+"  ".join("".join(str((bb_//NB**i)%NB) for i in range(3))+f" {v:.3f}" for bb_,v in enumerate(tv(RB))))
print("    by S slice (low to high): "+" ".join(f"{v:.3f}" for v in tv(RK)))
print("    by anchor bits (I4,I3) = (0,0) (0,1) (1,0) (1,1): "+" ".join(f"{v:.3f}" for v in tv(RI))+"   mean "+" ".join(f"{100*v:+.3f}" for v in RI[test].mean(0)))
ms=tv(RMS_); top=np.argsort(ms)[::-1][:8]; print("    largest (M pos, S pos) classes: "+"  ".join(f"({t_//10},{t_%10}) {ms[t_]:.3f}" for t_ in top))
np.savez(f"pop1k/seed0_z47{'' if ctx else '_noctx'}{'_screen' if SCR else '_screenctx' if SCX else ''}{BS}{'_w' if WTS else ''}_resid.npz",RB=RB,RK=RK,RMS=RMS_,RI=RI)
np.save(f"pop1k/seed0_z47{'' if ctx else '_noctx'}{'_screen' if SCR else '_screenctx' if SCX else ''}{BS}{'_w' if WTS else ''}_est.npy",est)
