"""Cell data for a learned correction ("learned quadrature"), on all 1,030 networks.
Configuration space of the top 3 (M, S, R values and positions) plus Q = x_(4), cut into equally likely cells:
  box b: each MF4 outer coordinate (u_Q ~ Beta(7,4); (u_R-u_Q)/(1-u_Q) ~ Beta(1,3); (u_M-u_R)/(1-u_R) ~ Beta(2,1);
         independent) split at its median, 8 boxes;  S interval k: K equal slices of S's conditional CDF in [R, M];
  arrangement a: the 720 MF3 arrangements (positions of M, S, R); Q's position is uniform over the other 7.
  -> 8 x K x 720 cells, each of probability 1/(5760 K).
Per cell, deterministic features from 2 forward rows at the cell's centre (box: conditional medians, i.e. the 0.25 /
0.75 quantiles; S: the slice midpoint; Q at the representative position given by z41's balanced (R,Q) pairs):
  the MF4 anchor row (top 4 exact, the rest at mu(Q)) and the MF3 anchor row (top 3 exact, the rest at mu(R)):
  their correctness bits I4, I3 and logit margins m4, m3 (target logit minus the best other).
Per cell, the training target: the cell's true accuracy, measured by nin independent draws inside the cell (shared
across networks).  With --screen, 12 more rows per cell: the MF4 row with each of its 6 lower inputs (in position order) moved to its
largest or smallest q = 8 Gauss-Legendre node (CDF 0.980 or 0.020 of Q): bits fl (row correct) and margins fm.
--boxes=n cuts each outer coordinate into n equal-probability pieces instead of 2 (n^3 boxes).
Saves pop1k/seed0_z46_cells[_screen][_bn][_kK].npz (gitignored).  Usage: python z46.py [K, default 8] [nin, default 32] [--screen] [--boxes=n]
"""
import sys, itertools, time, numpy as np, torch as th
from batch_train import forward
from z2 import load
from seeds_clocks import load as zload
from scipy.special import ndtr, ndtri, betaincinv
args=[a for a in sys.argv[1:] if not a.startswith('--')]; SCR='--screen' in sys.argv; R0=14 if SCR else 2
NB=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--boxes=')),2)); NBOX=NB**3
dev='cuda'; K=int(args[0]) if len(args)>0 else 8; nin=int(args[1]) if len(args)>1 else 32
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
Wall=[th.cat([a,b]) for a,b in zip(P,Wz)]; NT=Wall[0].shape[0]
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
ARR=np.array(list(itertools.permutations(range(10),3))); A=len(ARR)
QOF={0:7,7:0,1:3,3:1,2:5,5:2,4:6,6:4}                                      # z41's balanced pairs, R rank -> Q rank
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; QREP=np.array([f[QOF[f.index(r[2])]] for f,r in zip(free,ARR)])
OTH=np.array([[x for x in range(10) if x not in r] for r in ARR])          # the 7 positions left for Q and the lower 6
LOW6=np.array([[x for x in o if x!=qr] for o,qr in zip(OTH,QREP)])        # the MF4 row's lower positions, in order
UN=(np.array([-0.9602898564975363,0.9602898564975363])+1)/2               # smallest and largest q=8 Gauss-Legendre node
BETA=[(7,4),(1,3),(2,1)]; rng=np.random.default_rng(46)
def coords(ub):
    """(u_Q, v_R, v_M) -> values Q, R, M (arrays)"""
    fq=ub[...,0]; fr=fq+(1-fq)*ub[...,1]; fm=fr+(1-fr)*ub[...,2]; return fq,fr,fm
cells=[]; t0=time.time()
for b in range(NBOX):
    half=[(b//NB**i)%NB for i in range(3)]
    cen=np.array([betaincinv(*BETA[i],(half[i]+0.5)/NB) for i in range(3)]); fq,fr,fm=coords(cen); Q,R,M=ndtri(fq),ndtri(fr),ndtri(fm)
    s=ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K)
    # feature rows: (A, K, 2, 10)
    X=np.empty((A,K,R0,10)); ai=np.arange(A)[:,None]; ki=np.arange(K)[None,:]
    X[:,:,0]=mu(Q); X[ai,ki,0,ARR[:,0,None]]=M; X[ai,ki,0,ARR[:,1,None]]=s; X[ai,ki,0,ARR[:,2,None]]=R; X[ai,ki,0,QREP[:,None]]=Q
    X[:,:,1]=mu(R); X[ai,ki,1,ARR[:,0,None]]=M; X[ai,ki,1,ARR[:,1,None]]=s; X[ai,ki,1,ARR[:,2,None]]=R
    if SCR:
        for j in range(6):
            for h in range(2):
                rr=2+2*j+h; X[:,:,rr]=X[:,:,0]; X[ai,ki,rr,LOW6[:,j,None]]=ndtri(fq*UN[h])
    # target draws inside the cell: (A, K, nin, 10)
    U=(rng.random((A,K,nin,3))+np.array(half))/NB; ub=np.stack([betaincinv(*BETA[i],U[...,i]) for i in range(3)],-1)
    dq,dr,dm=coords(ub); us=dr+(dm-dr)*(np.arange(K)[None,:,None]+rng.random((A,K,nin)))/K
    T=np.empty((A,K,nin,10)); qi=rng.integers(0,7,(A,K,nin)); perm=np.argsort(rng.random((A,K,nin,7)),-1)
    pos=np.take_along_axis(np.broadcast_to(OTH[:,None,None,:],(A,K,nin,7)),perm,-1)       # random order of the 7 others
    T[np.arange(A)[:,None,None],np.arange(K)[None,:,None],np.arange(nin)[None,None,:],ARR[:,None,None,0]]=ndtri(dm)
    T[np.arange(A)[:,None,None],np.arange(K)[None,:,None],np.arange(nin)[None,None,:],ARR[:,None,None,1]]=ndtri(us)
    T[np.arange(A)[:,None,None],np.arange(K)[None,:,None],np.arange(nin)[None,None,:],ARR[:,None,None,2]]=ndtri(dr)
    low=ndtri(dq[...,None]*rng.random((A,K,nin,7))); low[...,0]=ndtri(dq)                 # slot 0 of the shuffled 7 holds Q
    np.put_along_axis(T,pos,low,-1)
    cells.append((X.astype(np.float32),T.astype(np.float32)))
print(f"inputs built: {time.time()-t0:.0f}s",flush=True)
C=NBOX*K*A; fl=np.zeros((NT,C,12),np.uint8) if SCR else None; fm=np.zeros((NT,C,12),np.float16) if SCR else None; I4=np.zeros((NT,C),np.uint8); I3=np.zeros((NT,C),np.uint8); m4=np.zeros((NT,C),np.float32); m3=np.zeros((NT,C),np.float32); tgt=np.zeros((NT,C),np.float16)
def margin(L,y):
    yt=y[...,None]; return (L.gather(-1,yt)[...,0]-L.scatter(-1,yt,-1e30).max(-1).values)
for c in range(0,NT,206):
    sl=slice(c,min(NT,c+206)); W=[w[sl].to(dev) for w in Wall]; N=W[0].shape[0]
    for b,(X,T) in enumerate(cells):
        for st in range(0,A,60):
            e=min(A,st+60); n_=e-st; RW=n_*K*R0; inp=np.concatenate([X[st:e].reshape(-1,10),T[st:e].reshape(-1,10)])
            with th.no_grad():
                L=forward(*W,th.tensor(inp,device=dev)[None].expand(N,-1,-1))
                Lf=L[:,:RW].view(N,n_,K,R0,10); y=th.tensor(ARR[st:e,1],device=dev)[None,:,None,None].expand(N,n_,K,R0)
                ok=(Lf.argmax(-1)==y); mg=margin(Lf,y)
                yt=th.tensor(ARR[st:e,1],device=dev)[None,:,None,None].expand(N,n_,K,nin)
                tg=(L[:,RW:].view(N,n_,K,nin,10).argmax(-1)==yt).float().mean(-1)
            cs=slice((b*A+st)*K,(b*A+e)*K)           # cell index = (b*720 + a)*K + k
            I4[sl,cs]=ok[...,0].reshape(N,-1).cpu().numpy(); I3[sl,cs]=ok[...,1].reshape(N,-1).cpu().numpy()
            m4[sl,cs]=mg[...,0].reshape(N,-1).cpu().numpy(); m3[sl,cs]=mg[...,1].reshape(N,-1).cpu().numpy(); tgt[sl,cs]=tg.reshape(N,-1).cpu().numpy()
            if SCR: fl[sl,cs]=ok[...,2:].reshape(N,-1,12).cpu().numpy(); fm[sl,cs]=mg[...,2:].reshape(N,-1,12).cpu().numpy().astype(np.float16)
    print(f"networks {c}-{sl.stop}: {time.time()-t0:.0f}s",flush=True)
np.savez(f"pop1k/seed0_z46_cells{'_screen' if SCR else ''}{'' if NB==2 else f'_b{NB}'}{'' if K==8 else f'_k{K}'}.npz",NB=NB,I4=I4,I3=I3,m4=m4,m3=m3,tgt=tgt,K=K,nin=nin,arr=ARR,qrep=QREP,**({'fl':fl,'fm':fm} if SCR else {}))
ref=np.load('pop1k/reference.npz'); e=100*(tgt.astype(float).mean(1)-ref['acc'])
print(f"{C} cells x {R0} feature rows ({R0*C} forwards per network); targets {nin} draws per cell")
print(f"check, mean cell target vs reference accuracy (a stratified sampling estimate): RMS {np.sqrt((e[:1024]**2).mean()):.4f} mean {e[:1024].mean():+.4f} points")
for name,v in (('mean I4 (box-centre MF4 rule)',I4),('mean I3 (box-centre MF3 rule)',I3)):
    e=100*(v.astype(float).mean(1)-ref['acc']); print(f"  uncalibrated {name}: RMS {np.sqrt((e[:1024]**2).mean()):.3f} mean {e[:1024].mean():+.3f}")
