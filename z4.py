"""Error anatomy across the population: where in input space each network's errors fall, and which of those places
carry the differences between networks (the part of accuracy a circuit model must explain).

Per input: top-3 values M, S, R and their ages (9 - position), gaps a = (M-S)/|M|, b = (S-R)/|M|, s = S/M, the z3
region, and what happened at the last step. Per network: the error mask on a shared pool.
Prints, for each partition of the clean and middle regions, each cell's mass, population error rate, and its share
of the across-network variance of total accuracy (cov(cell contribution, total error) / var(total error), which sums
to 1 over a partition of all inputs).

Usage: python z4.py pop1k/snap_2e28.npz [--pool 65536] -> pop1k/z4_<snap>.npz (masks + input variables)
"""
import argparse, os, numpy as np, torch as th
from batch_train import forward, target
from z2 import load

def input_vars(x):
    o=np.argsort(-x,1); T=x.shape[1]; v=np.take_along_axis(x,o[:,:3],1); M,S,R=v.T
    pM,pS,pR=o[:,0],o[:,1],o[:,2]; a=(M-S)/np.abs(M); b=(S-R)/np.abs(M)
    reg=np.full(len(x),3); reg[S<0]=0; reg[(S>=0)&(a<.05)]=1; reg[(S>=0)&(a>=.05)&(b<.03)]=2
    reg[(S>=0)&(a>=.1)&(b>=.1)]=4
    # event at the last step: 0 new max, 1 new 2nd, 2 neither
    last=np.where(pM==T-1,0,np.where(pS==T-1,1,2))
    return dict(M=M,S=S,R=R,aM=T-1-pM,aS=T-1-pS,aR=T-1-pR,a=a,b=b,s=S/M,r=R/M,reg=reg,last=last,
                order=np.where(pM<pS,0,1))                        # 0: max came first (S is a direct new 2nd)

@th.no_grad()
def error_masks(P, x, chunk=8192):
    N=P[0].shape[0]; out=np.zeros((N,len(x)),bool); xt=th.tensor(x)
    for i in range(0,len(x),chunk):
        xc=xt[i:i+chunk]; out[:,i:i+chunk]=(forward(*P,xc[None].expand(N,-1,-1)).argmax(-1)!=target(xc)[None]).numpy()
    return out

def share_table(E, cells, names, tot):
    """per cell: mass, mean error rate, mean contribution (points), share of across-network accuracy variance"""
    n=E.shape[1]; vt=tot.var(); rows=[]
    for c,nm in zip(cells,names):
        if c.sum()==0: continue
        con=E[:,c].sum(1)/n; rows.append((nm,c.mean(),E[:,c].mean(),con.mean()*100,np.cov(con,tot)[0,1]/vt))
    return rows

def show(title, rows):
    print(f"\n{title}\n  {'cell':22s} {'mass':>7s} {'err':>7s} {'contrib':>8s} {'var share':>9s}")
    for nm,m,e,c,sh in rows: print(f"  {nm:22s} {m*100:6.2f}% {e*100:6.2f}% {c:8.3f} {sh*100:8.1f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("pop"); ap.add_argument("--pool",type=int,default=2**16)
    ap.add_argument("--threads",type=int,default=6); a=ap.parse_args(); th.set_num_threads(a.threads)
    P,s=load(a.pop); x=np.random.default_rng(12).standard_normal((a.pool,10)).astype(np.float32)
    E=error_masks(P,x); V=input_vars(x); tot=E.mean(1)
    print(f"{E.shape[0]} networks, pool {a.pool}; accuracy {1-tot.mean():.4f} +- {tot.std()*100:.2f} points")
    R=["S<0","M-S tie","S-R tie","middle","clean"]
    show("regions",share_table(E,[V["reg"]==k for k in range(5)],R,tot))
    for k,rn in ((4,"clean"),(3,"middle")):
        m=V["reg"]==k
        show(f"{rn}: by age of 2nd max",share_table(E,[m&(V["aS"]==j) for j in range(10)],[f"aS={j}" for j in range(10)],tot))
        show(f"{rn}: by age of max",share_table(E,[m&(V["aM"]==j) for j in range(10)],[f"aM={j}" for j in range(10)],tot))
        sb=[0,.1,.2,.4,.6,.8,1]; show(f"{rn}: by S/M",share_table(E,[m&(V["s"]>=lo)&(V["s"]<hi) for lo,hi in zip(sb,sb[1:])],[f"S/M {lo}-{hi}" for lo,hi in zip(sb,sb[1:])],tot))
        show(f"{rn}: by last step",share_table(E,[m&(V["last"]==j) for j in range(3)],["last=new max","last=new 2nd","last=neither"],tot))
        show(f"{rn}: by order",share_table(E,[m&(V["order"]==j) for j in range(2)],["max first (direct 2nd)","2nd first (handoff)"],tot))
    d,f=os.path.split(a.pop); np.savez_compressed(os.path.join(d,"z4_"+f),E=np.packbits(E,1),x=x,snapshot=s)
