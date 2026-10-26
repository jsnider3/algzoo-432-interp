"""Independent check of ChatGPT round 6's exact last-input integral (q6_methods.final_integral): given the exact
state after nine inputs, P(correct) over the tenth input x ~ N(0,1) by sweeping the at most 16 ReLU gate roots plus
the two values where the target changes (x > S9: the new input is the 2nd max; x > M9: the old max is).
Check: a dense midpoint rule in u = Phi(x) (2^20 points, error at most (#breakpoints)/2^20) on the same prefixes,
for the mystery model, seeds 0-4 and 10 population networks; prefixes = 64 random N(0,1) nine-input draws and the 64
first q = 1 MF3 prefixes (prefix_rule).  Then the round's table (last_q{1,2,3}.npz from `q6_methods.py last`):
raw and dev-calibrated (affine fit on the 64 dev networks, 0::16) RMS on the held-out 64 (8::16), and costs.
Usage: python scripts/z53.py <dir with last_q1.npz, last_q2.npz, last_q3.npz>
"""
import sys, numpy as np, torch as th
sys.path.insert(0,'chatgpt2/round6'); import q6_methods as q6
from z2 import load
from seeds_clocks import load as zload
D=sys.argv[1]; dev='cuda' if th.cuda.is_available() else 'cpu'
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
nets=[(n,[np.asarray(z[i],float).reshape(-1,16) if i==0 else np.asarray(z[i],float) for i in range(3)]) for n,z in zip(['mystery','seed0','seed1','seed2','seed3','seed4'],Z)]
nets[:]=[(n,[w[0][0],w[1],w[2]]) for n,w in nets]+[(f'pop{i}',[P[0][i].numpy().astype(float),P[1][i].numpy().astype(float),P[2][i].numpy().astype(float)]) for i in range(0,1024,103)]
rng=np.random.default_rng(53); pre=np.r_[rng.standard_normal((64,9)),q6.prefix_rule(1)[0][:64]]
u=(np.arange(2**20)+.5)/2**20; xs=th.tensor(q6.ndtri(u),device=dev,dtype=th.float64)
worst=0; out=[]
for name,(wi,A,O) in nets:
    h=np.zeros((len(pre),16))
    for t in range(9): h=np.maximum(h@A.T+pre[:,t,None]*wi,0)
    rk=pre.argsort(-1); ps,pm=rk[:,-2],rk[:,-1]; S9,M9=pre[np.arange(len(pre)),ps],pre[np.arange(len(pre)),pm]
    exact,seg=q6.final_integral(h,wi,A,O,S9,M9,ps,pm)
    Wt=[th.tensor(w,device=dev,dtype=th.float64) for w in (wi,A,O)]; grid=np.empty(len(pre))
    for j in range(len(pre)):
        hh=th.relu((th.tensor(h[j],device=dev,dtype=th.float64)@Wt[1].T)[None]+xs[:,None]*Wt[0][None]); lg=hh@Wt[2].T
        tgt=th.where(xs>float(M9[j]),int(pm[j]),th.where(xs>float(S9[j]),9,int(ps[j])))
        grid[j]=float((lg.argmax(-1)==tgt).double().mean())
    err=np.abs(exact-grid); bound=(seg*2+18)/2**20; worst=max(worst,float((err/bound).max()))
    out.append(f"{name}: max |exact - grid| {err.max():.2e} (bound {bound.max():.1e}), mean P {exact.mean():.4f}, segments {seg.mean():.2f}")
print("exact last-input integral vs 2^20-point midpoint rule, 128 prefixes per network:"); print("  "+"\n  ".join(out))
print(f"  largest error / its discretization bound: {worst:.2f}  (<= 1 means agreement within the grid's resolution)")
acc=np.load('pop1k/reference.npz')['acc']; dv,hd,zo=np.arange(0,1024,16),np.arange(8,1024,16),np.arange(1024,1030); pq=acc*(1-acc)
print("\nround-6 table, rescored from last_q*.npz (points; MSE/sampling at each network's cost):")
for q in (1,2,3):
    L=np.load(f'{D}/last_q{q}.npz'); p,c=L['pred'],L['cost']+2/2880; e=100*(p-acc)
    X=np.c_[np.ones(len(p)),p]; cal=X@np.linalg.lstsq(X[dv],acc[dv],rcond=None)[0]; ec=100*(cal-acc)
    print(f"  q{q}: prefixes {len(q6.prefix_rule(q)[0])}, held-out cost {c[hd].mean():.2f} FE; raw held RMS {np.sqrt((e[hd]**2).mean()):.4f} mean {e[hd].mean():+.4f}; "
          f"dev-calibrated held RMS {np.sqrt((ec[hd]**2).mean()):.4f} mean {ec[hd].mean():+.4f} MSE/s {np.mean(ec[hd]**2)/(1e4*np.mean(pq[hd]/c[hd])):.3f}; zoo "+" ".join(f"{v:+.3f}" for v in ec[zo]))
