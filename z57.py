"""Oracle test for Hermite / low-degree polynomial propagation (ARC, "Competing with sampling": represent every unit
as a degree-<=d polynomial in the Gaussian inputs, propagate layer by layer; d is the largest integer with
n^d <= 1/eps^2, so d = 2 at about 600 forward-equivalents and d = 3 at about 4,000 for n = 10 inputs).
Here the projection is done as well as it can be: at each step the hidden state is replaced by its exact
least-squares projection onto Hermite polynomials of degree <= d (fitted on 2^18 N(0,1) inputs, 2^17 for d = 4),
then fed to the next step ("recursive"); the final argmax and the 2nd-argmax target are then evaluated exactly on
2^19 fresh inputs.  Any real propagation scheme also has to approximate the ReLU's projection and the final
indicator, so its error is not expected to be smaller than this oracle's.  Also "direct": the true final logits
projected onto degree <= d, and the variance share (R^2) the projection captures at the last step.
Usage: python z57.py
"""
import itertools, math, numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
dev='cuda' if th.cuda.is_available() else 'cpu'
HE=[lambda x: th.ones_like(x), lambda x: x, lambda x: x*x-1, lambda x: x**3-3*x, lambda x: x**4-6*x*x+3]
def basis(x,d):
    idx=[a for k in range(d+1) for a in itertools.combinations_with_replacement(range(10),k)]
    H=[[f(x[:,i]) for f in HE[:d+1]] for i in range(10)]; cols=[]
    for a in idx:
        c=th.ones(len(x),device=x.device,dtype=x.dtype)
        for i in set(a): c=c*H[i][a.count(i)]
        cols.append(c)
    return th.stack(cols,1)
def run(W,x,proj=None):
    wi,A,O=W; h=th.zeros(len(x),16,device=x.device,dtype=x.dtype)
    for t in range(10):
        h=th.relu(h@A.T+x[:,t,None]*wi)
        if proj is not None: h=proj(h,t)
    return h@O.T
def target(x): return x.scatter(-1,x.argmax(-1,keepdim=True),float('-inf')).argmax(-1)
g=th.Generator(device=dev).manual_seed(57); XF=th.randn(2**18,10,generator=g,device=dev,dtype=th.float64); XT=th.randn(2**19,10,generator=g,device=dev,dtype=th.float64)
Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]; nets={n:z for n,z in zip(['mystery','seed0','seed1','seed2','seed3','seed4'],Z)}
for tag,f in (('zoo-recipe pop',"pop1k/seed0.npz"),('mystery-recipe pop',"pop_unif/b128_n1024_seed4.npz")):
    P,_=load(f)
    for i in (0,341,682): nets[f"{tag} {i}"]=[p[i].double().numpy() for p in P]
yT=target(XT); I16=th.eye(16,device=dev,dtype=th.float64)
for d in (1,2,3,4):
    xf=XF[:2**17] if d==4 else XF; B=basis(xf,d); Q,R=th.linalg.qr(B); del B
    print(f"\ndegree {d}: {R.shape[0]} Hermite polynomials, fitted on {len(xf):,} inputs; scored on {len(XT):,} fresh inputs",flush=True)
    print(f"{'network':24s} {'true acc':>8s}  {'recursive':>9s}  {'direct':>7s}  {'R^2 h_10':>8s}  {'R^2 logit gaps':>14s}")
    for n,W in nets.items():
        W=[th.tensor(np.asarray(w),device=dev,dtype=th.float64).reshape(s) for w,s in zip(W,[(16,),(16,16),(10,16)])]
        C=[]
        def fitproj(h,t): c=th.linalg.solve_triangular(R,Q.T@h,upper=True); C.append(c); return Q@(Q.T@h)
        run(W,xf,fitproj)                                  # recursive projection on the fit inputs: coefficients per step
        cL=th.linalg.solve_triangular(R,Q.T@run(W,xf),upper=True); cH=th.linalg.solve_triangular(R,Q.T@run([W[0],W[1],I16],xf),upper=True)
        ok=th.zeros(3,device=dev,dtype=th.float64); sse=th.zeros(2,device=dev,dtype=th.float64); sst=th.zeros(2,device=dev,dtype=th.float64)
        s1=th.zeros(2,26,device=dev,dtype=th.float64)
        for xc,yc in zip(XT.split(2**16),yT.split(2**16)):
            Bc=basis(xc,d); it=iter(C)
            LT=run(W,xc); LR=run(W,xc,lambda h,t: Bc@next(it)); LD=Bc@cL; hT=run([W[0],W[1],I16],xc); hD=Bc@cH
            ok+=th.stack([(L.argmax(-1)==yc).double().sum() for L in (LT,LR,LD)])
            gT=LT-LT.mean(-1,keepdim=True); gD=LD-LD.mean(-1,keepdim=True)
            sse+=th.stack([((hT-hD)**2).sum(),((gT-gD)**2).sum()]); sst+=th.stack([(hT**2).sum(),(gT**2).sum()])
            s1[0,:16]+=hT.sum(0); s1[1,:10]+=gT.sum(0)
        n_=len(XT); var=sst-th.stack([(s1[0,:16]**2).sum()/n_,(s1[1,:10]**2).sum()/n_]); r2=1-sse/var; a=(ok/n_).tolist()
        print(f"{n:24s} {100*a[0]:7.2f}%  {100*a[1]:8.2f}%  {100*a[2]:6.2f}%  {r2[0].item():8.4f}  {r2[1].item():14.4f}",flush=True)
