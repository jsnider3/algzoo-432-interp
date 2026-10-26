import numpy as np, torch as th
from m1 import A, B
from seeds_clocks import T
th.manual_seed(0); x=th.randn(200000,10); y=x.argsort(-1)[:,-2]
def grads(Wih,Whh,Wo):
    P=[th.tensor(a,dtype=th.float64,requires_grad=True) for a in (Wih,Whh,Wo)]
    h=th.zeros(len(x),16,dtype=th.float64); X=x.double()
    for t in range(T): h=th.relu(X[:,t:t+1]*P[0]+h@P[1].T)
    l=th.nn.functional.cross_entropy(h@P[2].T,y); l.backward(); return [p.grad.numpy() for p in P]
gF=grads(B[0],A[1],A[2]); gW=grads(B[0],A[1],B[2])
cos=lambda a,b:(a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b)
print("initial gradients, full (s3 Whh+Wo) vs Whh-only (s4 Wo); same forward dynamics")
print(f"  n13 input weight grad: full {gF[0][13]:+.4f}, Whh-only {gW[0][13]:+.4f}")
print(f"  n13 incoming row grad: |full| {np.linalg.norm(gF[1][13]):.4f}, |Whh-only| {np.linalg.norm(gW[1][13]):.4f}, cosine {cos(gF[1][13],gW[1][13]):+.2f}")
print(f"  rank of n13 by |grad on incoming row| among 16: full {int((np.linalg.norm(gF[1],axis=1)>np.linalg.norm(gF[1][13])).sum())}, Whh-only {int((np.linalg.norm(gW[1],axis=1)>np.linalg.norm(gW[1][13])).sum())}")
print(f"  rank of n13 by |grad on input weight|: full {int((np.abs(gF[0])>abs(gF[0][13])).sum())}, Whh-only {int((np.abs(gW[0])>abs(gW[0][13])).sum())}")
print("  sign of -grad on n13's incoming weights from each neuron (full):", np.sign(-gF[1][13]).astype(int))
print("  sign of -grad on n13's incoming weights from each neuron (Whh-only):", np.sign(-gW[1][13]).astype(int))
