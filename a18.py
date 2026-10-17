from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(500000); lg, hs = run(x)
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]
h=hs[:,9]
# decompose h_n ≈ M*u_n(aM,aS) + S*v_n(aM,aS): per-cell fit, then show variance explained by ageS-only / ageM-only cell models
def r2cells(keys):
    pred=np.zeros_like(h)
    for k in np.unique(keys):
        s=keys==k; A=np.c_[M[s],S[s],x[s,9]]; c,*_=np.linalg.lstsq(A,h[s],rcond=None); pred[s]=A@c
    return 1-((pred-h)**2).mean(0)/h.var(0)
print("neuron:         ", np.arange(16))
print("R2 cells none   ", r2cells(np.zeros(len(x),int)))
print("R2 cells aM     ", r2cells(aM))
print("R2 cells aS     ", r2cells(aS))
print("R2 cells aM,aS  ", r2cells(aM*10+aS))
