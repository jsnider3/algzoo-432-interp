from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(300000); lg, hs = run(x)
t=9; X=x; o=np.argsort(X,1); srt=np.sort(X,1)
M=srt[:,-1]; S=srt[:,-2]; aM=t-o[:,-1]; aS=t-o[:,-2]; xt=X[:,-1]
h=hs[:,t]
# model 1: per (aM,aS) cell, linear in M,S,x_t + const
pred=np.zeros_like(h)
for a in range(T):
  for b in range(T):
    s=(aM==a)&(aS==b)
    if s.sum()<50: continue
    A=np.c_[M[s],S[s],xt[s],np.ones(s.sum())]; c,*_=np.linalg.lstsq(A,h[s],rcond=None); pred[s]=A@c
print("R2 per neuron, h ~ cell(aM,aS) x [M,S,x_t,1]:", 1-((pred-h)**2).mean(0)/h.var(0))
lp=pred@Wo.T; print("acc of readout on fitted state:", (lp.argmax(1)==o[:,-2]).mean(), " agreement w/ model:", (lp.argmax(1)==lg.argmax(1)).mean())
