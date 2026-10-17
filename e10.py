from common import *
x=sample(3000000,seed=15); lg,hs=run(x); pm=lg.argmax(1)
o=np.argsort(-x,1); srt=-np.sort(-x,1); S=srt[:,1]; neg=S<0
rk=np.argsort(o,1)[np.arange(len(x)),pm]
print("S<0 cases:", neg.sum(), " model pick rank distribution (0=max):", (np.bincount(rk[neg],minlength=10)/neg.sum()).round(3))
# is the pick then related to the age of the max, e.g. slot just after? distribution of (pred - p1)
d=pm[neg]-o[neg,0]; print(" pred - pos(max):", {k:round(float((d==k).mean()),3) for k in range(-9,10) if (d==k).mean()>0.02})
# n2 (running max register) vs M, S<0 vs S>0
M=srt[:,0]; print(" neuron 2 / M at end:  S<0:", np.median(hs[neg,-1,2]/M[neg]).round(2), "  S>0:", np.median(hs[~neg,-1,2]/M[~neg]).round(2))
