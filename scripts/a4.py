from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(100000); lg, hs, ms, J = run_jac(x)
print("homogeneity check", np.abs(np.einsum('not,nt->no',J,x)-lg).max())
o=np.argsort(x,-1); p1,p2,p3=o[:,-1],o[:,-2],o[:,-3]
# number of distinct activation patterns
pat = ms.reshape(len(x),-1).astype(np.uint8)
u, cnt = np.unique(pat, axis=0, return_counts=True)
print("distinct patterns:", len(u), "top counts", np.sort(cnt)[::-1][:10])
# look at J in a specific (p1,p2) group, e.g. p1=3, p2=6
s=(p1==3)&(p2==6)
print("n",s.sum()); Jm = J[s].mean(0); Js = J[s].std(0)
print("mean J (rows=logit, cols=input)\n", Jm); print("std J\n", Js)
