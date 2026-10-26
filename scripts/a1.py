from common import *
x = sample()
lg, hs = run(x); print("acc", acc(lg, x))
# activity stats at each time step
print("frac active per neuron per t:")
np.set_printoptions(precision=2, suppress=True, linewidth=220)
print((hs>0).mean(0).T)
print("mean act at final t:", hs[:,-1].mean(0))
# ablations at all timesteps: zero and mean
means = hs.mean(0)  # (T,H)
for n in range(H):
    def hz(t,h,pre,n=n): h=h.copy(); h[:,n]=0; return h
    def hm(t,h,pre,n=n): h=h.copy(); h[:,n]=means[t,n]; return h
    print(n, "zero-abl acc %.3f  mean-abl acc %.3f" % (acc(run(x,hook=hz)[0],x), acc(run(x,hook=hm)[0],x)))
