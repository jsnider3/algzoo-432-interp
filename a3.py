from common import *; from feats import features
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(50000); lg, hs = run(x)
for n in range(H):
    print(f"--- neuron {n}")
    for t in [3, 6, 9]:
        F = features(x, t); names=list(F); Fm = np.stack([F[k] for k in names],1)
        y = hs[:, t, n]
        # best single feature (with intercept)
        best = []
        for i,k in enumerate(names):
            c = np.corrcoef(Fm[:,i], y)[0,1]; best.append((c**2 if np.isfinite(c) else 0, k))
        best.sort(reverse=True)
        # full linear fit R2
        A = np.c_[Fm, np.ones(len(y))]; coef,*_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((A@coef-y)**2).mean()/y.var()
        print(f" t={t} full-lin R2={r2:.3f}  top1: " + ", ".join(f"{k}:{r:.2f}" for r,k in best[:4]))
