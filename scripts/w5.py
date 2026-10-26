# Matched numba kernels for a fair cost ratio: plain forward+check vs exact last-input integration.
# The sweep is incremental: crossing a hinge toggles one unit, a rank-1 update of the logit slopes/intercepts.
import numpy as np, math
from numba import njit

@njit(cache=True, fastmath=False)
def correct(x, wi, wh, wo):
    n, T = x.shape; H = wi.shape[0]; out = np.empty(n)
    h = np.empty(H); g = np.empty(H)
    for k in range(n):
        h[:] = 0.0
        for t in range(T):
            for i in range(H):
                s = wi[i]*x[k, t]
                for j in range(H): s += wh[i, j]*h[j]
                g[i] = s if s > 0 else 0.0
            h[:] = g
        best = -1e300; arg = -1
        for o in range(wo.shape[0]):
            s = 0.0
            for j in range(H): s += wo[o, j]*h[j]
            if s > best: best = s; arg = o
        # target = index of 2nd largest
        i1 = 0
        for t in range(1, T):
            if x[k, t] > x[k, i1]: i1 = t
        i2 = 1 if i1 == 0 else 0
        for t in range(T):
            if t != i1 and x[k, t] > x[k, i2]: i2 = t
        out[k] = 1.0 if arg == i2 else 0.0
    return out

@njit(cache=True)
def Phi(z):
    return 0.5*math.erfc(-z/math.sqrt(2.0))

@njit(cache=True, fastmath=False)
def condprob(y, wi, wh, wo):
    n, T9 = y.shape; H = wi.shape[0]; C = wo.shape[0]; out = np.empty(n)
    h = np.empty(H); g = np.empty(H); b = np.empty(H)
    s = np.empty(C); c = np.empty(C)
    bp = np.empty(H+2); who = np.empty(H+2, np.int64)
    for k in range(n):
        h[:] = 0.0
        for t in range(T9):
            for i in range(H):
                v = wi[i]*y[k, t]
                for j in range(H): v += wh[i, j]*h[j]
                g[i] = v if v > 0 else 0.0
            h[:] = g
        for i in range(H):
            v = 0.0
            for j in range(H): v += wh[i, j]*h[j]
            b[i] = v
        i1 = 0
        for t in range(1, T9):
            if y[k, t] > y[k, i1]: i1 = t
        i2 = 1 if i1 == 0 else 0
        for t in range(T9):
            if t != i1 and y[k, t] > y[k, i2]: i2 = t
        m1 = y[k, i1]; m2 = y[k, i2]
        # breakpoints: hinges (who=i) and task thresholds (who=-1)
        nb = 0
        for i in range(H):
            if wi[i] != 0.0:
                bp[nb] = -b[i]/wi[i]; who[nb] = i; nb += 1
        bp[nb] = m2; who[nb] = -1; nb += 1
        bp[nb] = m1; who[nb] = -1; nb += 1
        for p in range(1, nb):           # insertion sort (nb <= 18)
            v = bp[p]; w = who[p]; q = p-1
            while q >= 0 and bp[q] > v:
                bp[q+1] = bp[q]; who[q+1] = who[q]; q -= 1
            bp[q+1] = v; who[q+1] = w
        # state at z -> -inf: unit active iff a<0 (or a==0 and b>0)
        s[:] = 0.0; c[:] = 0.0
        for i in range(H):
            if wi[i] < 0 or (wi[i] == 0 and b[i] > 0):
                for o in range(C):
                    s[o] += wo[o, i]*wi[i]; c[o] += wo[o, i]*b[i]
        tot = 0.0; lo = -np.inf
        for p in range(nb+1):
            hi = bp[p] if p < nb else np.inf
            if hi > lo:
                mid = hi-1.0 if lo == -np.inf else (lo+1.0 if hi == np.inf else 0.5*(lo+hi))
                tg = i2 if mid < m2 else (T9 if mid < m1 else i1)
                L = lo; U = hi; ok = True
                for o in range(C):
                    if o == tg: continue
                    ds = s[tg]-s[o]; dc = c[tg]-c[o]
                    if ds > 0:
                        r = -dc/ds
                        if r > L: L = r
                    elif ds < 0:
                        r = -dc/ds
                        if r < U: U = r
                    elif dc <= 0:
                        ok = False; break
                if ok and U > L: tot += Phi(U)-Phi(L)
            if p < nb:
                i = who[p]
                if i >= 0:                   # crossing hinge of unit i: toggle it
                    sg = 1.0 if wi[i] > 0 else -1.0   # a>0: turns on going right; a<0: turns off
                    for o in range(C):
                        s[o] += sg*wo[o, i]*wi[i]; c[o] += sg*wo[o, i]*b[i]
                lo = hi
        out[k] = tot
    return out

if __name__ == "__main__":
    import time, sys
    from w1 import load, cond_prob, pl_vec
    rng = np.random.default_rng(3)
    for f in ["weights.npz", "weights_seed0.npz", "weights_seed2.npz"]:
        Wih, Whh, Wo = load(f); wi = Wih.reshape(-1).astype(float); W = (Wih, Whh, Wo)
        y = rng.standard_normal((20000, 9)); x = rng.standard_normal((20000, 10))
        assert np.allclose(condprob(y, wi, Whh, Wo), cond_prob(y, W), atol=1e-9)
        assert np.array_equal(correct(x, wi, Whh, Wo), pl_vec(x, W))
        x = rng.standard_normal((1 << 18, 10)); r = []
        for _ in range(9):
            t = time.perf_counter(); correct(x, wi, Whh, Wo); tp = time.perf_counter()-t
            t = time.perf_counter(); condprob(x[:, :9], wi, Whh, Wo); tc = time.perf_counter()-t
            r.append(tc/tp)
        print(f"{f}: forward {tp/len(x)*1e9:.0f} ns/sample, cond {tc/len(x)*1e9:.0f} ns/sample, median ratio {np.median(r):.2f} (range {min(r):.2f}-{max(r):.2f})", flush=True)

@njit(cache=True)
def _sweep(h, z9, m1, m2, i1, i2, T9, wi, wh, wo, b, s, c, bp, who, g):
    """given state h after 8 inputs and 9th input z9: P(correct | x1..x9) by the exact x10 sweep."""
    H = wi.shape[0]; C = wo.shape[0]
    for i in range(H):
        v = wi[i]*z9
        for j in range(H): v += wh[i, j]*h[j]
        g[i] = v if v > 0 else 0.0
    for i in range(H):
        v = 0.0
        for j in range(H): v += wh[i, j]*g[j]
        b[i] = v
    # update top-2 of the 9-prefix with z9 (position T9-1)
    if z9 > m1: m2 = m1; i2 = i1; m1 = z9; i1 = T9-1
    elif z9 > m2: m2 = z9; i2 = T9-1
    nb = 0
    for i in range(H):
        if wi[i] != 0.0:
            bp[nb] = -b[i]/wi[i]; who[nb] = i; nb += 1
    bp[nb] = m2; who[nb] = -1; nb += 1
    bp[nb] = m1; who[nb] = -1; nb += 1
    for p in range(1, nb):
        v = bp[p]; w = who[p]; q = p-1
        while q >= 0 and bp[q] > v:
            bp[q+1] = bp[q]; who[q+1] = who[q]; q -= 1
        bp[q+1] = v; who[q+1] = w
    s[:] = 0.0; c[:] = 0.0
    for i in range(H):
        if wi[i] < 0 or (wi[i] == 0 and b[i] > 0):
            for o in range(C):
                s[o] += wo[o, i]*wi[i]; c[o] += wo[o, i]*b[i]
    tot = 0.0; lo = -np.inf
    for p in range(nb+1):
        hi = bp[p] if p < nb else np.inf
        if hi > lo:
            mid = hi-1.0 if lo == -np.inf else (lo+1.0 if hi == np.inf else 0.5*(lo+hi))
            tg = i2 if mid < m2 else (T9 if mid < m1 else i1)
            L = lo; U = hi; ok = True
            for o in range(C):
                if o == tg: continue
                ds = s[tg]-s[o]; dc = c[tg]-c[o]
                if ds > 0:
                    r = -dc/ds
                    if r > L: L = r
                elif ds < 0:
                    r = -dc/ds
                    if r < U: U = r
                elif dc <= 0:
                    ok = False; break
            if ok and U > L: tot += Phi(U)-Phi(L)
        if p < nb:
            i = who[p]
            if i >= 0:
                sg = 1.0 if wi[i] > 0 else -1.0
                for o in range(C):
                    s[o] += sg*wo[o, i]*wi[i]; c[o] += sg*wo[o, i]*b[i]
            lo = hi
    return tot

@njit(cache=True)
def ndtri_(p):
    # inverse normal CDF (Acklam), refined with one Newton step
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2*math.log(p)); x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > 1-0.02425:
        q = math.sqrt(-2*math.log(1-p)); x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p-0.5; r = q*q; x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = Phi(x)-p; x = x - e*math.sqrt(2*math.pi)*math.exp(x*x/2)
    return x

@njit(cache=True)
def condprob2(P, U, wi, wh, wo):
    """P: (n,8) prefixes; U: (n,q) uniforms. x9 by q-point stratified sampling (unbiased), x10 exact."""
    n, T8 = P.shape; q = U.shape[1]; H = wi.shape[0]; C = wo.shape[0]; out = np.empty(n)
    h = np.empty(H); g = np.empty(H); b = np.empty(H); s = np.empty(C); c = np.empty(C)
    bp = np.empty(H+2); who = np.empty(H+2, np.int64)
    for k in range(n):
        h[:] = 0.0
        for t in range(T8):
            for i in range(H):
                v = wi[i]*P[k, t]
                for j in range(H): v += wh[i, j]*h[j]
                g[i] = v if v > 0 else 0.0
            h[:] = g
        i1 = 0
        for t in range(1, T8):
            if P[k, t] > P[k, i1]: i1 = t
        i2 = 1 if i1 == 0 else 0
        for t in range(T8):
            if t != i1 and P[k, t] > P[k, i2]: i2 = t
        tot = 0.0
        for j in range(q):
            z9 = ndtri_((j+U[k, j])/q)
            tot += _sweep(h, z9, P[k, i1], P[k, i2], i1, i2, T8+1, wi, wh, wo, b, s, c, bp, who, g)
        out[k] = tot/q
    return out
