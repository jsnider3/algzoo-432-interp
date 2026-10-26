# Analytic (sampling-free) accuracy estimate: mean-field boundary + weight-derived blur.
# For each arrangement of the top-3 positions and each (M,S,R): lower 7 inputs at their mean mu(R) given R; the
# logit margins m_j and their gradients g_j w.r.t. the lower inputs come from one tangent-forward pass. The lower
# inputs are iid truncated normals (sd sig(R)), so margin_j ~ N(m_j, (sig*|g_j|)^2) and
#   P(correct | M,S,R,pos) ~= prod_j Phi(m_j / (sig*|g_j|)).
# Integrand is smooth -> Gauss-Legendre in (M, a=(M-S)/|M|, b=(S-R)/|M|), concentrated near ties.
import numpy as np, itertools, sys, time
from scipy.special import ndtr
from scipy.stats import norm
from seeds_clocks import load
from j5 import gl
perms = np.array(list(itertools.permutations(range(10), 3)))
def tangent_logits(X, Wih, Whh, Wo):
    B = len(X); H = Whh.shape[0]; wi = Wih.reshape(-1)
    h = np.zeros((B, H)); dh = np.zeros((B, H, 10))
    for t in range(10):
        pre = X[:, t:t+1]*wi + h@Whh.T
        dpre = np.einsum('bjk,ij->bik', dh, Whh); dpre[:, :, t] += wi
        act = pre > 0; h = pre*act; dh = dpre*act[:, :, None]
    return h@Wo.T, np.einsum('oi,bik->bok', Wo, dh)
def cond_correct(Mv, Sv, Rv, W, mode="prod"):
    """Mv,Sv,Rv: (n,) values. returns (n,720) P(correct) for every arrangement, and the mean-field indicator."""
    n = len(Mv); lam = norm.pdf(Rv)/np.maximum(ndtr(Rv), 1e-300)
    mu = -lam; sig = np.sqrt(np.maximum(1-Rv*lam-lam**2, 0))
    P = np.tile(perms, (n, 1)); k = np.repeat(np.arange(n), 720); r = np.arange(n*720)
    X = np.repeat(mu[k][:, None], 10, 1); X[r, P[:, 0]] = Mv[k]; X[r, P[:, 1]] = Sv[k]; X[r, P[:, 2]] = Rv[k]
    low = np.ones((n*720, 10), bool); low[r[:, None], P] = False
    L, D = tangent_logits(X, *W)
    t = P[:, 1]; m = L[r, t][:, None]-L                               # margins (B,10), m_t = 0
    G = (D[r, t][:, None, :]-D)*low[:, None, :]                       # gradients wrt lower inputs
    s = sig[k][:, None]*np.sqrt((G**2).sum(2))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(s > 0, m/s, np.where(m > 0, np.inf, -np.inf))
    z[r, t] = np.inf
    pc = ndtr(z).prod(1) if mode == "prod" else ndtr(z.min(1))
    ind = (m > 0).sum(1) == 9
    return pc.reshape(n, 720), ind.reshape(n, 720)
def estimate(W, nM, na, nb, Mlo=-2.5, Mhi=4.6, Rmin=-5.0, chunk=24):
    mA, wA = gl(Mlo, 0, max(3, nM//3)); mB, wB = gl(0, Mhi, nM); Ms = np.r_[mA, mB]; wM = np.r_[wA, wB]
    u, wu = gl(0, 1, na); v, wv = gl(0, 1, nb)
    pts = []
    for Mi, wi in zip(Ms, wM):
        A = abs(Mi); amax = (Mi-Rmin)/A
        for ui, wui in zip(u, wu):
            a = amax*ui**2; Si = Mi-a*A; bmax = max((Si-Rmin)/A, 1e-9)
            for vi, wvi in zip(v, wv):
                b = bmax*vi**2; Ri = Si-b*A
                jac = A*A*(2*amax*ui)*(2*bmax*vi)                    # dS dR = |M|^2 da db ; a=amax u^2, b=bmax v^2
                d = norm.pdf(Mi)*norm.pdf(Si)*norm.pdf(Ri)*ndtr(Ri)**7
                pts.append((Mi, Si, Ri, wi*wui*wvi*jac*d))
    pts = np.array(pts); tot = 0.0; totI = 0.0; mass = 0.0
    for c in range(0, len(pts), chunk):
        q = pts[c:c+chunk]; pc, ind = cond_correct(q[:, 0], q[:, 1], q[:, 2], W)
        tot += (q[:, 3]*pc.sum(1)).sum(); totI += (q[:, 3]*ind.sum(1)).sum(); mass += 720*q[:, 3].sum()
    return tot, totI, mass, len(pts)*720
if __name__ == "__main__":
    f = sys.argv[1]; truth = float(sys.argv[2]); W = load(f)
    for nM, na, nb in [(6, 6, 6), (8, 8, 8), (10, 12, 12), (12, 16, 16), (16, 20, 20)]:
        t0 = time.time(); tot, totI, mass, npts = estimate(W, nM, na, nb)
        print(f"{f} grid {nM}x{na}x{nb}: mass {mass:.6f} | blur estimate {tot/mass:.5f} (err {100*(tot/mass-truth):+.3f}pp) | "
              f"step (mean-field only) {totI/mass:.5f} (err {100*(totI/mass-truth):+.3f}pp) | points {npts/1e6:.2f}M [{time.time()-t0:.0f}s]", flush=True)
