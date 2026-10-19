"""Numerical check of ChatGPT round 5's restricted lower bound (G(r), q*, R, the L_N table, the N^(-1/4) constant)
and of its ANOVA counterexample, m = |X+Y| - |X-Y| + 2|Z| with P(m > 0) = 5/6.  Usage: python check_lower_bound.py
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc
def G(r):
    a=np.arctan(r); return (35*a-28*np.sin(2*a)+7*np.sin(4*a)-4/3*np.sin(6*a)+np.sin(8*a)/8)/(35*np.pi)
# G(r) = P(X10 >= |X_{1:9}|/r): Monte Carlo and the incomplete-beta form of the cap area on S^9
rng=np.random.default_rng(0); X=rng.standard_normal((4_000_000,10)); nr=np.linalg.norm(X[:,:9],axis=1)
for r in (0.5,1.0,1.9):
    a=np.arctan(r); cap=0.5*betainc(4.5,0.5,np.sin(a)**2)   # normalized cap of angle a < pi/2 on S^9
    print(f"G({r}) formula {G(r):.6f}  cap {cap:.6f}  MC {np.mean(X[:,9]>=nr/r):.6f}")
qs=(1-np.sqrt(1-4*0.069))/2; R=brentq(lambda r:G(r)-qs,0.1,10); print(f"q* {qs:.12f}  R {R:.12f}")
# cap bound on S^8: (35/32) int_0^t sin^7 <= 35 t^8/256
t=0.3; print("S^8 cap check", 35/32*np.trapezoid(np.sin(np.linspace(0,t,10001))**7,np.linspace(0,t,10001)), 35*t**8/256, 0.5*betainc(4,0.5,np.sin(t)**2))
for N in (1e2,1e3,1e4,1e5,1e6,1e7):
    th_=(256/(175*N))**0.125; L=0.4*(G(R)-G(R*np.cos(th_))); s=np.sqrt(0.069/N)
    print(f"N {N:9.0f}  L_N {100*L:.6f}  sampling {100*s:.6f}  L^2/MSE {L*L/s/s:.3f}  L*N^0.25 {L*N**0.25:.10f}")
h=1e-6; dG=(G(R+h)-G(R-h))/2/h; print("asymptotic constant (1/5) R G'(R) (256/175)^(1/4) =", 0.2*R*dG*(256/175)**0.25)
# ANOVA counterexample
x,y,z=rng.standard_normal((3,4_000_000)); m=np.abs(x+y)-np.abs(x-y)+2*np.abs(z)
print("P(m>0)", np.mean(m>0), "5/6 =", 5/6, " P(x+sqrt3|z|>0)", np.mean(x+np.sqrt(3)*np.abs(z)>0))
xs=np.linspace(-2,2,5); print("E[m|X=x] (should be const 2E|Z| =", 2*np.sqrt(2/np.pi),")", [np.mean(np.abs(v+y[:400000])-np.abs(v-y[:400000]))+2*np.sqrt(2/np.pi) for v in xs])
