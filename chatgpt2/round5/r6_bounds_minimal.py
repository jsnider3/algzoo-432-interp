"""Restricted cone-oracle bound and analytic ANOVA counterexample; no data."""
from math import atan, cos, pi, sin, sqrt


def G(r):
    a = atan(r)
    return (35*a - 28*sin(2*a) + 7*sin(4*a)
            - 4*sin(6*a)/3 + sin(8*a)/8)/(35*pi)


V = 0.069
Q = (1-sqrt(1-4*V))/2
lo, hi = 0., 10.
for _ in range(80):
    mid = (lo+hi)/2
    if G(mid) < Q:
        lo = mid
    else:
        hi = mid
R = (lo+hi)/2
print(f"Mass cap={Q:.12f}; R={R:.12f}")
print("Queries; worst-error lower(pp); sampling RMS(pp); squared ratio")
for N in (100, 1000, 10000, 100000, 1000000, 10000000):
    theta = (256/(175*N))**(1/8)
    lower = 2*(G(R)-G(R*cos(theta)))/5
    sampling = sqrt(V/N)
    print(N, 100*lower, 100*sampling, (lower/sampling)**2)

# iid Gaussian X,Y,Z: m=|X+Y|-|X-Y|+2|Z|, first ANOVA m_A=2|Z|.
# Failure of m means XY<0 and |Z| is smallest: probability (1/2)*(1/3).
# Additive g=X+sqrt(3)|Z| fails in a wedge of angle pi/3.
true_p = 1-1/6
anova_p = 1.
additive_p = 0.5+atan(sqrt(3))/pi
assert abs(true_p-additive_p) < 1e-15
print("ANOVA example: true, ANOVA, other additive:",
      true_p, anova_p, additive_p)
