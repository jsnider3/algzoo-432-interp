"""Reproduce the original restricted-oracle bound in r6_bounds.md.

No data access, fitting, random numbers, or external packages are required.
These are minimax bounds for a larger geometric class, not measurements on
the 432-parameter RNN population.
"""
from math import atan, cos, pi, sin, sqrt


def cone_mass(radius):
    """P[X_10 >= ||X_1:9|| / radius], for X ~ N(0, I_10)."""
    angle = atan(radius)
    return (35 * angle - 28 * sin(2 * angle) + 7 * sin(4 * angle)
            - (4 / 3) * sin(6 * angle) + sin(8 * angle) / 8) / (35 * pi)


def bound_constants(variance=0.069):
    cap = (1 - sqrt(1 - 4 * variance)) / 2
    lower, upper = 0.0, 10.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if cone_mass(middle) < cap:
            lower = middle
        else:
            upper = middle
    return cap, (lower + upper) / 2


def lower_bound(queries, radius):
    angle = (256 / (175 * queries)) ** (1 / 8)
    return (2 / 5) * (cone_mass(radius) - cone_mass(radius * cos(angle)))


def main():
    variance = 0.069
    cap, radius = bound_constants(variance)
    print(f"Error-mass cap: {cap:.12f}; cone radius: {radius:.12f}")
    print("queries  lower_pp  sampling_pp  squared_ratio")
    for queries in [100, 1000, 10000, 100000, 1000000, 10000000]:
        lower = lower_bound(queries, radius)
        sampling = sqrt(variance / queries)
        print(f"{queries:>8} {100*lower:>10.6f} {100*sampling:>12.6f} "
              f"{(lower/sampling)**2:>14.6f}")
    print("\nIllustrative certificate radii from z26's estimated median tails:")
    print("patterns  forwards  estimated_tail_pp  midpoint_radius_pp  sampling_pp")
    for patterns, tail in [(1, .80103), (3, .60671), (10, .33667),
                           (30, .15146), (100, .04989), (300, .01299),
                           (1000, .00291)]:
        cost = 5040 * patterns
        print(f"{patterns:>8} {cost:>9} {100*tail:>18.3f} "
              f"{50*tail:>19.4f} {100*sqrt(variance/cost):>12.6f}")
    print("\nTruncation crossover from 0.362 points:", variance / .00362**2)


if __name__ == "__main__":
    main()
