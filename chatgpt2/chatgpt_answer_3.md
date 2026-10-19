# ChatGPT Pro, round 4 (QUESTION4: a deterministic top-4 integral on the sampling line)

Summary of the answer (chat "Estimator Comparison", 2026-10-19; 48 minutes of work). Its numbers are ChatGPT's own unless marked **checked**. It attached `q4_results.json` and `q4_reproduction_code.zip` (7 scripts: `q4_exact_reproduce.py`, `q4_spline.py`, `q4_reference.py`, `q4_score.py`, `q4_jump_reproduce.py`, `q4_certificate.py`, `q4_homogeneous_reproduce.py`; CPU, numpy/scipy/numba). They are not yet downloaded into the repo.

## Verdict

No deterministic estimator reached the sampling line anywhere in 10^3–10^5 forwards. Two positive results: a much cheaper exact S-line integrator (still 16x sampling's MSE), and an exact homogeneity reduction. It also found that the clamped first-order leakage model reverses the anchor at its boundary.

## 1. Accuracy versus compute (development on every 16th network, fixed rules, then all 1,024)

Boundary interpolation along S (logits on a grid in S, exact roots of the piecewise-linear interpolant, all 720 MF3 arrangements with shared prefixes):

| outer order | S segments | forwards | RMS (all 1,024) | sampling RMS | MSE / sampling |
|---|---|---|---|---|---|
| 1 | 1 | 669 | 8.857 | 1.014 | 76.2 |
| 1 | 2 | 1,029 | 2.328 | 0.818 | 8.1 |
| 2 | 4 | 6,999 | 1.169 | 0.314 | 13.9 |
| 4 | 8 | 51,060 | 0.629 | 0.116 | 29.4 |
| 4 | 16 | 97,186 | 0.562 | 0.084 | 44.6 |

Exact S lines (trace every ReLU piece along S, integrate correct intervals analytically, Gauss–Jacobi on the order-statistic CDFs: Phi(R) ~ Beta(8,3), (Phi(M)-Phi(R))/(1-Phi(R)) ~ Beta(2,1); for MF4 also Phi(Q) ~ Beta(7,4) and (Phi(R)-Phi(Q))/(1-Phi(Q)) ~ Beta(1,3)). MF3, outer order 3 (9 S-slices): all 1,024 networks, mean cost 36,312 forwards (19k–67k), **RMS 0.580 points against 0.144 for sampling at matched per-network cost (16.1x MSE)**; bias +0.063; median |error| 0.335, p95 1.18, worst 2.55 (network 333). Held-out: MF3 orders 1–4 give 3.56 / 1.58 / 0.63 / 0.63 points at 2.8k / 11.9k / 36.7k / 56k forwards; MF4 orders 1–2 give 3.67 / 1.61 at 18.8k / 152k. The q=3 estimate differs from measured MF3 accuracy by 0.32 RMS, so quadrature error and MF3's own bias are both present and partly cancel. Scalar/root/CDF work is charged; with it set to zero the verdict is unchanged.

Prefix sharing: MF3 per slice 2,692 -> 1,871 (shared prefixes) -> 1,528 (sparse updates between adjacent pieces) forwards; MF4 17,840 -> 11,292 -> 9,265. One MF4 value quadruple's whole arrangement tree is 17,254 recurrent steps + 5,040 readouts = 1,910 forwards (2.6x fewer than 5,040 separate), so 10^5 forwards buys about 52 quadruples before any lower-input correction.

Fresh zoo references (2^26 inputs, seed 202610190401): mystery 0.95310552, seed0 0.90999609, seed1 0.88013929, seed2 0.94851206, seed3 0.93207060, seed4 0.88958584 (SE 2.6–4.0e-5). MF3 exact-S q=3 errors: −0.31, −0.04, +1.33, −0.48, −0.77, −0.14 points.

## 2. The clamped first-order model reverses the anchor

P_raw = sum_j E_j − 5 I0 and P = clip(P_raw, 0, 1). Where I0 steps 0 -> 1 with the E_j continuous, P_raw drops by 5; if 1 < sum E_j < 5, P jumps from 1 to 0. Toy: I = 1[m + sum z_j > 0], z_j ~ U(−1,1) gives P(0−) = 1, P(0+) = 0, yet the integral is exactly right by symmetry. Network 0 example (positions M,S,R,Q = 4,1,3,6; lower inputs at mu(Q) = −0.671; boundary S* = 0.888131): sum E = 3.168 on both sides, P 1 -> 0, true conditional accuracy 0.721. **Checked** (reproduced exactly; true 0.722 ± 0.001). All 40 anchor crossings it audited on network 0 reversed.

## 3. Certificates and the residual

Interval and 1-D affine ReLU relaxations certify 0.03% / 29–31% of lower-input lines, mostly late positions that were cheap anyway: about 1% MAC saving. Exact-line P has |P − I0| > 0 on 36–41% of configurations (> 0.01 on 12–15%), more than the 16-node 12%. Signed residual on its diagnostic: unclamped +0.076 ± 0.040, clamped −0.078 ± 0.031, clipping alone −0.154 ± 0.020, so clipping, not omitted pair terms, may carry the −0.05 residual. A product model of independent break/rescue events cut conditional squared error 13.5% but not integrated bias.

Bugs reported: `z11.py` ignored `trace_line`'s overflow flag (none seen in 3.3M traces), and `lines.trace_line` counted an exact tie on a piece as wrong even when argmax's lowest-index rule makes it right. **Checked and fixed** in both files; ties occur in under 1 in 2 million draws.

## 4. Homogeneity reduction (the result it recommends keeping)

Retain the top k values v = rho u (|u| = 1) and write lower inputs x_j = rho z_j (z_j < u_k). Positive homogeneity removes rho from correctness, and the Jacobian is rho^9, so per arrangement the density integrates over scale to (12/pi^5)(1 + |z|^2)^(−5). The lower-input marginals become target-independent functions of r = u_k (D_k, mu_k, G_k; mu_4(0) = −1/2). A homogeneous MF4 + first-order model has noise-subtracted RMS bias 0.074 (16 dev networks), 0.073 (8 held-out), 0.056 (zoo). Writing u1 = R cos(theta), u2 = R sin(theta), every gate and logit in a region is A + B cos(theta) + C sin(theta), so theta integrates exactly by elementary roots, leaving 2 outer coordinates for MF4 (u3, r) instead of 3. Not yet made cheap: the arrangement and region cost remain.

## 5. Its recommended next direction

Homogeneous coordinates with angular root integration, plus a lower-input approximation built from the actual piecewise-affine margin responses (boundary displacements) rather than indicator expansions.

## Follow-up in this repo (`z13.py`, `z14.py`; REPORT 7.16)

z13 measures each top-4 configuration's true conditional accuracy p(c) with 256 lower-input draws (8,192 configurations; every 16th network + zoo) and scores models built from the same 1 + 6q rows. Local MSE as a share of p(1−p), population medians at q = 16: I0 0.18, clamped first order 0.12, unclamped 0.54, product 0.060, **margin-additive 0.010**. The clamped model reverses I0 on 2.0% of configurations (where p(c) ≈ 0.55), and those carry 65% of its local squared error; its near-zero integrated bias is a cancellation. The margin-additive model (first-order cut-HDMR in m = L_y − max L_o, P(m > 0) exact over the 16^6 node grid) is locally 12x better and its error is no longer concentrated at the boundary, but it is biased low: −0.09 / −0.13 / −0.15 points at q = 8 / 12 / 16. z14: making each rival difference additive instead is worse (−0.42), so the bias is not two inputs pushing different rivals.

The margin model's bias was traced afterwards (`z15.py`–`z19.py`, REPORT 7.16). Its exact limit is −0.18 points. Lower inputs a step or two apart push the same clock or memory neuron across its threshold, and the additive model counts the crossing once per input. Projecting the pair terms of inputs at most 2 positions apart onto one-input terms removes the bias: +0.03 points, RMS 0.048 on all 1,024 networks, comparable to the clamped indicator model's total, with local error 0.003·p(1−p) against its 0.12. It costs about 240–410 forwards per configuration.

The top-4 integral itself was then measured (`z20.py`–`z24.py`). The MF4 anchor integral by tensor Gauss–Jacobi rules in these same CDF coordinates is at best 2.4–2.7× sampling's MSE, at 1–2M forwards, and 4× at 4.8M. The homogeneous 2-D version converges erratically, because the angular range splits where u3 < −R. The lower-input penalty is 68% off the MF4 boundary, so a boundary formula cannot carry it. Fixed subsets of arrangements are 4–10× sampling's RMS, because the dependence on positions does not factor.
