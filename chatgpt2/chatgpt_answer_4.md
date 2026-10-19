# ChatGPT Pro, round 5 (QUESTION5: can a deterministic, mechanistic method match sampling?)

Summary of the answer (chat "Estimator comparison report", 2026-10-19; 52 minutes of work). Its numbers are ChatGPT's own unless marked **checked**. Checks are independent re-implementations from its description, not runs of its code: `z41` (the two-level estimator), `z42` (gating and screening), `z43` (arrangement dependence), `z44` (gate histories) and `round5/check_lower_bound.py`. Calibration and a learned correction, which round 5 did not try because the brief wrongly excluded fitting on the population, are in `z45`–`z47` (REPORT 7.16).

**What is saved, in `chatgpt2/round5/`** (taken from the chat's share link, https://chatgpt.com/share/6ad63c69-eda8-83e9-96ba-9b11c09d00cd):
- `answer.md`: the full answer as markdown, with LaTeX.
- The 10 inline scripts: `r6_core`, `r6_low`, `r6_event`, `r6_exact_anchor`, `r6_correction`, `r6_geometry`, `r6_verify`, `r6_bounds_minimal`, `r6_bounds_check` and `r6_summary`. They are CPU numpy/scipy/numba, run from the repo root beside `lines.py`, and all compile. They have not been run here.
- `run_sequence.sh`: ChatGPT's run order.

**Also saved, downloaded from the original chat:**
- `algzoo_r6_report.txt` (163 KB): the full write-up, proofs, every tested setting and a detailed script audit (§10 below).
- `algzoo_r6_results.json`: 46 result files, 96 methods per split, a 19-point frontier and per-network errors.
- `algzoo_r6_curve.png`: the error-versus-cost frontier for development and held out, against sampling. It stays above the sampling line from 48 to 2.5M forwards, and the gap widens with budget.

**Not saved:** `algzoo_r6_reproduction.zip` was not downloaded. Its new scripts are the inline ones above.

## Verdict

- **No sampling parity.**
- **Best result:** a deterministic two-level estimator. Its RMS is 0.225 points on development and 0.249 held out at 2.5M forwards, 172× and 222× sampling's MSE. That is 6.3× lower MSE than z38 at matched cost.
- **Lower bound:** a deterministic method that only sees in/out answers about a single polyhedral cone in R¹⁰ has worst-case error at least 0.0397·N^(−1/4). That beats sampling's RMS above about 3,000 queries. It does not transfer to these RNNs, to methods that read weights, or to the population average.

## 1. The two-level estimator

p̂ = Q_fine^MF3(I₃) + Q_coarse^MF4(F₄ − I₃), with F₄ = clip(I₄ + Σⱼ(Eⱼ − I₄), 0, 1) the clamped first-order model, Gauss–Legendre q = 8 per lower input.

- **Fine part:** the MF3 anchor, exact S-lines, outer order 6 (36 nodes × 720 lines).
- **Coarse part:**
  - 2 nodes in each of the 3 MF4 outer coordinates (Beta(7,4), Beta(1,3), Beta(2,1));
  - 8 conditional-CDF S midpoints;
  - 8 balanced (R, Q) placements per (M, S) class, 720 arrangements. The rank pairs among the 8 free positions are (0,7), (7,0), (1,3), (3,1), (2,5), (5,2), (4,6) and (6,4).
- **Cost:** each configuration has 49 model rows plus 1 MF3 row, so the correction costs 8 × 8 × 720 × 50 = 2,304,000 forwards per network (2,327,040 with ChatGPT's 1% allowance).

| | split | forwards | RMS | mean | sampling RMS | MSE / sampling |
|---|---|---|---|---|---|---|
| exact-S anchor + correction (**checked**, every digit) | dev | 2,491,426 | 0.2254 | +0.0555 | 0.01718 | 172.0 |
| | held | 2,497,761 | 0.2493 | +0.0440 | 0.01673 | 221.9 |
| interpolated anchor (32 S intervals) + correction | dev / held | 2,718,305 | 0.248 / 0.248 | −0.019 / −0.023 | | 228 / 239 |
| z38 matched (n3/K8/q8) | dev / held | 2,984,429 | 0.621 / 0.620 | +0.51 / +0.52 | | 1,572 / 1,646 |

Zoo, exact-S version (**checked**): +0.023, +0.353, +0.069, +0.360, −0.149, +0.191.

**Error budget** (**checked** pieces):
- **Exact-S MF3 anchor at order 6:** −0.044 mean, 0.063 RMS against the reference `mf3`. Anchor MACs average 164k per network (range 97k–257k).
- **Correction with a free reference anchor:** 0.234 dev, 0.251 held, mean +0.099 / +0.088.
- **The clamped q = 8 model's own error:** 0.065 RMS (brief §5.1, z20).

So most of the 0.25 is the coarse integration of the correction, not the model.

- **Control-variate effect (checked):** the coarse rule integrates I₃ itself 2.07 points high on every network. That error cancels in F₄ − I₃, which is why the two-level form works.
- **MF4 correction on the MF4 anchor:** with the reference `mf4` as a free anchor, the correction over all 5,040 arrangements at one outer node (n1/K8/q8, 2.0M forwards) gives 0.162 / 0.137, mean +0.005 / +0.015 (**checked**, zoo too). An exact MF4 anchor would cost far more than the MF3 one, which is why the estimator uses MF3.

## 2. Does the correction amortize across arrangements? No (direction (a))

The diagnostic uses 256 independent draws per MF4 arrangement on 64 development networks plus the zoo. It is a measurement only and is never used in an estimator.

| correction | SD across arrangement means (median network) | 10th–90th percentile | share explained by (M,S) positions | + relative rank order |
|---|---|---|---|---|
| true − MF3, over the 720 MF3 arrangements | 3.94 points (**checked**: 3.93) | 2.51–5.41 (**checked**) | 21.9% (**checked**: 22.1%) | 45.3% |
| true − MF4, over the 5,040 arrangements | 2.05 (**checked**: 2.02) | 1.15–3.08 (**checked**) | 16.9% (**checked**: 16.4%) | 34.9% |

- **The "1%" clue was a misreading.** z12's "arrangement means carry 1% of the penalty's variance" is a share of a large per-configuration variance. It does not mean arrangements share a correction. The correction varies by 2–4 points across arrangements, and the positions of M and S explain only a sixth to a fifth of that.
- **Thinned subsets, granting exact value integration:**

  | MF3 placements per (M,S) | RMS loss |
  |---|---|
  | 1 | 0.58 |
  | 2 | 0.28 |
  | 4 | 0.23 |

  | MF4 placements per (M,S) | RMS loss |
  |---|---|
  | 4 | 0.27 |
  | 8 (balanced) | 0.11 |
- **Coarse correction rules with a free anchor** (dev / held):
  - MF3 n1/K8/q4 (169k forwards): 0.455 / 0.468.
  - MF3 n2/K16/q8 (2.65M): 0.360 / 0.341. That is 15.7× the cost for little gain.

## 3. State propagation (direction (c)): the mean closure fails

- **Method:** a recursion over components carrying the hidden vector, M, S and both top positions. It uses exact branch probabilities and truncated means (or 2 CDF nodes) per event, with optional merging by positions and CDF bins.
- **Result:** +4.3 to +5.2 points of bias at every setting from 200 to 18k forwards, with or without merging. That is the size of MF2's leakage.
- **Reading:** collapsing each branch to a point is a mean-field approximation and inherits its bias. An exact law over affine components keeping past-input dependence was not attempted. Not checked.

## 4. Error geometry (direction (d)): histories proliferate

| time | median distinct gate histories | test mass in unseen histories |
|---|---|---|
| 3 | 474 | 0.18% |
| 5 | 7,774 | 8.9% |
| 7 | 20,980 | 27.8% |
| 9 | 38,936 | 55.7% |
| 10 | 46,867 | 68.8% |

8 dev networks, 32,768 discovery and 32,768 test inputs each.

- **Checked qualitatively.** At t = 3 my counts match (476, 0.21%). Later they are larger (t = 10: 54.5k distinct, 82% unseen), probably from a different history definition or network subset. The conclusion is the same.
- **Error fragments:** with the activation pattern, both true top positions and the predicted position fixed, a median of 5,846 error fragments is observed, and 97% of test errors fall in fragments absent from discovery. Not checked.
- **"Maximal convex pieces" is ill-defined.** {x ≤ 0} ∪ {y ≤ 0} has a continuum of maximal convex subcones. Count a chosen partition or a minimum cover instead.

## 5. Low budgets (direction (e))

| rule | forwards | RMS (dev) | MSE / sampling |
|---|---|---|---|
| MF2 n1/q2 | 96 | 7.8 | 7.9 |
| MF3 interpolant n1/K2 | 683 | 2.5 | 5.8 |
| MF3 interpolant n2/K4 | 5,449 | 1.2 | 10 |
| MF3 interpolant n4/K16 | 87k | 0.59 | 41 |

- None reaches parity.
- All 72 preplanned anchor × correction combinations per split are in its results file.
- Not checked.

## 6. A restricted lower bound

- **Model:** the error set is one closed polyhedral cone in R¹⁰ with any number of facets and Gaussian mass at most q* = (1 − √(1 − 4·0.069))/2 = 0.07456. The algorithm makes N adaptive Boolean membership queries; intermediate computation is free.
- **Theorem:** worst-case error L_N = (2/5)[G(R) − G(R cos θ_N)], with
  - G(r) = P(X₁₀ ≥ ‖X₁:₉‖/r), which has a closed form in α = arctan r;
  - G(R) = q*, so R = 1.90165;
  - θ_N = (256/(175N))^(1/8).
- **Proof:** an adversary answers as if the error set were the circular cone. The convex hull of the positive queries misses a radial shell on 4/5 of directions, because N caps of radius θ_N cover at most 1/5 of S⁸. A fine inscribed polytope agrees with every answer, so two legal cones with different masses give the same transcript.
- **Check:** I read the proof and it is sound. The rates match the known N^(−2/(d−1)) rate for convex bodies with d = 9.
- **Numbers (checked):**

  | N | L_N (points) | sampling RMS | L_N² / sampling MSE |
  |---|---|---|---|
  | 100 | 1.18 | 2.63 | 0.20 |
  | 10³ | 0.685 | 0.831 | 0.68 |
  | 10⁴ | 0.391 | 0.263 | 2.2 |
  | 10⁵ | 0.222 | 0.083 | 7.1 |
  | 10⁶ | 0.125 | 0.026 | 23 |
  | 10⁷ | 0.070 | 0.0083 | 72 |

  The asymptotic constant is 0.0397229.
- **Extension:** it also covers label-only affine-segment traces, at 2 queries per segment. It does not cover traces that reveal logits, affine coefficients or gates.
- **Scope:** the hard instances need facet counts that grow with N, while a fixed RNN has bounded complexity. A minimax bound over a larger class does not lower-bound the RNN family, and saying anything about the trained population needs a separate argument.

## 7. Corrections to the brief's "floor" claims

- **The additive floor (brief §5.1, "a floor near 0.07 points") is not universal.**
  - Counterexample (**checked**): m = |X+Y| − |X−Y| + 2|Z| has ANOVA projection 2|Z|, so P(m_A > 0) = 1, but P(m > 0) = 5/6. The additive g = X + √3|Z| also has P(g > 0) = 5/6.
  - What is valid: models converging in L² to the projection inherit its sign error.
- **Uncovered mass (z26) is a certificate radius (u/2), not an error floor.** z26's tails are estimates, not certified masses.
- **z22's non-locality** concerns its margin proxy and S-grid crossing test. It does not lower-bound all estimators.

## 8. Audit of the supplied scripts

- **z21 and z37** attach the charge of an optimized exact prefix integrator that is not in the archive. Actual row counts at K = 256 are 139× and 121× the quoted per-node charges. The brief already said that implementation is absent.
- **MF2's cost cannot be extrapolated from MF3's.** MF2's lower inputs move as μ(S), so its input path is curved.
- **z18 at q = 8:**
  - Adjacent pairs cost 241 rows, pairs within 2 positions average 411⅔ (range 241–625), and all pairs cost 1,009.
  - z18 evaluates all pairs before masking, so the reduced costs need those evaluations actually omitted.
  - There is no projection bug.
- **z38's count and comparator check out.**
- **Reference noise:**
  - RMS reference SE is 0.026 points.
  - At C = 10⁷, reference variance alone adds C/2²⁰ ≈ 9.5 to a raw MSE/sampling ratio.
  - The shared reference inputs correlate the networks' errors.

## 9. Proposed next experiment

- **Experiment:** integrate the two-level correction exactly in S.
  - Trace the correct intervals of all 49 model streams and the MF3 row, then merge their endpoints. Between adjacent endpoints the clamped correction is constant.
  - Clip on each common interval, not after averaging.
- **Pilot gate:** freeze an 8-network pilot. Continue only if correction RMS drops by at least 0.10 or charged cost by at least 4×.
- **Partial-success gate:** held-out correction RMS at most 0.10 with correction cost at most 10× the anchor's.
- **No promise:** ChatGPT does not claim it closes the 220× gap.

## 10. Extra findings in the downloaded report (`round5/algzoo_r6_report.txt`, "Round 6 audit")

- **The exact-S tree costs are unverified, and the REPORT's only in-range wins depend on them.**
  - Where the charges come from:
    - z21, z23, z24, z27, z28 and z31 charge 9,300 forwards per MF4 outer node.
    - z37 charges 1,528 per MF3 node and 191 per MF2 node.
    - These are ChatGPT round 4's own measurements of an optimized prefix tree (`chatgpt_answer_3.md`). That code was never downloaded or checked.
  - Against the supplied dense grids, the charges are 139× (z21 at K = 256), 277× (K = 512) and 121× (z37) too low. ChatGPT's example: MF3 at order 4 costs 2,949,120 forwards rather than 24,448. At that cost its 0.127 RMS is 61× sampling's MSE, not 0.51×.
  - The fair comparison is the exact tracer the repo does have, `lines.trace_line` (no prefix sharing). With it, an MF3 node costs 4,566 forwards on development and 4,742 held out (**checked**, from the order-6 run in §1). That is about 3× the charge.
  - So with verified code, the MF3 anchor's "1.5–2.6× better than sampling" becomes roughly 1.2–2× worse. This affects REPORT §7.16 (line 1255) and the scorecard (line 1344).
  - The z21 convergence curves (z21, z27, z28, z31) have the same caveat.
  - The prefix tree remains a claimed 3× speed-up until reproduced.
- **Reference precision varies across the z21 logs.** The K = 64 and K = 256 orders 4–6 use 2²² inputs (SE about 0.014 points), not the 0.003 the brief states (§5.2, line 174). The K = 512 rows and orders 9–10 use 2²⁶.
- **MF2's anchor is much easier to sample.** Its mean p(1−p) is 0.40 of true accuracy's, against 0.97 for MF3 and MF4. Anchor-versus-sampling comparisons should use each cohort's own p(1−p).
- **z18's convolution work is not counted.** It is about 4q⁴ scalar operations, roughly 2.9 forward-equivalents at q = 8 and 46 at q = 16, before sorting and search.
- **z18's 0.048 is a noise-subtracted figure.** The raw RMS on all 1,024 is 0.064 with 0.043 measurement noise. The postprocessing that produced 0.048 is not in the archive.
- **The brief's "any white-box lower bound would be a circuit lower bound" is too categorical.** A bound first needs a computational model, primitives, precision and a target. Over a fixed finite population, a lookup table answers everything unless the admissible computations are formalized.
- **Follow-up idea for state propagation:**
  - At each propagated state, split the next input's axis at every gate threshold plus M and S, and integrate each interval analytically. This keeps correlations within each point and shares time prefixes.
  - Component counts can still grow exponentially, so it is an experiment, not a proven replacement.
