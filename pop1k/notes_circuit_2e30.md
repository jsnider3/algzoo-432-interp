# Circuit model on the converged (2³⁰) population (2026-10-18)

These are the `notes_circuit_2e28.md` analyses rerun on `pop1k/seed0.npz` (step 1,048,576). The outputs are in `pop1k/z{2..7}_seed0.txt` and `chatgpt2/roles_run_2e30/`.
- Accuracy is 0.924 ± 3.4 points across networks; the 5th, 50th and 95th percentiles are 0.860, 0.932 and 0.964.
- Under ARC's criterion (p(1−p) ≈ 0.070), sampling reaches 1 point at 700 forwards, 0.5 at 2,800, 0.3 at 7,800 and 0.1 at 70,000.

## What changed from 2²⁸

Almost nothing structural. Error rates are a little lower everywhere.

| | 2²⁸ | 2³⁰ |
|---|---|---|
| accuracy (z4 pool) | 0.908 ± 3.61 | 0.923 ± 3.47 |
| variance share of ties (M–S + S–R) | 33% | 37% |
| variance share of middle + clean | 66% | 62% |
| last-step handoff error (non-tie) | 14.1% | 12.3% (SD 13.5, corr. with accuracy −0.81) |
| same, with 0 → 5+ comparators | 40% → 4% | 38% → 3% |
| mean field − true (points) | +0.42 ± 0.585 | +0.27 ± 0.52 |
| mean field after one affine calibration | 0.55 | 0.50 |
| leakage predicted from network features | 0.43 | 0.39 |
| best population predictor (ChatGPT hybrid GBM) | 1.05 | 1.00 |
| weights-only GBM | 1.17 | 1.11 |
| latency-profile GBM (8 sampled rates) | 0.79 | 0.74 |

**Mean-field error shrank only modestly with training.** Its RMS is 0.59 points. Evaluated exactly and for free, mean field would be worth about 2,000 samples, and it plateaus there. So, per the plan, the lower-input leakage needs its own mechanistic model before mean-field quadrature can match sampling across the compute curve.

## Two strategies (scratch `strat3.py`)

- **The selectivity score stays bimodal**, with only 35 networks between 0.3 and 0.5.
- **The strategy is fixed by 2²⁸.** 98.8% of networks keep their pointer label (8 gained one, 4 lost one), and selectivity correlates 0.997 between the two snapshots.
- **Telling them apart is cheap.** The label from 32 random probes agrees with the 4,096-probe label on 98.6% of networks; 128 probes give 99.4%. Weights alone give AUC 0.86.
- **The strategies do not separate in error structure.** This contradicts the zoo-based expectation that selection networks confine errors to ties.

  | | n | accuracy | MF − true (points) | handoff k=0 | handoff k=3 | errors on ties | errors on clean |
  |---|---|---|---|---|---|---|---|
  | selection (pointer) | 549 | 0.9275 | +0.23 ± 0.48 | 11.7% | 1.7% | 58% | 11% |
  | memory (no pointer) | 475 | 0.9195 | +0.31 ± 0.56 | 12.9% | 2.3% | 58% | 13% |

- **So a per-strategy estimator would not help much.** The leakage spread within each group is nearly the population's.
- Leakage correlates −0.43 with accuracy and +0.37 with last-step handoff error.

## Informative inputs (`z7.py`)

- **The most informative inputs are last-step handoffs**: 85% of the top 500, against 9% of all inputs. They are spread across clean, middle and M–S-tie cases. The best single input has corr −0.59 with accuracy.
- **Greedy-selected inputs with a ridge fit plateau at about 1 point:**

  | K | selected inputs | K random inputs |
  |---|---|---|
  | 16 | 1.59 | 2.60 |
  | 64 | 1.17 | 2.44 |
  | 256 | 0.99 | 1.50 |
  | 1,024 | 0.99 | 0.65 |

  This is the same shape as at 2²⁸.
- **Role features predict which networks fail a given top input** with AUC 0.75–0.90 (cross-validated). The correlation with the comparator count is −0.3 to −0.4 for handoff inputs and about −0.1 for direct-write inputs.

## Bugs fixed on the way

- `z3.py` and `z6.py` parsed `init_Wih` as a step; they now filter on `s<digits>`.
- ChatGPT's `roles_mechanism.py` has the same bug, so it was rerun with `--snapshot s1048576` and left unedited.

## Mechanistic model of the lower inputs (`z8.py`, `pop1k/z8_seed0_{closures,floor,score}.txt`)

Given the top-3 values and positions, the 7 lower inputs are iid N(0,1) truncated below R. The aim was to compute P(correct | top-3) from the weights, with no population fitting, and then integrate it over top-3 configurations.

1. **Propagating means and covariances fails.** This is the step from the plan. The table is from 48 networks × 48 configurations against 4,096 redraws each. The true margin between the target logit and the runner-up has median 3.95 and median SD 0.82.

   | closure | margin mean, median abs error | margin SD ratio, median (IQR) |
   |---|---|---|
   | soft gates + independent residuals | 1.57 | 10.7 (5.3–25) |
   | soft gates, Stein part only | 0.87 | 2.2 (1.2–4.6) |
   | exact bivariate ReLU covariance | 1.38 | 7.7 (4.0–17) |
   | **hard gates at the mean-field point** | **0.26** | **0.85 (0.53–1.02)** |

   - Each neuron's own mean and SD come out roughly right in most networks.
   - The answer is a small difference of large activities: the margin is about 1–4 while the hidden state is 20–90. The network cancels lower-input noise in that difference, and any closure that treats the ReLU residuals as independent destroys the cancellation.
   - The soft-gate means also diverge in 5 of the 1,024 networks, those with recurrent gain between 4 and 24.
2. **The delta method fails too.** It uses the hard gates at the mean-field point, which keep the cancellation, plus a Gaussian for the margins.
   - Counting every competitor, its accuracy is about 10 points too low, because lower-input logits win too often (61% of its errors against 12% in truth). The Gaussian lets a lower input run past R.
   - Restricted to the max and R competitors it does better, but is still −0.82 ± 1.06 points across all networks. The cause is gate flips: per configuration the mean is off by about 0.3, and the SD ratio ranges from about 0.5 to 1.05.
   - **The Gaussian shape is not the problem.** A Gaussian with the empirical mean and SD reproduces the true P(margin > 0) in every region.
3. **First-order ANOVA (additive model): the best version.**
   - Method: each lower input's own effect on the two margins comes from exact network passes at q equal-probability nodes, with the others at their mean. The effects are summed, and P(both margins > 0) is taken under the sum.
   - Accuracy at q = 8:
     - The margin SD ratio is 0.98 (IQR 0.92–1.11) and the per-configuration MAE is 0.035.
     - Against truth it is −0.39 ± 0.44 points (RMS 0.585), against +0.45 ± 0.60 (RMS 0.75) for mean field with the same two competitors.
     - The truth itself carries about 0.13 points of noise.
   - Where the bias sits: it is pessimistic in the middle region (−0.23) and in the ties (−0.24), and optimistic for S < 0 (+0.10). That points to negative interactions between lower inputs, which first order ignores.
   - Its Gaussian version is just as good.
4. **The top-3 integral is the real obstacle.** The lower inputs carry only 45% of the variance (median); the other 55% lies between top-3 configurations. So even an exact P(correct | top-3) would beat sampling by at most 1.8× if configurations were sampled. The additive model costs 1 + 3.85q forward-equivalents per configuration.
5. **End-to-end scorecard.** The ANOVA model is integrated over an ordered-Sobol top-3 design and scored against the 2²⁰-input accuracy.

   | q | configurations | cost (forwards) | RMSE (points) | sampling at same cost | worth (samples) |
   |---|---|---|---|---|---|
   | 2 | 256 | 2,227 | 1.16 | 0.56 | 518 |
   | 2 | 1,024 | 8,909 | 0.71 | 0.28 | 1,406 |
   | 4 | 1,024 | 16,794 | 0.63 | 0.20 | 1,782 |

   That is 4–10× short of sampling. For comparison, plain mean field on the same kind of design (z5, cost 1 per configuration) is worth about 320 samples at 256 configurations (1.25×) and about 810 at 1,024 (0.8×). It roughly matches sampling at low budgets and then hits its floor.

**Verdict.**
- The additive model is the first per-configuration mechanistic model that predicts P(correct | top-3) well: an MAE of 3.5 points per configuration, and a cross-network SD 25% smaller than mean field's.
- But it doesn't move the ARC comparison, for two reasons:
  - its floor of about 0.55 points is set by interactions;
  - more importantly, most of the variance sits in the top-3 integral, where the method spends 9–16 forwards per configuration.
- Matching sampling would need both:
  - a top-3 integral much better than QMC, using structure across arrangements and values: homogeneity, errors settling after about 3 steps since the last event, and smooth blurred tie boundaries;
  - a lower-input model costing only about 1–2 forwards per configuration.

### Follow-ups: cheaper and more accurate lower-input models (scratch `mp9.py`, `mp10.py`; same 2,048 configurations and truth as `z8 floor`)

| variant | cost (fwd/config) | error vs truth (points) |
|---|---|---|
| first-order additive, Gaussian, q = 8 (z8) | 32 | −0.39 ± 0.44 |
| first-order additive, exact sum, q = 4 | 16 | −0.15 ± 0.45 |
| each input exact at its arrival step, then linear (adjoint) | 3 | −8.8 ± 6.8 |
| exact through the max, then linear | 11 | −1.7 ± 1.5 |
| exact through the last top-3 event, then linear | 15 | −0.39 ± 0.64 |
| second-order (all 21 pairs), q = 4 | about 150 | +0.21 ± 0.35 |
| first / second order with all 9 competitor logits | 16 / 150 | −1.95 ± 1.20 / −1.22 ± 0.61 |

**1. Linear propagation does not work anywhere.**
- The adjoint is correct (it matches finite differences to 1e-5 at small perturbations), but real perturbations flip gates downstream.
- For inputs that arrive before the max, the linear error reaches about 150 against typical effects under 1.
- Even after the last top-3 event the linear tail adds error, consistent with gates flipping during stretches with no top-3 event.

**2. The bias depends on the node rule.** It is −0.15 at q = 4 with an exact sum, against −0.39 at q = 8 with a Gaussian. The bias estimates should be read as ±0.2.

**3. Pairwise interactions are real.** The spread drops from 0.45 to 0.35 points (about 0.33 after removing truth noise). But second order costs about 150 forwards per configuration, and it still leaves a floor of about 0.35, which caps its worth at about 5,700 samples at any compute.

**4. The S<0 offset is lower-position wins.**
- The two-competitor model is +0.10 too optimistic for S<0 at every order, because there the network often picks a lower input's position.
- With all 9 competitors, S<0 becomes exact (−0.004), but every other region turns pessimistic: summed effects on the lower positions' own logits produce spurious wins.
- So all 9 competitors would be needed only for S<0, which is 1% of the mass.

## Option 1, a better sampler: exact integration along the 2nd-max value (`v1.py`, `pop1k/v1_*.txt`)

- **The estimator.** Given everything but S's value, S is N(0,1) truncated to (R, M) and the target stays at pS.
  - Along that segment the network is piecewise affine; the median line has 6.8 pieces.
  - So P(correct | rest) is a sum of exact Φ masses, and averaging it over samples is unbiased (checked on 2¹⁸ inputs).
- **Ceilings** (per-sample variance reduction p(1−p)/Var, random inputs):

  | line | zoo models | population median |
  |---|---|---|
  | x₁₀ (REPORT 7.15) | 1.8–2.1 | 1.9 |
  | **S in (R, M)** | **2.1–4.6** (mystery 4.6) | **3.0** |
  | M in (S, ∞) | 1.5–2.0 | 1.7 |
  | R in (4th, S) | 1.6–2.1 | 1.9 |

  - Paired with ordered Sobol at 2¹², the per-sample factors are 2.3–3.8 for x₁₀ and 5.2–12 for S.
  - An S-free 10-D design, which never samples S, gives 6.8–22.
- **Cost** (multiply-adds; a forward is 2,880):
  - The S tracer costs 5.4 forwards per line if every piece is recomputed, or 4.4 if sibling pieces are updated incrementally.
  - x₁₀ costs 3.05 and 1.18 under the same two countings.
- **Net efficiency against plain sampling at matched compute** (zoo, 2¹⁴, 32 scrambles; full / incremental counting):

  | estimator | full | incremental |
  |---|---|---|
  | ordered Sobol | 1.3–3.1× | — |
  | Sobol + x₁₀ | 1.1–2.0× | 2.8–5.3× |
  | Sobol + S | 1.2–2.9× | 1.5–3.4× |
  | combination: S line if pS ≥ 5, else x₁₀ | 1.4–3.9× | 2.3–6.4× |

  On the population, medians are 1.25/1.51× for Sobol + S and 1.27/3.28× for Sobol + x₁₀.
- **Verdict:** the S line has the best ceiling, but its tracing cost eats the gain. Every variant stays in the same 2–6× band, a constant factor. These figures are noisy (±25–50%; the plain-sampling control reads 0.8–1.6×).

## Option 2, deterministic: the exact mean-field integral (`z9.py`)

- **The estimator.**
  - For each of the 720 top-3 arrangements and each R, the mean-field input is affine in (M, S), so the network is piecewise affine on the slice R < S < M.
  - `z9` walks the slice's linear regions and integrates φ(M)φ(S) exactly over each region's incorrect polygon.
  - The R integral is composite Gauss–Legendre with weight φ(R)Φ(R)⁷.
  - There is no sampling anywhere. Single slices match Monte Carlo of the mean-field network (`z9 check`).
  - Cost counts every pass, gate check, clip and polygon integral, at 5,120 flops per forward.
- **It converges to the mean-field value.**

  | model | R nodes | estimate | cost (forwards) | error vs truth |
  |---|---|---|---|---|
  | mystery | 9 | 0.951293 | 3.8M | −0.183 points |
  | mystery | 18 | 0.953164 | 7.7M | +0.005 (luck: the R<0 panel is off by −1.2e-4) |
  | mystery | 27 | 0.953360 | 11.5M | +0.024 |
  | mystery | 36 | 0.953262 | 15.3M | +0.014 |
  | mystery | 54 | 0.953207 | 23.0M | +0.009 |

  - With R<0 split finer (16 nodes on [−0.5, 0]), the mystery estimate is **0.953213**, converged to 2e-7. The sampled mean-field reference is 0.95323 ± 0.00005, and the truth is 0.953119.
  - The integrand bends sharply on R ∈ (−0.15, 0). A single panel there converges only algebraically.
- **The mean-field bias sets a parity budget**, p(1−p)/bias². Above that cost, any mean-field estimate loses to sampling, even one computed for free.

  | model | converged mean field | truth | bias (points) | parity budget | cost of a converged integral |
  |---|---|---|---|---|---|
  | mystery | 0.953213 | 0.953119 | +0.009 | 5.0M | about 15–20M |
  | seed 2 | 0.948323 | 0.948510 | −0.019 | 1.4M | about 45M |
  | seed 0 | 0.910885 | 0.910090 | +0.080 | 0.13M | about 5M |
  | population (RMS leakage 0.59) | | | 0.59 | about 2,000 | |

  Seed 2 at 27 nodes read −0.009, but that was also luck: its R<0 panel was off by 1e-4.
- **Where the error and the cost sit** (mystery):
  - R < 0 is 5.5% of inputs, 19% of the error, and 72% of the cost: about 2,000 forwards per slice against about 200 for R > 0.
  - Seed 2: 18% of the error and 80% of the cost.
  - The error is spread over arrangements: it takes 139 of 720 to hold half of it, and 349 to hold 80%. Grouping arrangements does not compress it.
- **Verdict.**
  - The top-3 geometry under mean field is now solved exactly and deterministically. As a cost, it is about the same as sampling the mean-field network to the same precision, because there are too many linear regions.
  - The bar is blocked twice over:
    - the integral is 3–40× over the parity budget;
    - the mean-field bias itself (lower-input fluctuations, see z8) caps every mean-field-based estimate at 2,000 to 5M forwards' worth of precision.
  - A mechanistic estimate that meets the bar needs both:
    - a leakage model accurate to about 0.01 points, where z8's best is 0.35;
    - a top-3 integral about 100–1,000× cheaper than enumerating regions.

## The leakage model: top-4 jointly, the 6 smallest inputs independently (`z10.py`, `z11.py`, `lines.py`)

**Anatomy (`z10`, 2²⁰ inputs, 1,024 networks + zoo; `pop1k/z10_seed0.txt`).**
- Mostly a repeat of `notes_circuit_2e28.md` §2: the leakage is spread over all lower inputs. No single position, zone, event subset or stand-in value localizes it.
- MF3 against the truth flips about 5% of inputs. The flips occur at every mean-field margin.
- **New: where it lives.**
  - S < 0 and R < 0 < S contribute +0.04 to +0.05 each; mean field gets those inputs wrong too.
  - All the spread is in R > 0, where two mechanisms nearly cancel:
    - the true network picks the 4th-largest's position: +0.45 points, which MF3 cannot produce;
    - S-vs-R swaps: 3% gross flips both ways, net −0.35.
  - Keeping the 4th exact (MF4) removes both mechanisms. What remains is a fairly uniform optimism, +0.30 ± 0.28.

**The model.**
- P(correct | top-k) = I₀ + Σⱼ (Eⱼ[I] − I₀).
  - I₀ is the network on the MFk point.
  - Eⱼ[I] is the accuracy with lower input j alone varied over its law (N(0,1) truncated below the k-th largest), the others at their mean.
- Eⱼ is computed at q equal-probability nodes, or exactly by tracing the piecewise-affine line (`lines.trace_line`, which generalizes `v1.trace_S`; it matches a 20k-point grid to 4e-5).
- Scored per network against the true accuracy on the same 2¹⁸ inputs, where pool noise is 0.03–0.04 per network. The exact rows use 2¹⁶ inputs, every 8th network, and noise 0.066.

| model | forwards per configuration | mean (points) | SD | RMS | corr. with true leakage |
|---|---|---|---|---|---|
| MF3 alone | 1 | +0.247 | 0.511 | 0.567 | |
| MF3 + first order, q = 4 | 29 | +0.189 | 0.190 | 0.268 | 0.93 |
| MF3 + first order, q = 8 | 57 | −0.000 | 0.156 | 0.156 | 0.96 |
| MF4 alone | 1 | +0.304 | 0.284 | 0.416 | |
| MF4 + first order, q = 4 | 25 | +0.124 | 0.092 | 0.154 | 0.96 |
| MF4 + first order, q = 8 | 49 | +0.032 | 0.061 | 0.069 | 0.98 |
| **MF4 + first order, q = 16** | 97 | **−0.013** | **0.055** | **0.057** | 0.98 (slope 0.99) |
| MF4 + first order, exact lines | about 65 | −0.051 | 0.078 | 0.093 | |
| MF5 + first order, q = 8 | 41 | +0.037 | 0.045 | 0.058 | 0.98 |
| MF6 + first order, q = 4 | 17 | +0.076 | 0.059 | 0.096 | 0.97 |

- **Node resolution, not interactions, was the q = 4 shortfall.**
  - The flips from a small input are narrow and sit near its upper end, in near-ties with the 4th. Equal-probability midpoints under-sample them.
  - The mean drifts from +0.12 to +0.03 to −0.01 as q goes 4, 8, 16, and the exact limit is −0.05.
- **After removing pool noise**, the exact first-order model's error is about −0.05 ± 0.04 points per network (RMS about 0.07), against 0.57 for MF3.
  - The earlier best was z8's pairs, about 0.35–0.41 RMS.
  - REPORT 7.13's first order around MF3 overshot. The reason is that the 4th-largest's near-ties with R are a joint effect. Making the 4th exact is what makes the rest additive.
  - Around MF3, first order is unbiased at q = 8 but has 3× the spread (0.156).
- **Mechanistic reading.**
  - The decision depends on the top-4 values and positions jointly.
  - Each of the 6 smallest inputs acts on it independently, through the real network, with no interactions among them to within about 0.05 points.
  - It is a presumption of independence that holds, once the 4th-largest is excluded from it.

**What it buys against the ARC bar.**
- With the top-4 integral done for free, this leakage model's ceiling is p(1−p)/ε² ≈ 0.07/(0.0007)² ≈ 150k forwards' worth of precision. MF3's ceiling is about 2,000.
- It is not yet an estimator:
  - it costs about 65 forwards per top-4 configuration;
  - the top-4 integral has 5,040 position arrangements and 4 values;
  - integrated by sampling configurations, it is about 65× less efficient per forward than sampling the network.
- So the bottleneck is again the top-k integral, now with an 80× higher ceiling.

**What the top-4 integrand looks like (`z12.py`, `pop1k/z12_seed0.txt`; every 16th network + zoo, 2¹⁷ sampled configurations, q = 16).**
- I₀ has variance 0.97·p(1−p) over configurations, and P̂ = I₀ + penalty has 0.79·p(1−p). So sampling configurations gains only about 1.3× per configuration, while costing about 65 forwards each.
- The penalty is nonzero on 12% of configurations (p10–p90: 7–19%). Where the MF4 margin is below 0.5 (6% of configurations) sit 80% of its square and about half its mean; within margin 1 (12%) sit 92% of its square.
- The 5,040 arrangement means explain 6–8% of the variance of the truth, I₀ and P̂ (25% for seed 1), and 1% of the penalty's. The rest is the value integral within each arrangement.
- So a deterministic top-4 integral has to locate each arrangement's boundary in the 4 values, and apply the penalty only in a shell around it. This is posed in `QUESTION4_FOR_CHATGPT_PRO.md`.

## ChatGPT round 4 and the local check (z13, z14; 2026-10-19)

ChatGPT (chatgpt2/chatgpt_answer_3.md) found no deterministic estimator on the sampling line. Its best is MF3 with exact S lines and shared-prefix arrangement trees: 0.58 RMS at 36k forwards against 0.14 for sampling (16x MSE). It found that the clamped first-order model reverses the anchor at its boundary: Σ E_j − 5 I0 drops by 5 when I0 steps up, so P̂ goes 1 → 0 whenever 1 < Σ E_j < 5. Reproduced on network 0: Σ E = 3.168, true conditional accuracy 0.72.

z13 (8,192 configurations x 256 lower-input draws; every 16th network + zoo) gives local MSE / p(1−p) at q = 16 as follows:
- I0 0.18
- P̂ 0.12
- unclamped 0.54
- product 0.060
- margin-additive 0.010

Reversals cover 2.0% of configurations (p(c) ≈ 0.55 there) and carry 65% of P̂'s local error, so P̂'s ~0 integrated bias is a cancellation. The margin-additive model is locally 12x better and continuous across the boundary, but it is biased −0.09/−0.13/−0.15 points at q = 8/12/16 (negative in 61/64 networks). z14: per-rival additivity is worse, at −0.42. The source of the margin bias is open; pair terms in the margin would be the next test.

Fixed: lines.trace_line ties now follow the lowest-index argmax rule; z11 exact mode raises on overflow.

## The margin model's bias (z15-z19; 2026-10-19)

- z15 (draw by draw, exact one-input effects): first order in the margin is −0.18 (p10 −0.30, p90 −0.09).
  - Sign errors: 0.67% of draws false-wrong, 0.50% false-right.
  - The residual is mean-zero and 39% of draws are exactly additive, but it opposes large Σd.
  - The median anchor gives the same, −0.179.
  - Second order at the draws: −0.075.
- z16: shared gate flips (≥2 one-input runs) carry 79% of the net bias; rival change 52%; neither 5%.
- z19: shared flips sit in clock-like "other" and always-on neurons (88% vs 72% of neurons), not in comparators or pointers. Pair-interaction energy is 62% at gaps 1–2 and 64% in pairs both before S.
- z17: sampled first-order ANOVA (K = 32 common backgrounds) gives +0.06. K < pool without common backgrounds is just noise.
- z18: second-order cut terms projected onto one-input functions keep the exact convolution.
  - All pairs over-correct (+0.05).
  - Pairs ≤ 2 apart are best: held out +0.02, noise-subtracted RMS 0.042 (dev 0.073), local MSE 0.003·p(1−p).
  - The clamped indicator model on the same footing: local 0.12. Its RMS is barely resolved, because its per-network noise is 0.15–0.20.
  - All 1,024 networks (2,048 configurations): pairs ≤ 2 apart +0.026, RMS 0.048; indicator −0.044, RMS 0.03 (0–0.05). So the two are comparable in total; the margin model's gain is local only.
  - Gauss–Legendre nodes in the CDF converge by q ≈ 6–8; midpoints had not converged by q = 16 (that was z13's "bias grows with q").
  - Cost about 240 (q = 6) to 410 (q = 8) rows per configuration.

## The top-4 integral, measured (z20-z23; 2026-10-19)

- z20 resolves the clamped indicator model's total (256 networks, 2^18 configurations, noise 0.019).
  - q=16 midpoints: −0.019 mean, RMS 0.053.
  - GL8: −0.044, RMS 0.067.
  - Unclamped: +0.125, RMS 0.144.
  - On all 1,024 networks (2^17 configurations, noise 0.027): q=16 midpoints −0.015 / RMS 0.052; GL8 −0.040 / RMS 0.065; unclamped +0.136 / RMS 0.160.
  - So the indicator and pair-margin models are comparable in total, at about 0.05.
- z21: the MF4 anchor integral E[I0] by a tensor Gauss–Jacobi outer rule in CDF coordinates, brute-force S grid, all 5,040 arrangements. Cost is taken as about 9,300 forwards per node (ChatGPT's exact-S tree).
  - K=64, orders 1–4: RMS 3.6 / 1.27 / 0.18 / 0.22 against sampling's 0.28 / 0.10 / 0.055 / 0.035.
  - K=256, orders 4/5/6: RMS 0.18 / 0.039 / 0.030 against sampling's 0.035 / 0.025 / 0.019. The K=64 floor near +0.05 was the S grid.
- z23: homogeneous coordinates (scale out, 2-D outer, θ arcs) converge erratically: 5.3, 2.7, 2.8, 0.68, 0.22, 0.34, 0.12, 0.46, 0.35, 0.079, 0.13, 0.28 at orders 1–12. The kinks come from the θ domain splitting where u3 < −R. No gain.
- z22: the lower-input penalty is not boundary-local. Of the first-order margin model's −0.54 points, −0.36 (68%) sits on S-lines where the anchor never changes its answer.
  - The two-term (mean shift + smearing) formula has the wrong sign (+0.78), because crossings sit at near-ties at the line ends.
  - Freezing the law at crossings recovers only −0.19.
- The correction inside the enumeration costs about 5,040 × 4 S-points × 240 rows ≈ 4.8M forwards per outer node. The only decomposable version (unclamped) is biased +0.14 (all 1,024; +0.125 on every 4th).
- z21 at K=512 with a 2^26 reference: order 6 0.031 (sampling 0.019), order 7 0.032 (0.015), order 8 0.026 (0.013). The error falls more slowly than sampling's, so the MSE ratio rises from 2.4 to 4–4.5 beyond 2M.
- z24: per-arrangement anchor accuracy has SD 8 points across arrangements.
  - (M, S) positions explain 28% of its variance, or 41% with the R/Q order relations.
  - Additive in the 4 positions: 18%; all pairwise position interactions: 48%.
  - Fixed R/Q-placement subsets are 4–10× sampling's RMS at the reduced cost.
- z25 (closed form, one exact linear logit map per activation pattern, patterns taken at K grid points per arrangement):

  | K | build cost (forwards) | coverage | error, nearest map | error, exact where covered | sampling at that cost |
  |---|---|---|---|---|---|
  | 1 | 5k | 17% | 2.7 | 2.7 | 0.37 |
  | 16 | 81k | 50% | 1.2 | 0.83 | 0.093 |
  | 81 | 408k | 67% | 1.0 | 0.48 | 0.041 |
  | 256 | 1.3M | 76% | 0.76 | 0.33 | 0.023 |

- z26: an arrangement's value space holds a median of 460 linear regions (160–700). The top 1 / 10 / 100 patterns carry 20% / 67% / 95% of the mass.
- z26 rerun at 2^16 draws: 560 patterns seen (median), 460 at 2^15.
  - Coverage by the top K = 1 / 10 / 30 / 100 / 300 / 1000 patterns: 20% / 66% / 85% / 95% / 98.7% / 99.7%; Good–Turing unseen mass 0.23%.
  - Oracle closed form with z25's 1.4% fallback rate: 0.47 at 50k forwards (sampling 0.12), 0.21 at 151k (0.068), parity near K = 300 (1.5M).
- z21 rate: from order 5 to 8, error falls as nodes^−0.25, so the MSE ratio rises 2.5 → 4.6.
- z28: Smolyak sparse grids of the same rules give 0.83 / 0.32 / 0.38 / 0.46 at levels 5–8 (260k–4M). They do not converge.
- z27 (homogeneous, 9 smooth pieces): with 64 θ-points per arc, orders 4–7 give 0.145 / 0.045 / 0.076 / 0.067 (θ floor). With 256 points per arc: 0.176 / 0.041 / 0.052 / 0.035 (rate nodes^−0.19). No better than the CDF tensor rule.
- z29 (clock-state simplification): no neuron is always on (the most active is on 94.8% of steps, median).
  - Making the 7 neurons active ≥ 90% linear: bias 2.3 RMS, 1.8% of answers change, 232 regions per arrangement left.
  - Making 10 neurons (≥ 80%) linear: bias 16 points, 104 regions.
- z30 (measured oracle closed form, 100 arrangements, noise removed):
  - Greedy selection: K = 3 / 10 / 30 / 100 give 0.33 / 0.17 / 0.11 / 0.04, against sampling's 0.215 / 0.118 / 0.068 / 0.037.
  - Selection by mass: 1.22 / 0.93 / 0.54 / 0.21.
  - The oracle costs about 80M forwards per network, and this is the anchor only.
- z31 (adaptive boxes, GL2 + centre): 1.46 / 1.63 / 0.37 points at 144 / 216 / 432 nodes, far worse than the tensor rule.
- z21 orders 9–10 (K = 256): 0.022 at 6.8M (4.0× MSE) and 0.011 at 9.3M (1.5×). The order 5–8 'rate −0.25' was a plateau; over orders 5–10 the rate is −0.50, the same as sampling's. So the anchor reaches parity near 10^7, but there the lower-input models (0.05) are far above sampling's 0.009.
- z31 at 864 nodes: 0.063 (sampling 0.010).
- z33 (patterns discovered along S-lines at the outer nodes, all kept, nearest-centroid fallback): with 1 / 8 / 27 lines, coverage is 31% / 59% / 73% and the error 2.9 / 1.67 / 1.28 (sampling 0.28 / 0.097 / 0.053). Worse than integrating the lines.
- z32 (2^17 × 64, noise 0.007): noise-subtracted RMS on the dev networks is PA_gap1 0.075 / PA_gap2 0.076 / PT_gap1 0.081 / PT_gap2 0.085. Adjacent triples don't help. The residual is spread between networks (SD about 0.07).
- z34 (ideal additive model: exact first-order ANOVA, K = 32 common backgrounds, 2^15 × 32): mean +0.063, SD 0.041, noise-subtracted RMS 0.074. This is the additive family's floor.
- z35 (least-violation fallback on the greedy sets): disagreement 2.2% / 1.6% / 0.95% at K = 10 / 30 / 100, worse than the centroid fallback (1.3% / 0.9% / 0.6%).
- z36 (joint product GL rule over the six lower inputs): RMS 0.23 at q = 2 (64 rows) and 0.15 at q = 3 (729 rows), worse than the pair model.
- z37 (other k): MF2 anchor bias 5.7 RMS, its integral 0.03 at 1.7k (sampling 0.67). MF3 anchor bias 0.68 RMS, its integral 0.127 / 0.074 / 0.073 at 24k / 55k / 98k (sampling 0.178 / 0.119 / 0.089), so it beats sampling for the anchor. The correction is the obstacle: per configuration, inside 720 arrangements, at least about 41k per node; first-order MF3 model 0.156.
- z38 (end-to-end deterministic estimate, MF3 outer × first-order correction at every configuration): 5.8 / 3.2 / 2.1 / 0.63 / 0.34 points at 84k / 334k / 668k / 3.0M / 10.5M forwards. Sampling gives 0.096 / 0.048 / 0.034 / 0.016 / 0.009.
