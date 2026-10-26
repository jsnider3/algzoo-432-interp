# Can a deterministic, mechanistic method match random sampling for the accuracy of tiny RNNs?

The attached zip holds **1,024 independently trained networks** of one small architecture: ARC's AlgZoo 2nd-argmax RNN, a 432-parameter ReLU RNN that outputs the position of the 2nd-largest of 10 Gaussian inputs. It also holds the code of a long research project on these networks, its write-up (`REPORT.md`), every script's printed output, and a per-network reference table for scoring.

This document stands alone. It defines the task, describes how the networks compute, lists every route measured so far with its numbers, names the directions that have not been tried, and ends with a prioritized list of work (§8).

**Where things stand.**
- The accuracy splits into two parts:
  - **The anchor**: a low-dimensional integral over the largest few inputs, with the others at their conditional mean.
  - **The correction**: the effect of the smaller inputs.
- Deterministic quadrature integrates the anchor better than sampling when 2 or 3 inputs are kept exact.
- Given the top 4 inputs, a model of the smaller inputs' effect gets the accuracy right to about 0.05 points (RMS across networks), at 240–410 forwards per top-4 configuration.
- But no deterministic method found so far combines the two cheaply. The best end-to-end deterministic estimator is 3–4 orders of magnitude worse than sampling in MSE.
- The crux is computing the correction deterministically: a 6–8-dimensional, discontinuous, non-local integral over the smaller inputs.

**What is wanted.** Either a method that meets the bar in §1 (or comes measurably closer), or the strongest argument that methods of a stated kind cannot. Please work through §8 in order, as far as time allows, and measure everything the way §8 describes.

## 1. The goal

ARC's estimation challenge: they would "consider it a major research breakthrough" to produce a **mechanistic** estimate, one that "reasons deductively based on the structure of the model" rather than from sampled examples, that "matches the performance of random sampling in terms of mean squared error versus compute". The comparison is averaged over independently trained networks, which is why the population exists.

- Plain sampling with C forward passes has MSE p(1−p)/C for a network of accuracy p.
- So at each compute level C, the estimate's **MSE averaged over the 1,024 networks** must be at most the average of p(1−p)/C. Equivalently, an estimate with error ε may cost at most about p(1−p)/ε² forward passes.
- Across the population p(1−p) averages about 0.069, so sampling reaches the following precision:

  | forwards per network | 10 | 100 | 700 | 7,800 | 70,000 | 780,000 | 10⁷ |
  |---|---|---|---|---|---|---|---|
  | sampling RMS error (points) | 8.3 | 2.6 | 1 | 0.3 | 0.1 | 0.03 | 0.008 |

- "Points" are percentage points of accuracy throughout.
- ARC's phrasing asks for the whole curve. A method on or below the line over part of the range (say 10²–10⁴ forwards) is still a result worth reporting, with the range stated.

## 2. Setup

- **Architecture** (AlgZoo `DistRNN`):
  - h_t = ReLU(W_ih·x_t + W_hh·h_{t−1}), with h_0 = 0 and no biases; logits = W_o·h_T; the answer is the argmax (ties go to the lowest index).
  - H = 16 and T = 10, so 16 + 256 + 160 = 432 parameters.
- **Task.** x ~ N(0, I₁₀), and the target is the position of the 2nd-largest input. Accuracy is P_x[argmax logits = target].
- **The population** (`pop1k/seed0.npz`).
  - 1,024 networks, trained independently with AlgZoo's recipe (`batch_train.py`) to 2³⁰ sequences.
  - The weights are the keys `s1048576_{Wih,Whh,Wo}`, with shapes (1024, 16), (1024, 16, 16) and (1024, 10, 16). The file also holds `init_*` (initializations) and `acc_2e20` (accuracy on 2²⁰ inputs).
  - Accuracy is 92.4 ± 3.4 points across networks (5th, 50th and 95th percentiles: 86.0, 93.2, 96.4).
- **Reference table** (`pop1k/reference.npz`, made by `z40.py`; printed summary in `pop1k/z40_seed0.txt`). It is meant for scoring without a GPU.
  - It gives, for all 1,024 networks plus the 6 zoo models, on the same 2²⁰ inputs:
    - the true accuracy `acc` (SE about 0.025 points);
    - the accuracies `mf2`, `mf3` and `mf4` of the mean-field anchors defined in §4;
    - their standard errors `se_acc` and `se_leak2/3/4`;
    - `names`.
  - Because the inputs are paired, each anchor's leakage (anchor minus truth) has an SE of 0.017–0.025 points.
  - The networks' errors are correlated, so one shared input set gives them a common offset. Here `acc` is 0.03 points below `acc_2e20` on average, and agrees with an independent 2²⁰ set (`z10`) to −0.009.
  - That noise is fine for scoring down to about 0.05 points (10⁵–10⁶ forwards). Below that, score against a larger independent reference.
  - Rows 0–1023 are the population. Rows 1024–1029 are the zoo, in the order mystery, seed0, seed1, seed2, seed3, seed4.
- **Out-of-population check.** ARC's six released models of the same architecture, trained by ARC with the same recipe: `weights.npz` (the "mystery" model) and `weights_seed0..4.npz`. Load them with `seeds_clocks.load`. Their accuracies are in the reference table.
- **Unit of cost.** One forward pass of the target network, counted as 2,880 multiply-adds.
  - Any other work on the target's weights (line tracing, region walking, root finding, polygon or cone integrals, propagating distributions) is converted at that rate.
  - **Work that never looks at the target's weights is free.** That includes studying the other 1,023 networks and designing rules from them, if the rules are then fixed.

## 3. How the networks compute

A long study of the mystery model is in `REPORT.md` §1–§3 and §7.9; the population is in §7.16 and `pop1k/notes_circuit_2e30.md`.

**3.1 The common design** (REPORT §1).
- The network runs an **online, time-invariant 2nd-max tracker**. It never stores *where* the max and 2nd max are. It stores *how long ago* they appeared.
- In the mystery model, the ages are stored as the phases of two damped rotating clocks that turn at the same speed (about 43° per step). Each clock's amplitude scales with the value it tracks.
- At each step, ReLU **comparator** neurons test the new input against thresholds that blend M (the running max) and S (the running 2nd max). Depending on the result:
  - **a new max** restarts the max clock and hands the old max clock's phase to the 2nd-max clock (the **handoff**);
  - **a new 2nd max** restarts the 2nd-max clock;
  - **anything else** lets both clocks keep rotating.
- **The more fundamental view.**
  - A sparse **write-pointer neuron** fires on, or one step after, each new max or new 2nd max. The clocks are the eigenmode breakdown of the recurrent weights' impulse response to that pulse.
  - In the mystery model (pointer n5), W_o·W_hh^a·e₅ decodes every age a correctly from the weights alone.
  - The readout is a Fourier-like decoder of the 2nd-max clock's phase, and it is the softmax-regression optimum for these features.
- **The other design.** Some networks are "memory" networks. They keep a decaying record of past inputs instead of a pointer, and the max is suppressed by an opposite-sign write.

**3.2 Variation across the population** (REPORT §7.16, `z2.py`–`z7.py`).
- **Two variants.**
  - About 54% of networks have a write pointer: 38% fire on the event and 17% one step late. The rest are memory networks.
  - The variant is fixed early in training and can be read from about 32 probe inputs. The two variants barely differ in where they err.
- **Neuron roles per network**, on average:
  - 6.0 always-on neurons (the clock and memory channels);
  - 3.1 comparators (96% of networks have at least one);
  - 0.9 max detectors (78% have one);
  - 5.3 others, mostly clock-like.
- **Where errors come from.**
  - One source is blur near ties (M ≈ S, or S ≈ R, the 3rd largest).
  - The other is failed handoffs, above all when the new max arrives at the last step.
  - Errors settle within about 3 steps after the last event.
- **Where networks differ.**
  - About 62% of the across-network variance in accuracy lies in the non-tie regions, where handoffs are the largest factor. The ties carry the other 37%.
  - A last-step handoff fails 12.3% of the time on average, with an SD of 13.5 across networks and a correlation of −0.81 with accuracy. It fails 38% of the time with no comparators and 3% with five or more.
  - Accuracy is 88.5–90.3 with 0–2 comparators and 94.1–94.9 with 4–6.

**3.3 Structural facts that matter for integration.**
- **Positive homogeneity.** With no biases, h(cx) = c·h(x) for c > 0. So every linear region of the input space is a polyhedral **cone** through the origin. The correct set is a union of cones, and the accuracy is a sum of Gaussian cone measures (solid angles).
- **Independence of ranks.** The inputs are iid, so each step's relative rank among the inputs so far is uniform and independent of the others. So the sequence of events (new max, new 2nd max, neither) has a product law. Also, given the top k values, the other 10 − k inputs are iid N(0, 1) truncated below the k-th largest.
- **Hidden-state Markov chain.** (h_t, M_t, S_t, position of S_t) is a Markov chain driven by iid inputs, and correctness is a function of its final state.
- **Within one activation pattern the network is linear.** The logits are Σ_t c_t x_t for that pattern's coefficients, so the margin to each rival is a linear form.
- **The analog channels switch off rarely, and that matters** (`z29.py`).
  - The most active neuron is on for 94.8% of steps (population median).
  - Treating the 7 neurons (median) that are on at least 90% of steps as linear biases the accuracy by 2.3 points RMS and still leaves about 230 linear regions per arrangement.
- **The logit margin is a small difference of large activities** (`z8.py`): about 1–4, against activities of 20–90.
  - The network cancels lower-input noise in that difference.
  - Moment closures with independent ReLU residuals destroy the cancellation, giving a margin SD 2–11× too large.
  - Hard gates fixed at the mean-field point keep it (SD ratio 0.85).
  - The delta method fails on gate flips (−0.82 ± 1.06 points).
- **How the smaller inputs interact** (`z15.py`–`z19.py`).
  - Nearby small inputs push the same clock or memory neuron across its threshold. Clock-like and always-on neurons hold 88% of these shared flips, against 72% of all neurons.
  - 62% of pair interactions' squared size comes from inputs 1–2 positions apart, and 64% from pairs that both precede the 2nd max.
  - In a typical draw one pair carries 72% of the interaction.

## 4. Decomposition and terms

- **Top-k configuration.** The positions and values of the k largest inputs.
  - The position pattern is an **arrangement**: 10·9 = 90 for k = 2, 720 for k = 3, and 5,040 for k = 4.
  - For k = 4 the ordered values are M > S > R > Q. The density is φ(M)φ(S)φ(R)φ(Q)Φ(Q)⁶, the same for every arrangement.
- **CDF coordinates for the outer integral** (exact order-statistic laws, used by `z21.py` and `z37.py`):
  - **k = 4:** Φ(Q) ~ Beta(7, 4); (Φ(R) − Φ(Q))/(1 − Φ(Q)) ~ Beta(1, 3); (Φ(M) − Φ(R))/(1 − Φ(R)) ~ Beta(2, 1); and Φ(S) is uniform between Φ(R) and Φ(M).
  - **k = 3:** Φ(R) ~ Beta(8, 3); (Φ(M) − Φ(R))/(1 − Φ(R)) ~ Beta(2, 1); and Φ(S) is uniform between.
  - **k = 2:** Φ(M) ~ Beta(10, 1); and Φ(S)/Φ(M) ~ Beta(9, 1).
- **S-line.** The 2nd-largest value varied over its range, with everything else fixed. The network is piecewise linear along it, so `lines.trace_line` integrates correctness along it exactly.
  - Tracing the S-line exactly for all 5,040 arrangements at one MF4 outer node (fixed Q, R and M) costs about 9,300 forwards. That uses shared recurrent prefixes across arrangements and sparse updates between adjacent pieces.
  - That is the charge per outer node used below. The optimized implementation behind the figure is not in the zip.
- **MFk anchor (mean field).** The top k inputs are kept, and every other input is set to its conditional mean below the k-th largest, μ(v) = −φ(v)/Φ(v). I₀ is the network's correctness there.
  - The **leakage** is anchor accuracy minus true accuracy, and the **correction** is minus that.
  - On all 1,024 networks (`z40.py`, paired, in points):

  | anchor | exact inputs | leakage mean | SD across networks | RMS | zoo (mystery, seeds 0–4) |
  |---|---|---|---|---|---|
  | MF2 | 2 | +4.81 | 2.14 | 5.26 | +3.1 +5.0 +6.4 +3.6 +4.2 +6.6 |
  | MF3 | 3 | +0.25 | 0.51 | 0.57 | +0.00 +0.09 +1.38 −0.00 −0.25 −0.10 |
  | MF4 | 4 | +0.31 | 0.29 | 0.42 | +0.16 +0.31 +0.75 +0.23 +0.09 +0.22 |

  - The leakage varies by network, so no constant offset fixes it, and network features predict MF3's leakage only to 0.39 points.
  - **Anatomy of MF3's leakage.** It lives where R > 0, as two mechanisms that nearly cancel:
    - the true network answering at the 4th-largest's position, worth +0.45 points, which MF3 cannot do;
    - S-vs-R swaps, netting −0.35.
  - Keeping the 4th largest exact (MF4) removes both, leaving a fairly uniform optimism.
- **Lower-input model.** An approximation of P(correct | top-k configuration) that accounts for the 10 − k smaller inputs. Integrating it over the top-k configurations gives an accuracy estimate.

## 5. What has been measured

Each script prints its results to `pop1k/z*_seed0*.txt`, and REPORT §7.16 discusses each one. Unless stated, "development" means every 16th population network and "held out" means every 16th starting from network 8.

### 5.1 Lower-input models for k = 4 (the 6 smaller inputs, given the top 4)

Scored against sampled true conditional accuracies; sampling is used only to measure.

| model | rows (forwards) per configuration | local MSE / p(1−p) | integrated error across networks: mean, RMS (points) |
|---|---|---|---|
| I₀ (MF4 anchor) | 1 | 0.17 | +0.31, 0.40 |
| first order in the indicator, clamped: P̂ = clip(I₀ + Σⱼ(Eⱼ − I₀), 0, 1), q = 16 midpoint nodes per input (`z11`, `z20`) | 97 | 0.12 | −0.015, 0.052 (all 1,024) |
| the same, Gauss–Legendre q = 8 | 49 | 0.12 | −0.040, 0.065 |
| the same, unclamped | 97 | 0.54 | +0.136, 0.160 |
| first order in the margin m = L_target − max_other L_o, with P(m > 0) an exact convolution over the node grid (`z13`, `z15`) | 49 | 0.011 | −0.17, 0.19 |
| margin + all 15 pair terms, projected onto one-input terms (`z18`) | 1,009 | 0.005 | +0.05, 0.062 |
| **margin + pairs at most 2 positions apart, projected (`z18`)** | **240–410** | **0.003** | **+0.03, 0.048 (all 1,024)** |
| + projected triples spanning up to 2 or 3 positions (`z32`, development) | more | | RMS 0.074–0.101, against 0.075–0.077 without |
| ideal first-order ANOVA projection of the margin (`z34`) | not computable cheaply | | +0.063, 0.074 |
| joint product Gauss–Legendre rule over all 6 inputs (`z36`) | 64 / 729 | | 0.23 / 0.15 |
| MF5 + first order in the indicator, q = 8 (`z11`) | 41 | | +0.037, 0.058 |
| MF3 + first order in the indicator, 7 inputs, q = 8 (`z11`) | 57 | | −0.000, 0.156 |

- **The clamped indicator model gets its total by cancellation.** It reverses I₀ on 2.0% of configurations, where the true conditional accuracy is about 0.55. Those carry 65% of its local squared error.
  - Mechanism: where I₀ steps from 0 to 1 and the Eⱼ are continuous, the raw sum drops by 5. So whenever 1 < ΣEⱼ < 5, P̂ jumps from 1 to 0.
- **The margin model's −0.18 bias is double counting** (`z15`–`z19`).
  - A gate flipped by two one-input runs is counted twice; in the joint run it flips once. Shared flips explain 79% of the net bias on draws with a sign error, and changes of winning rival 52% (the two overlap).
  - The projection hⱼ(z) = dⱼ(z) + Σₗ Eₗ[Dⱼₗ(z, Zₗ)] averages each input's effect over its neighbours' spread. That keeps the model additive, so P(m > 0) is still an exact convolution.
- **The additive family has a floor near 0.07 points.** On the development set, the ideal sum of one-input effects is itself 0.074 RMS off, and every truncated scheme lands at 0.075–0.085. The nearby-pair model's RMS is 0.073 on the development set, 0.042 held out and 0.048 on all 1,024.

### 5.2 The anchor integral by deterministic outer rules

**MF4 anchor, tensor Gauss–Jacobi in the k = 4 CDF coordinates** (`z21`).
- Each outer node is charged 9,300 forwards, and the S-integral is a converged grid.
- Errors are RMS over 16 networks (every 64th) + zoo, against a sampled reference of the same anchor (noise 0.003 points).

| order | outer nodes | forwards | RMS error (points) | sampling at that cost | MSE / sampling |
|---|---|---|---|---|---|
| 1 | 1 | 9,300 | 3.6 | 0.28 | 160 |
| 2 | 8 | 74k | 1.27 | 0.10 | 160 |
| 3 | 27 | 251k | 0.18 | 0.055 | 11 |
| 4 | 64 | 595k | 0.18 | 0.035 | 25 |
| 5 | 125 | 1.16M | 0.039 | 0.025 | 2.4 |
| 6 | 216 | 2.0M | 0.031 | 0.019 | 2.7 |
| 7 | 343 | 3.2M | 0.032 | 0.015 | 4.5 |
| 8 | 512 | 4.8M | 0.026 | 0.013 | 4.0 |
| 9 | 729 | 6.8M | 0.022 | 0.011 | 4.0 |
| 10 | 1,000 | 9.3M | 0.011 | 0.009 | 1.5 |

- Over orders 5–10 the error falls as nodes^(−0.5), the same rate as sampling, with a plateau over orders 5–9. So the rule runs parallel to sampling's line, 1.5–5× above it in MSE. It touches the line only near 10⁷ forwards, for the anchor alone.

**Other outer rules for the MF4 anchor are worse:**
- **Smolyak sparse grids** of the same Gauss–Jacobi rules (`z28`): 0.83, 0.32, 0.38 and 0.46 points at levels 5–8 (260k–4M forwards). They do not converge, because the negative weights amplify kinks.
- **An adaptive box rule** with 2×2×2 Gauss–Legendre per box (`z31`): 0.063 points at 864 nodes, 40× sampling's MSE.
- **Homogeneous coordinates** (`z23`, `z27`).
  - The method: integrate the overall scale out exactly. Per arrangement the density becomes (12/π⁵)(1 + |z|²)⁻⁵ on the unit-norm retained values and lower inputs. Within a region every gate is A + B cos θ + C sin θ, so the angle integrates by elementary roots, leaving a 2-D outer integral.
  - The result: errors of 0.68, 0.22, 0.34, 0.12 and 0.46 points at orders 4–8 (`z23`), erratic, 3–13× sampling's RMS over 10⁵–1.3M forwards.
  - Splitting the domain into 9 smooth pieces (`z27`) gives 0.176, 0.041, 0.052 and 0.035 at orders 4–7 (1.3M–4.1M forwards). Over the last three orders the error falls as nodes^(−0.19). What remains is the network's own non-smoothness.

**MF3 anchor and the full MF3 estimator, k = 3 coordinates.**

| method | forwards | RMS error | sampling at that cost |
|---|---|---|---|
| MF3 anchor, tensor Gauss–Jacobi, against the anchor (`z37`, 16 networks + zoo) | 14k / 24k / 55k / 98k | 0.31 / 0.13 / 0.074 / 0.073 | 0.24 / 0.18 / 0.12 / 0.089 |
| MF2 anchor, against the anchor (`z37`) | 573 / 1.7k | 0.16 / 0.03 | 1.16 / 0.67 |
| MF3 exact S-lines with shared prefixes, outer order 3 (all 1,024, against the true accuracy) | 36k (19k–67k) | 0.58 | 0.14 |
| MF3, logits interpolated along S with exact roots, against the truth | 669 / 1,029 / 7.0k / 51k / 97k | 8.9 / 2.3 / 1.17 / 0.63 / 0.56 | 1.0 / 0.82 / 0.31 / 0.12 / 0.084 |

- **The only in-range wins are integrals of anchors, not accuracies.** The MF3 anchor integral is 1.5–2.6× better than sampling in MSE between 2×10⁴ and 10⁵ forwards, and MF2's far better at 10³.
- **Prefix sharing** saves 2–3× (MF3 per slice: 2,692 → 1,871 → 1,528 forwards; MF4: 17,840 → 11,292 → 9,265).
- **The exact MF3 region walk** (`z9`, CPU) converges to the MF3 value (mystery 0.953213, to 2e-7) at 5–45M forwards. 72–80% of the cost is R < 0, where regions are many.

### 5.3 The arrangements cannot be thinned (`z24`)

- Across the 5,040 arrangements the MF4 anchor accuracy has an SD of 8 points.
- The positions of M and S explain 28% of it, and 41% once the order of R and Q relative to them is added.
- **The dependence on positions does not factor.**
  - Additive position effects explain 18% of the variance across arrangements.
  - All six pairwise position interactions (463 parameters) explain 48%.
- **Fixed subsets** of the 56 (R, Q) placements per (M, S) class are 4–10× worse in RMS than sampling at the reduced cost. With 1 placement (90 arrangements) the error is 1.2–2.1 points, against sampling's 0.19; with 8 placements, 0.27 against 0.067.

### 5.4 Closed forms from linear regions

One exact linear map per activation pattern costs about one forward to build.

- **How many regions there are** (`z26`).
  - One arrangement's 4-D value space, with the lower inputs at μ(Q), holds a median of 560 regions seen in 2¹⁶ draws (190–870 across networks), and the count is still rising.
  - The most common region carries 20% of the mass, the top 10 carry 66% and the top 100 carry 95%.
  - The uncovered mass at K = 1, 10, 30, 100, 300 and 1,000 most probable patterns is 80%, 34%, 15%, 5.0%, 1.3% and 0.29%. A Good–Turing estimate puts the never-seen mass at 0.23%.
- **Patterns taken at K grid points per arrangement** (`z25`; nearest-map fallback):

  | patterns per arrangement | forwards | mass covered | error (points) | sampling |
  |---|---|---|---|---|
  | 1 | 5k | 17% | 2.7 | 0.37 |
  | 16 | 81k | 50% | 1.2 | 0.093 |
  | 81 | 408k | 67% | 1.0 | 0.041 |
  | 256 | 1.3M | 76% | 0.76 | 0.023 |

- **Patterns chosen by an oracle** (`z30`). The patterns are chosen on 2¹⁴ training draws per arrangement, and scored on held-out draws:

  | K | build cost | error, K most massive | error, greedy for agreement | sampling |
  |---|---|---|---|---|
  | 3 | 15k | 1.22 | 0.33 | 0.215 |
  | 10 | 50k | 0.93 | 0.17 | 0.118 |
  | 30 | 151k | 0.54 | 0.11 | 0.068 |
  | 100 | 504k | 0.21 | 0.04 | 0.037 |

  This is the closest any deterministic form came: 1.4–1.6× sampling's RMS. But the oracle itself costs about 80M forwards per network, and this is the anchor only.
- **Patterns discovered along S-lines at the outer nodes** (`z33`): 2.9, 1.67 and 1.28 points at 9k, 74k and 250k forwards. That is 13–24× sampling's RMS.
- **A least-violated-region fallback** (`z35`) is worse than nearest centroid.
- **Clock-state linearization** (`z29`) is in §3.3: a 2.3-point bias, with about 230 regions left.

### 5.5 The correction (anchor → accuracy)

- **What the k = 4 integrand looks like** (`z12`).
  - I₀ is a 0/1 indicator with variance 0.97·p(1−p) over configurations. P̂ is barely smoother (0.79·p(1−p)).
  - The penalty P̂ − I₀ is nonzero on 12% of configurations (7–19% across networks). 80% of its square lies where the MF4 logit margin is below 0.5 (6% of configurations).
  - **The 5,040 arrangement means explain 6–8% of the variance of I₀ and P̂, and only 1% of the penalty's.** The rest of each is within arrangements, in the 4 values.
- **It is not local to the anchor's decision boundary** (`z22`; 1,024 sampled S-lines).
  - Of the first-order margin model's line penalty (−0.54 points), 68% sits on S-lines where the anchor never changes its answer: rare large lower-input effects flip answers the anchor gets confidently right.
  - A two-term boundary formula, a mean shift plus smearing, gets the sign wrong (+0.78). Freezing the lower-input law at each crossing recovers only −0.19.
- **Cost inside the enumeration.** About 240 forwards (the nearby-pair model) × about 4 S-points × 5,040 arrangement lines comes to about 4.8M forwards per outer node. For MF3 it is at least 720 × (S-points) × (rows), about 41k per node even with the cheapest first-order model, which is itself 0.156 off.
- **End to end** (`z38`): the MF3 outer rule × K S-points × 720 arrangements, with the clamped first-order model over the 7 smaller inputs at every configuration. No sampling anywhere.

  | forwards | 84k | 334k | 668k | 3.0M | 10.5M |
  |---|---|---|---|---|---|
  | RMS error against the truth (points) | 5.8 | 3.2 | 2.1 | 0.63 | 0.34 |
  | sampling | 0.096 | 0.048 | 0.034 | 0.016 | 0.009 |

### 5.6 Other frames

- **Events** (`z39`).
  - The anchor: keep every record input exact (each new max or new 2nd max, 4.9 per sequence on average), and put each "neither" input at its conditional mean below the running 2nd max at that time.
  - Its error is +1.9 points (RMS 2.0). Keeping the last one or two "neither" inputs exact as well gives RMS 1.4 and 1.0.
  - Conditioning the smaller inputs on the final top 4 loses less than conditioning them on the running state.
- **Ages of the max and 2nd max** is the top-2 frame (MF2, 5.3 points off).

### 5.7 Routes that sample or fit (excluded by the constraints, listed for completeness)

- Population-informed strata or peer-margin importance weighting: about 2× sampling (`z1`).
- Exact integration along the S-line or the last input, averaged over sampled inputs: 1.2–4× net (`v1`).
- Ordered Sobol designs: 1.4–3.0×.
- Mechanism-defined adaptive strata: 1.4×, with a ceiling of 1.2–5× even with perfect stratum knowledge.
- Predicting accuracy from neuron roles or weights: 1.0-point CV RMSE at best, worth about 700 samples (`z2`, `z3`).
- On the zoo models only (before the population existed):
  - A boundary-aware quadrature over the top 3 with prefix caching had an honest randomized-replicate SE 6–25× sampling's (REPORT §7.13).
  - A weight-derived Gaussian blur of the mean-field boundary converges to −2.6 points (§7.15).

## 6. Why it is hard, and what a proof would need

**Diagnosis:**
- The integrand is a 0/1 indicator in 10 dimensions. Every decomposition leaves a discontinuous integrand somewhere.
  - Even after the S-line is integrated exactly, the measured outer rules converge at sampling's rate at best (`z21`), or slower (`z27`, `z28`, `z31`).
- The difficulty is spread out rather than concentrated:
  - between arrangements (SD 8 points, non-factoring);
  - within each arrangement's value space (hundreds of linear regions);
  - and off the boundary, in the correction (68%).
- Making fewer inputs exact makes the anchor integral cheap but moves the difficulty into the correction. Making more inputs exact does the reverse.

**What a proof would need:**
- Any computation on the weights can, given enough compute, find the accuracy exactly; `z9` does it for MF3. So the question is only cost, at one fixed size.
- **A general impossibility proof seems out of reach.** A white-box lower bound at a fixed size would be a concrete circuit lower bound for an explicit function, and even superlinear circuit lower bounds for explicit functions are open.
- **In the black-box (membership-oracle) model, deterministic volume estimation is provably weaker than randomized** (Elekes 1986; Bárány–Füredi 1987; against Dyer–Frieze–Kannan 1991). The networks are white-box, though.
- **What could still be proved: a bound for a restricted but meaningful class**, with constants tied to measured quantities. Examples:
  - an information-based-complexity bound for deterministic quadrature on a function class that provably contains these integrands, such as indicators of unions of K polyhedral cones in d dimensions, or such indicators composed with the order-statistic maps;
  - a bound for methods that evaluate the network only on piecewise-linear paths, or only through region enumeration, in terms of region counts and mass tails (`z26`);
  - a bound in terms of the correction's non-locality (`z22`).

## 7. Directions not yet tried, and the clue behind each

**(a) Amortize the correction across arrangements.**
- The clue: the arrangement means carry 1% of the penalty's variance, against 6–8% of the anchor's (`z12`). The anchor depends on the exact positions (§5.3), but the correction may not.
- If the per-arrangement correction integral is nearly the same for whole classes of arrangements, a deterministic rule could compute it on a few designed arrangements per class. The classes would be fixed in advance from the population, for example by the positions of M and S only, or by the ages. The anchor would still be done on all of them.
- First measure the SD across arrangements of the per-arrangement correction (true minus MF4 anchor, or minus MF3) against the precision needed (about 0.05–0.1 points overall).

**(b) Two levels.**
- Integrate the cheap anchor (MF3, or MF2) finely, and the difference between a good correction model and the anchor on a much coarser outer rule.
- `z38` used one grid for both. The open question is how coarse the correction's rule can be for a given error.

**(c) Propagate the hidden state's distribution forward in time.**
- The state (h_t, M_t, S_t, position of S_t) is a Markov chain with iid inputs (§3.3), so the accuracy is a deterministic recursion over time on a mixture of components. Each new input is integrated exactly over its three event branches (x > M, S < x < M, x < S) with Gauss rules on the truncated normal.
- **The warning:** `z8` and `z39` show that independent closures and mean fields lose the network's cancellations.
- **The suggestion:** carry each component's state as an affine function of the past "neither" inputs within a fixed activation pattern, so its law stays exact while the pattern holds. Split a component only where a gate can flip with non-negligible mass, and merge by event history and ages.
- This does what the region walk does, but in time order, sharing work across all arrangements with the same history. The ages structure (§3.1) suggests the number of live components may stay small.

**(d) Geometry of the error set.**
- The incorrect set has only 7.6% of the mass. It is a union of polyhedral cones (§3.3), possibly far fewer than the regions.
- Count its maximal convex pieces per network. Compute their Gaussian measures deterministically: solid-angle series, or orthant-probability recursions in the effective dimension.

**(e) The low-budget end.**
- At 10–10³ forwards sampling's RMS is 8–1 points. Is any deterministic weight-based computation on or below the line there? Candidates: a few-node MF2 or MF3 rule, or a coarse version of (c).
- The constant population mean is excluded as a fitted prediction, but a mechanistic computation of similar accuracy is not.

**(f) Anything better.** A description of the network in which both the integral and the correction are cheap. If proposing a new frame, please first say what its anchor's bias is, then what its integral costs, as in §5.6.

## 8. Work requested, in priority order

1. **Audit (keep this short).**
   - Check the scoring and cost accounting of `z18`, `z21`, `z37` and `z38` for errors that would move a conclusion:
     - the charge per outer node;
     - the sampling comparator √(mean p(1−p)/C);
     - the adequacy of the references;
     - double counting of rows.
   - An external audit has found real bugs in this code before: a tie rule in `lines.trace_line`, and an ignored overflow flag in `z11.py`. Both are fixed.
2. **The correction: try (a)–(e), or better.**
   - For each direction:
     - give inline code;
     - give its cost in forwards;
     - give its RMS error across the development networks and the held-out networks (against `pop1k/reference.npz`), next to sampling's √(mean p(1−p)/C) at that cost and the MSE ratio;
     - give the six zoo errors.
   - Stop a direction once it is clearly off the line, and say why, with the measurement that shows it.
   - A partial result is welcome, for example a deterministic correction for MF3 or MF4 that costs at most 10× the anchor integral and is accurate to 0.1 points.
3. **The best end-to-end deterministic curve.** Combine the best anchor rule and the best correction into one estimator, and report RMS error against forwards from 10² to 10⁷ (or as far as feasible). Compare it with `z38`.
4. **A lower-bound argument** of the kind in §6, as strong and as concrete as possible.
5. **An honest verdict**, and the single experiment most likely to change it.

**Reporting.** For each method, a table with these columns:
- forwards per network;
- RMS error across networks;
- mean error;
- sampling RMS at that cost;
- MSE ratio;
- zoo errors.

Develop on the development set, and report the held-out set without retuning.

## 9. Constraints

- **Mechanism, not statistics.** The estimate for a network must be a deterministic computation on that network's own weights.
  - Not allowed:
    - random or quasi-random point sets over the input or configuration space, including Sobol or lattice rules used in place of sampling;
    - control variates, and importance or stratified sampling (so sampling the correction on top of a deterministic anchor is excluded);
    - any fitted prediction of accuracy or error from the population, such as a regression or a calibration of the final number.
  - Allowed:
    - points the method chooses from the weights (region vertices, line pieces, roots, quadrature nodes);
    - design choices learned from other networks in the population (grids, arrangement classes, which pairs to keep), if they are fixed before the method is applied to a network.
- **Random sampling is fine for measuring**: scoring an estimate against the truth, or describing an integrand.
- **Compute.**
  - Develop on CPU on a subset if needed; `pop1k/reference.npz` makes scoring free.
  - We will rerun everything on an RTX 2060 SUPER (8 GB) with 8 CPU cores, so GPU code (torch) is welcome.
- **Please paste all code inline.** Sandbox files can expire before we download them.

## 10. Building blocks

- `batch_train.forward(Wih, Whh, Wo, x)`: logits (N, B, 10) for x of shape (N, B, 10), batched over N networks. `batch_train.target(x)` gives the target positions.
- `z2.load('pop1k/seed0.npz')` returns ([Wih, Whh, Wo] as float32 torch tensors, step).
- `seeds_clocks.load(path)` returns a zoo model's (Wih, Whh, Wo) as NumPy arrays.
- `common.py` is a NumPy forward pass (`run`), exact to about 1e-12, with an exact Jacobian (`run_jac`).
- `z10.mu_below(v)` gives E[x | x < v]. `z10.topk_mf(x, k)` builds the MFk input.
- `lines.trace_line(Wih, Whh, Wo, x, p, lo, hi, tgt)` (numba) returns P(correct | x_p ~ N(0, 1) truncated to (lo, hi)) exactly, plus the cost and an overflow flag.
- `z11.expansion(x, k, q)` builds the MFk point plus each lower input alone at q equal-probability nodes.
- `z9.Slice` is the exact region walk on an MF3 slice (CPU).
- **Copy these rather than import them.** `z18.py`, `z21.py` and `z37.py` are scripts with no `__main__` guard: importing one runs its whole experiment, which needs CUDA.
  - `z18.margin(L, y)` computes the margin.
  - `z18.conv_p(m0, h)` computes the exact P(m > 0) for an additive margin over 6 inputs on a node grid. It uses the module's node weights `w2`.
  - `z21.beta_rule(n, a, b)` and `z37.rule(n, a, b)` give Gauss–Jacobi rules for Beta(a, b).
  - `z21.G` and `z37.G` evaluate the outer integrand: all arrangements × S-points.

## 11. Files in the zip

The zip holds the whole repository, minus the git history, a training checkpoint, some archives and some large regenerable outputs.

- **Population:**
  - `pop1k/seed0.npz` (weights, `acc_2e20`), and `pop1k/reference.npz` (§2).
  - `pop1k/seed0.roles.npz` and `pop1k/seed0.buckets.npz`: per-neuron features and roles.
  - `pop1k/z*_seed0*.txt`: every script's printed output.
  - `pop1k/notes_circuit_2e30.md`: the population numbers in detail.
- **Write-up:** `REPORT.md`. §7.16 has everything in §4–§6 above, with more detail. §1–§3 and §7.9 cover the mechanism, §7.13 and §7.15 the earlier estimation attempts on the zoo, and §9 is a ledger of every idea tried. `README.md` has setup and a script index.
- **Scripts, by topic:**
  - **population and roles:** `batch_train`, `z1`–`z7`;
  - **lower inputs:** `z8`, `z10`, `z11`, `lines`, `lw2`, `z13`–`z20`, `z32`, `z34`, `z36`;
  - **the integrand:** `z12`;
  - **outer rules:** `z21`, `z23`, `z27`, `z28`, `z31`, `z37`;
  - **locality and arrangements:** `z22`, `z24`;
  - **closed forms:** `z25`, `z26`, `z29`, `z30`, `z33`, `z35`;
  - **end to end:** `z38`;
  - **events:** `z39`;
  - **reference table:** `z40`;
  - **exact region walk:** `z9`;
  - **sampling-based, for comparison:** `v1`.
- **Zoo:** `weights.npz` (mystery) and `weights_seed0..4.npz`.
- **Background only:** the `QUESTION*_FOR_CHATGPT_PRO.md` files and `chatgpt2/`, from earlier phases of the project. `REPORT.md` supersedes them where they differ.
