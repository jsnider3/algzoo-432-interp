# Estimating the accuracy of 1,024 tiny RNNs faster than sampling: check every method, then find better ones

The attached zip holds **1,024 independently trained networks** of one small architecture: ARC's AlgZoo 2nd-argmax RNN, a 432-parameter ReLU RNN that outputs the position of the 2nd-largest of 10 Gaussian inputs. It also holds:
- the code of a long research project on these networks;
- its write-up (`REPORT.md`);
- every script's printed output;
- a per-network reference table for scoring;
- precomputed training data for a learned estimator.

This document stands alone. It defines the task, describes how the networks compute and lists every route measured so far with its numbers. It then asks for two things (§8):
- a check of whether each current method could be done better, and whether its numbers are right;
- an exploration of every new approach that might work.

**Where things stand.**
- **The bar.** Sampling a network's accuracy with C forward passes has MSE p(1−p)/C. The goal is a computation on the network's weights that does at least as well at each budget C, averaged over networks (§1).
- **One method comes close at the cheap end: a correction learned from the population** (§5.9).
  - The configuration space of the largest inputs is cut into 720 equally likely cells.
  - Each cell gets 2 deterministic forward rows.
  - A small neural network, trained on the population, maps those rows plus free statistics of the weights to each cell's accuracy.
  - With every cost counted (1,440 forward rows plus the model's own arithmetic, 1,548–1,780 forward-equivalents in all), its MSE is 1.10–1.13× sampling's on the 1,024 networks (8-fold cross-fitting).
  - Counting forward rows only, a larger model reaches 0.80–0.85×, but its inference costs about 6,450 forward-equivalents, which puts it at 4.7×.
- **It falls behind as the budget grows.** In forward rows alone (larger model) it is 1.3× sampling's MSE at 2,880 rows, 1.7× at 5,760, 2.1× at 11.5k, and 16× at 92k with finer cells. Every other method measured is further from the line (§5.10).
- **ARC's own six models transfer worse,** at 1.8–3.2× in forward rows alone. The "mystery" model is consistently 1.4–1.9 points high.
- **Why it falls behind.** Its remaining error is shared across all of a network's cells. Each network has its own rate at which the smaller inputs break answers the mean-field anchor gets right, and two anchor rows per cell do not reveal it.
- **The best purely deterministic estimator without fitting,** an exact anchor plus a coarse correction, is 170–220× sampling's MSE at 2.5M forwards (§5.6).

**What is wanted.**
1. A check of every method in §5: is there a better way to do it, and are its numbers right?
2. Every new approach worth trying, above all ones that reach the sampling line with all costs counted, extend it to larger budgets and carry over to ARC's six models.

Please work through §8 in order, as far as time allows, and measure everything as §8 describes.

## 1. The goal

ARC's estimation challenge: they would "consider it a major research breakthrough" to produce a **mechanistic** estimate, one that "reasons deductively based on the structure of the model" rather than from sampled examples, that "matches the performance of random sampling in terms of mean squared error versus compute". The comparison is averaged over independently trained networks, which is why the population exists.

- Plain sampling with C forward passes has MSE p(1−p)/C for a network of accuracy p.
- So at each compute level C, the estimate's **MSE averaged over the networks** must be at most the average of p(1−p)/C. Equivalently, an estimate with error ε may cost at most about p(1−p)/ε² forward passes.
- Across the population p(1−p) averages 0.069, so sampling reaches the following precision:

  | forwards per network | 10 | 100 | 420 | 1,440 | 7,800 | 70,000 | 780,000 | 10⁷ |
  |---|---|---|---|---|---|---|---|---|
  | sampling RMS error (points) | 8.3 | 2.6 | 1.28 | 0.69 | 0.3 | 0.1 | 0.03 | 0.008 |

- "Points" are percentage points of accuracy throughout.
- ARC's phrasing asks for the whole curve. A method on or below the line over part of the range is still a result worth reporting, with the range stated.

**What counts here.**
- **On the network being estimated**, the estimate must be a deterministic computation on that network's own weights.
  - It may evaluate the network at points it chooses: fixed designs, quadrature nodes, or points chosen from the weights.
  - It may also trace lines, walk linear regions, propagate distributions and compute statistics of the weights.
  - It may not use random or quasi-random samples of that network's inputs, or sampling-based variance reduction such as control variates or importance or stratified sampling. Sampling is the baseline being competed against.
- **On the population, anything goes.** Fitting, learning, calibrating, and sampling the other networks as much as needed are all allowed and encouraged; that is what the population is for.
  - A model or design learned from the population must be fixed before it is applied to a network that was not used to fit it.
  - This offline work is not charged to the estimate, but please report it.
- **Generalization is part of the score.**
  - Report error on networks not used in fitting: cross-fitting over the 1,024, or the development / held-out split of §5.
  - Report it also on ARC's six models, which were trained by ARC, not here.
- **Mechanism is welcome wherever possible.** When a learned component helps, please say what it has learned in terms of how the networks compute (§3).

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
  - It covers all 1,024 networks plus the 6 zoo models, on the same 2²⁰ inputs. It gives:
    - the true accuracy `acc`, with SE about 0.025 points;
    - the accuracies `mf2`, `mf3` and `mf4` of the mean-field anchors defined in §4;
    - their standard errors `se_acc` and `se_leak2/3/4`;
    - `names`.
  - Because the inputs are paired, each anchor's leakage (anchor minus truth) has an SE of 0.017–0.025 points.
  - The networks' errors are correlated, so one shared input set gives them a common offset. Here `acc` is 0.03 points below `acc_2e20` on average, and agrees with an independent 2²⁰ set (`z10`) to −0.009.
  - That noise is fine for scoring down to about 0.05 points (10⁵–10⁶ forwards). Below that, score against a larger independent reference.
  - Rows 0–1023 are the population. Rows 1024–1029 are the zoo, in the order mystery, seed0, seed1, seed2, seed3, seed4.
- **Out-of-population check.**
  - These are ARC's six released models of the same architecture, trained by ARC with the same recipe: `weights.npz` (the "mystery" model) and `weights_seed0..4.npz`.
  - Load them with `seeds_clocks.load`. Their accuracies are in the reference table.
- **Unit of cost.** One forward pass of the target network, counted as 2,880 multiply-adds.
  - **All other arithmetic on the target network is converted at that rate.** That includes:
    - line tracing, region walking, root finding, and polygon or cone integrals;
    - propagating distributions;
    - evaluating a learned model on the target network's outputs.
  - **Work that never looks at the target network is free.** That includes all study, fitting and sampling of the other networks, and anything precomputed per cell or per design point that does not depend on the target.
  - Please report forward rows and other arithmetic separately, then their total.

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
- **How the smaller inputs change the answer** (`z42.py`).
  - Relative to the MF4 anchor (§4), the smaller inputs act as two opposing flows:
    - they break answers the anchor gets right (−2.3 points on average);
    - they fix answers it gets wrong (+1.9).
  - The net is −0.4.
  - Very negative smaller inputs almost only break (−0.50 against +0.08). Smaller inputs near the 4th largest do both.

## 4. Decomposition and terms

- **Top-k configuration.** The positions and values of the k largest inputs.
  - The position pattern is an **arrangement**: 10·9 = 90 for k = 2, 720 for k = 3, and 5,040 for k = 4.
  - For k = 4 the ordered values are M > S > R > Q. The density is φ(M)φ(S)φ(R)φ(Q)Φ(Q)⁶, the same for every arrangement.
- **CDF coordinates for the outer integral** (exact order-statistic laws, used by `z21.py`, `z37.py`, `z41.py` and `z46.py`):
  - **k = 4:** Φ(Q) ~ Beta(7, 4); (Φ(R) − Φ(Q))/(1 − Φ(Q)) ~ Beta(1, 3); (Φ(M) − Φ(R))/(1 − Φ(R)) ~ Beta(2, 1); and Φ(S) is uniform between Φ(R) and Φ(M). The three Beta coordinates are independent.
  - **k = 3:** Φ(R) ~ Beta(8, 3); (Φ(M) − Φ(R))/(1 − Φ(R)) ~ Beta(2, 1); and Φ(S) is uniform between.
  - **k = 2:** Φ(M) ~ Beta(10, 1); and Φ(S)/Φ(M) ~ Beta(9, 1).
- **S-line.** The 2nd-largest value varied over its range, with everything else fixed. The network is piecewise linear along it, so `lines.trace_line` integrates correctness along it exactly.
  - Its measured cost is about 6.3 forwards per line: 4,566 per MF3 outer node of 720 lines (`z41`).
  - That counts only the matrix multiply-adds. Root finding, sorting and CDF evaluations are extra.
- **A cost caveat.** Several scripts (`z21`, `z23`, `z24`, `z27`, `z28`, `z31`, `z37`) charge 9,300 forwards per MF4 outer node and 1,528 per MF3 node. Those figures come from an optimized prefix tree (shared recurrent prefixes across arrangements, sparse updates between adjacent pieces) that was described but never reproduced here. Its code is not in the zip.
- **MFk anchor (mean field).** The top k inputs are kept, and every other input is set to its conditional mean below the k-th largest, μ(v) = −φ(v)/Φ(v). I_k is the network's correctness there.
  - The **leakage** is anchor accuracy minus true accuracy, and the **correction** is minus that.
  - On all 1,024 networks (`z40.py`, paired, in points):

  | anchor | exact inputs | leakage mean | SD across networks | RMS | zoo (mystery, seeds 0–4) |
  |---|---|---|---|---|---|
  | MF2 | 2 | +4.81 | 2.14 | 5.26 | +3.1 +5.0 +6.4 +3.6 +4.2 +6.6 |
  | MF3 | 3 | +0.25 | 0.51 | 0.57 | +0.00 +0.09 +1.38 −0.00 −0.25 −0.10 |
  | MF4 | 4 | +0.31 | 0.29 | 0.42 | +0.16 +0.31 +0.75 +0.23 +0.09 +0.22 |

  - **Anatomy of MF3's leakage.** It lives where R > 0, as two mechanisms that nearly cancel:
    - the true network answering at the 4th-largest's position, worth +0.45 points, which MF3 cannot do;
    - S-vs-R swaps, netting −0.35.
  - Keeping the 4th largest exact (MF4) removes both, leaving a fairly uniform optimism.
- **Lower-input model.** An approximation of P(correct | top-k configuration) that accounts for the 10 − k smaller inputs. Integrating it over the top-k configurations gives an accuracy estimate.
- **Clamped first-order model.** F₄ = clip(I₄ + Σⱼ(Eⱼ − I₄), 0, 1), where Eⱼ integrates lower input j alone over q Gauss–Legendre nodes in its truncated-normal CDF coordinate.

## 5. What has been measured

Each script prints its results to `pop1k/z*_seed0*.txt`, and REPORT §7.16 discusses each one. Unless stated:
- "development" means every 16th population network (0, 16, …), and "held out" means every 16th starting from network 8;
- "cross-fitted" means 8 folds by network index mod 8;
- MSE ratios are against sampling at the stated cost.

### 5.1 Lower-input models for k = 4 (the 6 smaller inputs, given the top 4)

Scored against sampled true conditional accuracies; sampling is used only to measure.

| model | rows (forwards) per configuration | local MSE / p(1−p) | integrated error across networks: mean, RMS (points) |
|---|---|---|---|
| I₄ (MF4 anchor) | 1 | 0.17 | +0.31, 0.40 |
| clamped first order, q = 16 midpoint nodes (`z11`, `z20`) | 97 | 0.12 | −0.015, 0.052 (all 1,024) |
| the same, Gauss–Legendre q = 8 | 49 | 0.12 | −0.040, 0.065 |
| the same, unclamped | 97 | 0.54 | +0.136, 0.160 |
| first order in the margin m = L_target − max_other L_o, with P(m > 0) an exact convolution over the node grid (`z13`, `z15`) | 49 | 0.011 | −0.17, 0.19 |
| margin + all 15 pair terms, projected onto one-input terms (`z18`) | 1,009 | 0.005 | +0.05, 0.062 |
| margin + pairs at most 2 positions apart, projected (`z18`) | 241–412 (mean 241 for adjacent pairs, 412 within 2) | 0.003 | +0.03, 0.048 (all 1,024; raw 0.064 with 0.043 measurement noise) |
| + projected triples spanning up to 2 or 3 positions (`z32`, development) | more | | RMS 0.074–0.101, against 0.075–0.077 without |
| ideal first-order ANOVA projection of the margin (`z34`) | not computable cheaply | | +0.063, 0.074 |
| joint product Gauss–Legendre rule over all 6 inputs (`z36`) | 64 / 729 | | 0.23 / 0.15 |
| MF5 + clamped first order, q = 8 (`z11`) | 41 | | +0.037, 0.058 |
| MF3 + clamped first order, 7 inputs, q = 8 (`z11`) | 57 | | −0.000, 0.156 |

- **The clamped indicator model gets its total by cancellation.** It reverses I₄ on 2.0% of configurations, where the true conditional accuracy is about 0.55. Those carry 65% of its local squared error.
  - Mechanism: where I₄ steps from 0 to 1 and the Eⱼ are continuous, the raw sum drops by 5. So whenever 1 < ΣEⱼ < 5, P̂ jumps from 1 to 0.
- **The margin model's −0.18 bias is double counting** (`z15`–`z19`).
  - A gate flipped by two one-input runs is counted twice; in the joint run it flips once. Shared flips explain 79% of the net bias on draws with a sign error, and changes of winning rival 52% (the two overlap).
  - The projection hⱼ(z) = dⱼ(z) + Σₗ Eₗ[Dⱼₗ(z, Zₗ)] averages each input's effect over its neighbours' spread. That keeps the model additive, so P(m > 0) is still an exact convolution.
- **z18's costs leave out its convolution.** The convolution is about 4q⁴ scalar operations: roughly 3 forward-equivalents at q = 8 and 46 at q = 16. z18 also evaluates all pairs before masking, so the reduced costs assume the unused rows are skipped.
- **The 0.074 is a floor only for models that converge to the ideal projection, not for every additive model.**
  - Counterexample: m = |X+Y| − |X−Y| + 2|Z|. Its projection is 2|Z|, so P(m_A > 0) = 1, while P(m > 0) = 5/6.
  - The additive g = X + √3|Z| gets 5/6 exactly.

### 5.2 The anchor integral by deterministic outer rules

All costs in this subsection use the unreproduced charges of §4: 9,300 forwards per MF4 node, 1,528 per MF3 node and 191 per MF2 node. At the measured exact-tracer cost, MF3 nodes cost 3× more. MF2's lower inputs move with μ(S), so its S path is curved and the affine-line charge does not apply to it.

**MF4 anchor, tensor Gauss–Jacobi in the k = 4 CDF coordinates** (`z21`). Errors are RMS over 16 networks (every 64th) + zoo, against a sampled reference of the same anchor. That reference's noise is 0.014 points for orders 4–6 at K = 64 and K = 256, and 0.003 at K = 512 and for orders 9–10.

| order | outer nodes | forwards (charged) | RMS error (points) | sampling at that cost | MSE / sampling |
|---|---|---|---|---|---|
| 1 | 1 | 9,300 | 3.6 | 0.28 | 160 |
| 2 | 8 | 74k | 1.27 | 0.10 | 160 |
| 3 | 27 | 251k | 0.18 | 0.055 | 11 |
| 4 | 64 | 595k | 0.18 | 0.035 | 25 |
| 5 | 125 | 1.16M | 0.039 | 0.025 | 2.4 |
| 6 | 216 | 2.0M | 0.031 | 0.019 | 2.7 |
| 10 | 1,000 | 9.3M | 0.011 | 0.009 | 1.5 |

- **Other outer rules are worse:**
  - Smolyak sparse grids do not converge (`z28`).
  - An adaptive box rule is 40× sampling's MSE (`z31`).
  - Homogeneous coordinates are erratic (`z23`), and fall only as nodes^(−0.19) even with their kinks removed (`z27`).
- **The MF3 anchor alone** (`z37`, against the anchor, not the truth) gives 0.31 / 0.13 / 0.074 / 0.073 points at 14k / 24k / 55k / 98k charged forwards.
  - At the charged cost that is 1.5–2.6× better than sampling. At the measured tracer cost it is about 1.2–2× worse.
  - The MF3 anchor is also 0.57 points from the truth.
- **The MF3 anchor by exact S-lines, outer order 6** (`z41`): −0.044 points mean and 0.063 RMS against the reference anchor. It costs 164k forwards per network (97k–257k), measured.
- **The MF3 anchor with logits interpolated along S and exact roots of the interpolant** (`chatgpt2/round5/r6_core.py`, against the truth): 2.5 / 1.2 / 0.82 / 0.59 points at 683 / 5.4k / 24k / 87k forwards, that is 5–46× sampling's MSE.
- **The exact MF3 region walk** (`z9`, CPU) converges to the MF3 value (mystery 0.953213, to 2e-7) at 5–45M forwards.

### 5.3 Arrangements

- **The anchor does not factor over positions** (`z24`).
  - Across the 5,040 arrangements the MF4 anchor accuracy has an SD of 8 points.
  - The positions of M and S explain 28% of it.
  - Additive position effects explain 18%, and all pairwise position interactions (463 parameters) 48%.
- **The correction does not amortize either** (`z43`; 256 draws per arrangement, 64 development networks).
  - The SD across arrangement means is 2.02 points for true − MF4 (10th–90th percentile 1.14–3.05) and 3.93 for true − MF3 over the 720 MF3 arrangements (2.50–5.41).
  - The positions of M and S explain 16% and 22% of that. With the relative rank order of R and Q added, 35% and 45%.
- **Thinned arrangement subsets for the correction**, granting exact value integration (RMS loss in points):
  - MF3 with 1, 2 or 4 (R) placements per (M, S) class: 0.58, 0.28, 0.23;
  - MF4 with 4 or 8 balanced (R, Q) placements: 0.27 and 0.11.
  - The balanced pairs, as ranks among the 8 positions other than M and S, are (0,7), (7,0), (1,3), (3,1), (2,5), (5,2), (4,6) and (6,4).

### 5.4 Closed forms from linear regions

- **Regions per arrangement** (`z26`).
  - One arrangement's 4-D value space, with the lower inputs at μ(Q), holds a median of 560 regions seen in 2¹⁶ draws, and the count is still rising.
  - The top 100 regions carry 95% of the mass.
  - The uncovered mass is a certificate radius, not an error floor: a fallback can still be accurate.
- **Patterns at grid points** (`z25`): 2.7 / 1.2 / 1.0 / 0.76 points at 5k / 81k / 408k / 1.3M forwards.
- **Patterns chosen by an oracle** (`z30`): 1.4–1.6× sampling's RMS, but the oracle costs about 80M forwards. Discovering patterns along S-lines instead (`z33`) is 13–24× sampling's RMS.
- **Gate histories proliferate** (`z44`; 2¹⁶ inputs split into discovery and test halves).
  - At step 10, a median of 54.5k distinct on/off histories appear.
  - 82% of the test half's mass lies in histories absent from the discovery half.
  - With the activation pattern, both true top positions and the predicted position fixed, 97% of test errors fall in error fragments absent from discovery. That is a separate measurement in `chatgpt2/round5/`.
- **Clock-state linearization** (`z29`) is in §3.3: a 2.3-point bias, with about 230 regions left.

### 5.5 The correction (anchor → accuracy)

- **What the k = 4 integrand looks like** (`z12`).
  - I₄ is a 0/1 indicator with variance 0.97·p(1−p) over configurations. F₄ is barely smoother (0.79·p(1−p)).
  - F₄ − I₄ is nonzero on 12–13% of configurations. 80% of its square lies where the MF4 logit margin is below 0.5.
- **It is not local to the anchor's decision boundary** (`z22`).
  - 68% of the first-order margin model's line penalty sits on S-lines where the anchor never changes its answer.
  - That non-locality concerns that proxy. It is not a lower bound for other estimators.
- **The two opposing flows** of §3.3 (`z42`).
  - Screening each lower input at its largest and smallest Gauss–Legendre node flags 12.7% of configurations, and those hold 98.8% of |F₄ − I₄|.
  - Screening only the largest node misses the very negative inputs. The result is worse than no correction at all.

### 5.6 Deterministic estimators end to end (no fitting)

- **One grid for both parts** (`z38`): the MF3 outer rule × K S-points × 720 arrangements, with the clamped first-order model at every configuration. 5.8 / 2.1 / 0.63 / 0.34 points at 84k / 668k / 3.0M / 10.5M forwards, 3–4 orders of magnitude worse than sampling in MSE.
- **Two levels** (`z41`, reproducing `chatgpt2/round5/r6_correction.py`).
  - **The rule.** The exact-S MF3 anchor at outer order 6, plus Q_coarse(F₄ − I₃).
  - **The coarse part.** It uses 2 nodes per MF4 outer coordinate, 8 conditional-CDF S midpoints, q = 8 and the 720 balanced arrangements. That is 50 rows per configuration, 2.3M forwards.
  - **Result:**

  | variant | forwards | RMS (development / held out) | MSE / sampling |
  |---|---|---|---|
  | as above | 2.47M | 0.225 / 0.249 | 170 / 220 |
  | lower-input rows only on the 50% of configurations with the smallest MF4 margin (`z42`) | 1.36M | 0.242 / 0.260 | 108 / 132 |
  | the same, 30% | 0.92M | 0.290 / 0.281 | 104 / 104 |
  | extreme-node screening first (`z42`) | 1.03M | 0.236 / 0.255 | 79 / 97 |

  - **Where its error comes from.** Most is the coarse integration of the correction: 0.23–0.25 with the anchor taken from the reference. The model contributes 0.065 and the anchor 0.063.
  - **The control variate at work.** The coarse rule integrates I₃ itself 2.07 points high, and that cancels in F₄ − I₃.
- **The measured deterministic frontier falls as cost^(−0.28)** from 700 to 2.5M forwards, against sampling's cost^(−0.5).

### 5.7 Other frames

- **Events** (`z39`). Keeping every record input exact and setting each "neither" input to its conditional mean below the running 2nd max is +1.9 points off (RMS 2.0).
- **State propagation with a mean closure** (`chatgpt2/round5/r6_event.py`).
  - The method carries the hidden vector, M, S and both positions forward in time, with exact branch probabilities and each branch collapsed to its truncated mean or 2 CDF nodes.
  - It is biased by +4.3 to +5.2 points at 200–18k forwards, with or without merging, the size of MF2's leakage. An exact law over affine components has not been tried.
- **Ages of the max and 2nd max** is the top-2 frame (MF2, 5.3 points off).

### 5.8 Fitting on the population

**Predicting from the weights** (`z2`, `z3`; the earlier analyses are in `chatgpt2/`).
- Ridge regression on 20 weight statistics has a cross-validated RMS of 1.28 points. The statistics are the mean, SD, min and max over neurons of W_ih, the self-loop, and the log norms of each neuron's recurrent row and column and readout column.
- 1.28 points is below sampling's error at any budget under about 420 forwards.
- Neuron-role features measured on 4,096 probe inputs give 1.11 (ridge) and 1.0–1.05 (gradient boosting).

**Calibration** (`z45`): cross-validated linear fits of the true accuracy on deterministic per-network quantities.

| deterministic part | + network features | forwards | CV RMS (points) | sampling | MSE / sampling |
|---|---|---|---|---|---|
| coarse MF4/MF3 anchors at 1 outer node | weight statistics (free) | 11.5k | 0.65 | 0.25 | 7.0 |
| coarse anchors at 8 outer nodes | role features (4,096 probes) | 96k | 0.30 | 0.085 | 13 |
| exact MF3 anchor + coarse anchors | role features | 265k | 0.29 | 0.051 | 32 |
| two-level, screened, as components | role features | 1.0M | 0.18 | 0.026 | 46 |

- Calibration cuts the MSE 2–4× at every budget. Zoo errors are mostly under 0.4 points.
- **Sampling-based uses of the population**, for comparison only (not allowed for the estimate): population-informed strata or peer-margin importance weighting gain about 2× over plain sampling (`z1`).

### 5.9 A correction learned from the population (`z46`, `z47`)

**Cells** (`z46`). The top-4 configuration space is cut into equally likely cells:
- **boxes:** n equal-probability pieces of each of the three independent MF4 Beta coordinates, n³ boxes;
- **slices:** K equal slices of S's conditional CDF between Φ(R) and Φ(M);
- **arrangements:** the 720 MF3 arrangements (positions of M, S and R). Q's position is uniform over the other 7.

Each cell has probability 1/(720·n³·K).

**Per-cell features: 2 deterministic forward rows at the cell's centre.**
- The centre is the conditional median of each box piece and the midpoint of the S slice.
- The MF4 row keeps the top 4 exact, with Q at a representative position (the balanced pair of §5.3 for R's rank) and the other 6 inputs at μ(Q).
- The MF3 row keeps the top 3 exact, with the other 7 at μ(R).
- Features: both rows' correctness bits and logit margins.

**Training target.** Each cell's true accuracy, measured offline with 32–128 draws inside the cell. The draws are shared across networks. The mean of a network's cell targets matches the reference to 0.03–0.16 points.

**The model** (`z47`).
- **Inputs:**
  - 4-dimensional embeddings of the box digits, the slice and the four positions;
  - the 2 bits;
  - the 2 margins, raw and divided by the network's RMS margin;
  - network context: the network's mean bits, mean |margins|, share of disagreeing bits and margin scale, optionally plus the 20 weight statistics.
- **Architecture and training:**
  - an MLP of widths 128, 128 and 64;
  - soft binary cross-entropy against the cell targets;
  - Adam with one-cycle learning rate (max 3e-3), batches of 65,536 (network, cell) pairs, at least 1,500 steps.
- **The estimate** for a network is the mean of the model over its cells.

**Results** (forward rows only; see the cost note below):

| cells (boxes × slices × 720) | forward rows | RMS, all 1,024 (cross-fitted; 2 training seeds) | MSE / sampling | zoo MSE / sampling |
|---|---|---|---|---|
| 1 × 1 | 1,440 | 0.64, 0.62 | 0.85, 0.80 | 1.9, 1.8 |
| 1 × 2 | 2,880 | 0.56, 0.56 | 1.29, 1.32 | 2.9, 2.8 |
| 1 × 4 | 5,760 | 0.45, 0.45 | 1.70, 1.71 | 3.2, 3.1 |
| 1 × 8 (development / held out) | 11.5k | 0.36 / 0.35 | 2.1 | |
| 8 × 8, no weight statistics (development / held out) | 92k | 0.33 / 0.36 | 16 | |

- **Weight statistics matter.** Without them the 1 × 1 model is 3.2× (development / held out).
- **The mystery model is consistently 1.4–1.9 points high.** The other five zoo models are mostly within 0.4 points.
- **Cost note: the table counts forward rows only.**
  - The 128-128-64 MLP's inference is 25.8k network-dependent multiply-adds per cell. The embedding and context terms of the first layer are precomputed per cell and per network.
  - That is about 9 forward-equivalents per cell, or 6,452 for the 720 cells of the 1 × 1 model. Counted, the total is 7,892 and the MSE ratio 4.68×.
  - **Smaller models keep most of the accuracy** (1 × 1, weight statistics, cross-fitted, one seed):

  | MLP widths | inference (forward-equivalents) | total cost | RMS, all 1,024 | MSE / sampling, all costs | development / held out | zoo MSE / sampling |
  |---|---|---|---|---|---|---|
  | 128, 128, 64 | 6,452 | 7,892 | 0.640 | 4.68 | 6.0 / 3.9 | 10.3 |
  | 32, 32 | 340 | 1,780 | 0.654 | 1.10 | 2.3 / 0.98 | 2.3 |
  | 16, 16 | 108 | 1,548 | 0.710 | 1.13 | 1.09 / 1.04 | 1.8 |
  | 8 | 24 | 1,464 | 0.837 | 1.48 | 1.41 / 1.40 | 1.6 |

  - So with all costs counted the best point is about 1.1× sampling's MSE at 1.5–1.8k forward-equivalents: close to the line, not on it.
- **Diagnostics** (the 8 × 8 model):
  - **Not overfitting.** In-sample RMS on the 896 training networks is 0.346, against 0.349 on test networks.
  - **Where the error sits.** Almost all of it is in cells where both anchor bits are 1. It is concentrated in the two extreme S slices: S next to R (0.157 of the RMS) and S next to M (0.119), against 0.03–0.05 elsewhere.
  - **Shared across a network.** It is spread over every position pattern and correlated across them. Cells with the max at the last position carry 0.145 RMS on their own, yet removing them leaves the total unchanged (0.347).
  - So the remaining error is a network-level rate of lower inputs breaking right answers, not per-cell noise.
- **Screening rows don't pay.** Adding the 12 extreme-node screening rows to every cell gives 0.24 / 0.22 RMS at 645k rows (51×). Using them only as network context, on every 16th cell, gives 0.27 / 0.32 at 127k (16×).

### 5.10 Best known at each budget

| forwards | best estimate | RMS (points) | sampling RMS | MSE / sampling |
|---|---|---|---|---|
| 0 | weights-only ridge | 1.28 | — | below sampling under about 420 forwards |
| 1,548–1,780 (1,440 rows + small-model inference) | learned cells 1 × 1, widths 16-16 or 32-32, + weight statistics | 0.65–0.71 | 0.62–0.67 | 1.10–1.13 |
| 2,880 rows (+ inference) | learned cells 1 × 2, large model | 0.56 | 0.49 | 1.3 in rows only |
| 11.5k | calibrated coarse anchors + weights; learned cells 1 × 8 | 0.65; 0.36 | 0.25 | 7.0; 2.1 (rows only) |
| 96k | calibrated coarse anchors + roles | 0.30 | 0.085 | 13 |
| 265k | calibrated exact MF3 + coarse anchors + roles | 0.29 | 0.051 | 32 |
| 1.0M | calibrated two-level screened components | 0.18 | 0.026 | 46 |
| 2.5M | two-level, uncalibrated | 0.23–0.25 | 0.017 | 170–220 |

### 5.11 Lower bounds

- **A restricted bound** (`chatgpt2/round5/check_lower_bound.py` checks its numbers).
  - A deterministic method that only learns whether chosen points are errors, about a single polyhedral cone in R¹⁰ of mass at most 0.0746, has worst-case error at least 0.0397·N^(−1/4) after N queries.
  - That exceeds sampling's RMS above about 3,000 queries.
  - It does not transfer to these RNNs, to methods that read the weights, or to an average over trained networks.
- **A general bound seems out of reach.** A white-box lower bound at a fixed size would need a formal computational model; over a fixed finite population, a lookup table answers everything unless the admissible computations are restricted.

## 6. Why it is hard

- **The integrand is a 0/1 indicator in 10 dimensions.** Every decomposition leaves a discontinuous integrand somewhere. The difficulty is spread out:
  - between arrangements (SD 8 points, non-factoring);
  - within each arrangement's value space (hundreds of linear regions);
  - off the anchor's boundary, in the correction;
  - and, for learned estimators, in a network-level breaking rate that cheap rows do not reveal.
- **Exactness trades against cost.** Making fewer inputs exact makes the anchor cheap but moves the difficulty into the correction. Making more inputs exact does the reverse.
- **Learning helps most where errors are systematic.**
  - In principle, a calibrated cell model whose residuals were independent across cells would beat sampling's rate, with error falling like C^(−(d+1)/(2d)) in d dimensions.
  - The measured residuals are instead shared within a network. So more cells do not help unless they also reveal more about that network.

## 7. Directions not yet tried

These are suggestions; better ideas are welcome.

**A. Make the learned estimator cheaper and better.**
- **(a) Inference cost.**
  - Find the smallest model, or a lookup table over the bits, binned margins and cell, that keeps the accuracy.
  - Or restructure: a network-level readout of per-cell features aggregated by a fixed design (sums of bits and margins in cell groups) costs almost nothing.
- **(b) Learned designs.**
  - Learn which points to evaluate in each cell, on the population, then fix them. These could be per-cell probe inputs optimized end to end through the estimate, or a fixed set of N shared probe inputs with a learned readout from the network's logits at those probes, compared with sampling at N forwards.
  - The points are fixed in advance, so the estimate stays deterministic.
- **(c) Measure the breaking rate cheaply.**
  - Find a few designed rows that reveal each network's rate of lower inputs breaking right answers. Candidates: the extreme S slices, cells where both anchors are right, very negative lower inputs, and the max at the last position.
  - Feed them in as network context. Select the rows by their value on the population.
- **(d) Training objective.** Train on the network-level error (the cell mean against the network's accuracy), not per-cell cross-entropy, or on both.
- **(e) Weight features with the right symmetry.** Use encoders invariant to permuting the 16 hidden neurons (set or graph networks over neurons). Also use mechanistic features computed from the weights alone: comparator thresholds, pointer and clock structure, handoff margins.
- **(f) Multi-level learned estimators.** A coarse learned estimate plus learned corrections at finer levels, each fitted to the residual of the level above.
- **(g) Deterministic refinement.** Refine the cells where the model is least certain for this network. That stays deterministic given the weights.
- **(h) The zoo.**
  - Why are ARC's models, the mystery model above all, off? Are their weights or features outside the population's range?
  - Fixes could reweight toward similar networks or train new networks closer to ARC's recipe. More networks can be trained: about 42 ms per step for 1,024 networks at once on one GPU.

**B. Deterministic pieces to combine with learning.**
- **(i) Exact S-integration of the correction.** Trace the correct intervals of each row of F₄ − I₃ along S, merge the endpoints, and integrate the clipped model exactly. An exact S-line costs about 6.3 forwards, less than the 8-point grid it would replace.
- **(j) Exact last-input integration** by a gate sweep, since the last input passes through one step only.
- **(k) Shared recurrent prefixes** across rows and cells: reproduce or refute the claimed 2–3× saving.
- **(l) State propagation with affine components** keeping past-input dependence exact within a pattern, possibly with a closure learned from the population.
- **(m) Error-cone geometry** with a defined finite cover, possibly learned merging.

**C. Different estimators.**
- **(n) Surrogates.** Distil each network into a simpler surrogate whose accuracy is computable exactly, plus a learned correction for the surrogate's error.
- **(o) The sphere.** Use homogeneity to integrate the radius out exactly and work on S⁹.
- **(p) Anything else**, including frames not listed here. For a new frame, please give its anchor's bias and its integral's cost first.

**D. Theory.**
- **(q)** When can a population-learned estimator with weight access beat C^(−1/2)? How fast must network-level information grow with C?
- **(r)** Is there a meaningful lower bound for some formal class of such estimators?

## 8. Work requested, in priority order

1. **Audit (keep this short, but complete).**
   - **`z45`–`z47`:**
     - leakage between training and test networks;
     - the cell decomposition's correctness;
     - target noise;
     - cross-fitting;
     - selection effects (several settings were tried);
     - the cost accounting, including the inference cost in §5.9.
     - With every cost counted, the best point is 1.10–1.13× (one training seed). Is that right, and is there a model or design that gets below 1×?
   - **`z40`–`z44`:** the reference table, the two-level reproduction, gating and screening, arrangement dependence and gate histories.
   - **Comparators and noise:** the sampling comparator √(mean p(1−p)/C), and whether the reference noise matters at each budget.
2. **A better way to do each thing in §5.** For every component, say whether a better method exists, and measure the improvement:
   - the reference and scoring;
   - the cost accounting;
   - the anchors and outer rules;
   - the lower-input models;
   - the two-level estimator;
   - calibration;
   - the cell design, features, model and training of the learned estimator;
   - the weight features.
3. **Extend the learned estimator.**
   - Reach at most 1× sampling's MSE with all costs counted, first at about 10³–10⁴ forwards, then at 10⁵. Currently it is 1.10–1.13× at 1.5–1.8k.
   - Measure it on cross-fitted population networks and on ARC's six models.
4. **New approaches** from §7 and beyond. For each: inline code, its cost, its errors and the zoo errors.
5. **The best end-to-end curve** from 10² to 10⁶ forwards against sampling, all costs counted.
6. **What the learned components have learned**, in terms of §3: for example, which features carry the network-level breaking rate, and why the mystery model is off.
7. **An honest verdict**, and the single experiment most likely to change it.

**Reporting.** For each method, a table with these columns:
- forward rows per network, other arithmetic in forward-equivalents, and the total;
- offline (population) cost;
- RMS error across networks (development / held out, or cross-fitted over all 1,024);
- mean error;
- sampling RMS at the total cost, and the MSE ratio;
- the six zoo errors.

Please say how each setting was chosen. Select on development and report held out without retuning, or cross-fit.

## 9. Practicalities

- **Sampling is fine for measuring:** scoring an estimate, describing an integrand, or making training targets on other networks.
- **Compute.**
  - The GPU scripts fall back to the CPU when no GPU is present. `pop1k/reference.npz` makes scoring free.
  - The learned estimator's cell data for the 1 × 1, 1 × 2 and 1 × 4 layouts is in the zip (`pop1k/seed0_z46_cells_b1_k{1,2,4}.npz`). The other layouts regenerate with `z46.py` (minutes on a GPU, longer on a CPU).
  - We will rerun everything on an RTX 2060 SUPER (8 GB) with 8 CPU cores and 31 GB of RAM, so GPU code (torch) is welcome.
- **Please paste all code inline.** Sandbox files can expire before we download them.

## 10. Building blocks

- `batch_train.forward(Wih, Whh, Wo, x)`: logits (N, B, 10) for x of shape (N, B, 10), batched over N networks. `batch_train.target(x)` gives the target positions.
- `z2.load('pop1k/seed0.npz')` returns ([Wih, Whh, Wo] as float32 torch tensors, step).
- `seeds_clocks.load(path)` returns a zoo model's (Wih, Whh, Wo) as NumPy arrays.
- `common.py` is a NumPy forward pass (`run`), exact to about 1e-12, with an exact Jacobian (`run_jac`).
- `z10.mu_below(v)` gives E[x | x < v]. `z10.topk_mf(x, k)` builds the MFk input.
- `lines.trace_line(Wih, Whh, Wo, x, p, lo, hi, tgt, MAXP)` (numba, float64) returns P(correct | x_p ~ N(0, 1) truncated to (lo, hi)) exactly, plus the cost in forwards and an overflow flag. Retry overflowing lines with a larger MAXP.
- `z11.expansion(x, k, q)` builds the MFk point plus each lower input alone at q equal-probability nodes.
- `z9.Slice` is the exact region walk on an MF3 slice (CPU).
- **Learned estimator.**
  - `z46.py [K] [nin] [--boxes=n] [--screen]` writes `pop1k/seed0_z46_cells*.npz`. It holds, for all 1,030 networks and every cell, `I4`, `I3`, `m4`, `m3` and `tgt`; with `--screen`, also `fl` and `fm`.
  - `z47.py [epochs] [--boxes=n] [--k=K] [--weights] [--cv] [--seed=s] [--hidden=a,b,…] [--screen | --screen-ctx]` trains and scores the model, printing a table and saving each network's estimates.
- **Per-network quantities.** `pop1k/seed0_z41.npz` holds, for all 1,030 networks:
  - the exact-S MF3 anchor and its cost;
  - the coarse MF4/MF3 anchor integrals;
  - the break and fix parts of the correction, with and without screening;
  - the MF4-on-MF4 correction.
- **Copy these rather than import them.** `z18`, `z21`, `z37` and `z41`–`z47` are scripts with no `__main__` guard: importing one runs its whole experiment.
  - `z18.margin(L, y)` computes the margin.
  - `z18.conv_p(m0, h)` computes the exact P(m > 0) for an additive margin over 6 inputs on a node grid.
  - `z21.beta_rule(n, a, b)` and `z37.rule(n, a, b)` give Gauss–Jacobi rules for Beta(a, b).

## 11. Files in the zip

The zip holds the whole repository, minus the git history, a training checkpoint, some archives and the larger regenerable outputs.

- **Population:**
  - `pop1k/seed0.npz` (weights, `acc_2e20`) and `pop1k/reference.npz` (§2);
  - `pop1k/seed0.roles.npz` and `pop1k/seed0.buckets.npz`: per-neuron features (z2's 24, measured on 4,096 probes) and roles;
  - `pop1k/seed0_z41.npz`: the per-network quantities above;
  - `pop1k/seed0_z46_cells_b1_k{1,2,4}.npz`: the learned estimator's cell data;
  - `pop1k/seed0_z47*_est.npy`: the learned estimator's per-network estimates;
  - `pop1k/z*_seed0*.txt`: every script's printed output;
  - `pop1k/notes_circuit_2e30.md`: the population numbers in detail.
- **Write-up:** `REPORT.md`.
  - §7.16 has everything in §4–§5 above, with more detail.
  - §1–§3 and §7.9 cover the mechanism, and §7.13 and §7.15 the earlier estimation attempts on the zoo.
  - §9 is a ledger of every idea tried.
  - `README.md` has setup and a script index.
- **Scripts, by topic:**
  - **population and roles:** `batch_train`, `z1`–`z7`;
  - **lower inputs:** `z8`, `z10`, `z11`, `lines`, `lw2`, `z13`–`z20`, `z32`, `z34`, `z36`;
  - **the integrand and the correction:** `z12`, `z22`, `z42`;
  - **outer rules:** `z21`, `z23`, `z27`, `z28`, `z31`, `z37`;
  - **arrangements:** `z24`, `z43`;
  - **closed forms and regions:** `z25`, `z26`, `z29`, `z30`, `z33`, `z35`, `z44`;
  - **end to end:** `z38`, `z41`;
  - **events:** `z39`;
  - **reference table:** `z40`;
  - **calibration:** `z45`;
  - **learned correction:** `z46`, `z47`;
  - **exact region walk:** `z9`;
  - **sampling-based, for comparison:** `v1`.
- **External analyses:** `chatgpt2/round5/` holds an analysis with its own code (`r6_*.py`): the two-level estimator, the interpolated MF3 anchor, event propagation, geometry diagnostics and the lower bound, with a report and results. `z41`–`z44` check its main numbers.
- **Zoo:** `weights.npz` (mystery) and `weights_seed0..4.npz`.
- **Background only:** the other `QUESTION*_FOR_CHATGPT_PRO.md` files and the rest of `chatgpt2/`, from earlier phases of the project. `REPORT.md` supersedes them where they differ.
