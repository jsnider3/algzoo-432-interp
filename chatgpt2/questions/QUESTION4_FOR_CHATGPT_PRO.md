# A deterministic accuracy estimate for a population of tiny RNNs, at sampling's cost

The attached zip contains **1,024 independently trained networks** of one small architecture: ARC's AlgZoo 2nd-argmax RNN, a 432-parameter ReLU RNN that outputs the position of the 2nd-largest of 10 Gaussian inputs. It also contains the research code and write-up behind them.

We want a method that, given any one network from this population, **estimates its accuracy mechanistically, from its weights, at least as well as random sampling does for the same compute**. The method is scored across all 1,024 networks.

We have a model of the hard part of the integrand, the effect of the smaller inputs, that is accurate enough across the population. What we lack is a cheap way to integrate it over the 4 largest inputs, and that is the question.

## The goal

ARC's estimation challenge: they would "consider it a major research breakthrough" to produce a **mechanistic** estimate, one that "reasons deductively based on the structure of the model" rather than from sampled examples, that "matches the performance of random sampling in terms of mean squared error versus compute". The comparison is averaged over independently trained networks, which is why we trained a population.

Concretely:
- Plain sampling with C forward passes has MSE p(1−p)/C for a network of accuracy p.
- So at every compute level C, the estimate's **MSE averaged over the 1,024 networks** must be at most the average of p(1−p)/C. Equivalently, an estimate with error ε may cost at most about p(1−p)/ε² forward passes, and this must hold at every C, not at one precision.
- Across the population p(1−p) averages about 0.070, so sampling reaches the following precision:

  | forwards per network | 700 | 7,800 | 70,000 | 780,000 |
  |---|---|---|---|---|
  | sampling RMS error (points) | 1 | 0.3 | 0.1 | 0.03 |

"Points" are percentage points of accuracy throughout.

## Setup

- **Architecture** (AlgZoo `DistRNN`):
  - h_t = ReLU(W_ih·x_t + W_hh·h_{t−1}), with h_0 = 0 and no biases; logits = W_o·h_T.
  - H = 16 and T = 10, so 16 + 256 + 160 = 432 parameters.
- **Task.** x ~ N(0, I₁₀), and the answer is the position of the 2nd-largest input. Accuracy is P_x[argmax logits = answer].
- **The population** (`pop1k/seed0.npz`).
  - 1,024 networks, trained independently with AlgZoo's recipe (`batch_train.py`) to 2³⁰ sequences. The weights are the keys `s1048576_{Wih,Whh,Wo}`, with shapes (1024, 16), (1024, 16, 16) and (1024, 10, 16).
  - The true accuracy of each network is in `acc_2e20`, measured on 2²⁰ inputs (SE about 0.026 points, small enough for scoring up to about 10⁵ forwards).
  - Accuracy is 0.924 ± 3.4 points across networks (5th, 50th and 95th percentiles: 0.860, 0.932, 0.964).
- **Out-of-population check.** ARC's six released models of the same architecture (`weights.npz`, the "mystery" model, and `weights_seed0..4.npz`). They were trained by ARC with the same recipe.
- **Unit of cost.** One forward pass of the target network, counted as 2,880 multiply-adds (as in `v1.py` and `lines.py`).
  - Any other work on the target's weights (line tracing, region walking, root finding, polygon integrals) is converted at that rate, as `z9.py` does.
  - **Work that never looks at the target's weights is free.** That includes studying the other networks in the population.

## How the networks work, and how they differ (REPORT.md §7.16, §5, §7.9; `pop1k/notes_circuit_2e30.md`)

- **They share one design with two variants.** Every network tracks the running max M and running 2nd max S online, and comparator neurons decide each event. Each new input is tested against thresholds that blend M and S: it can be a new max (the old max is "handed off" to become the 2nd max), a new 2nd max, or neither.
  - About 54% of networks are **selectors**. A sparse write-pointer neuron fires on, or one step after, each event, and the state encodes how long ago the max and 2nd max appeared.
  - The rest are **memory** networks, which keep a decaying record of past inputs instead.
  - The variant is fixed early in training and can be read from about 32 probe inputs. The two variants barely differ in where they err.
- **Neuron roles per network**, on average:
  - 3.1 comparators (96% of networks have at least one);
  - 0.9 max detectors (78% have one);
  - 6.0 always-on neurons;
  - 5.3 others.
- **Where errors come from.** One source is blur near ties (M ≈ S, or S ≈ R, the 3rd largest). The other is failed handoffs, above all when the new max arrives at the last step. Errors settle within about 3 steps after the last event.
- **Where networks differ.** About 62% of the across-network variance in accuracy is in the non-tie regions, where handoffs are the largest factor. Last-step handoffs alone carry about a quarter of all the variance, from 8% of inputs.
  - A last-step handoff fails 12.3% of the time on average, with an SD of 13.5 across networks and a correlation of −0.81 with accuracy.
  - It fails 38% of the time with no comparators and 3% with five or more.
  - The ties carry the other 37% of the variance.
- **Every network is positively homogeneous:** h(cx) = c·h(x) for c > 0.

## What is established, across the population

**1. The decomposition.**
- A top-4 configuration c is an arrangement a (the positions of the 4 largest inputs, one of 10·9·8·7 = 5,040) plus their values v₁ > v₂ > v₃ > v₄.
- Its density is φ(v₁)φ(v₂)φ(v₃)φ(v₄)Φ(v₄)⁶ for every arrangement.
- Given c, the other 6 inputs are iid N(0,1) truncated below v₄. So accuracy = E_c[P(correct | c)].

**2. Mean field, and why it isn't enough on its own.**
- **"MFk" mean field** keeps the top k inputs exact and replaces every other input by its conditional mean given that it lies below the k-th largest: μ(v) = −φ(v)/Φ(v).
- **"Leakage"** is accuracy(MFk network) − true accuracy.
- **MF3's leakage is +0.25 ± 0.51 points across the 1,024 networks** (RMS 0.57). It varies by network, so no constant correction fixes it, and network features predict it only to about 0.4 points. At that error, even an MF3 accuracy computed exactly and for free would be worth only about 2,000 samples.
- **The leakage lives where R > 0, as two mechanisms that nearly cancel.** One is the true network answering at the 4th-largest's position, worth +0.45 points on average, which MF3 cannot do. The other is S-vs-R swaps, netting −0.35. Keeping the 4th largest exact (MF4) removes both.

**3. A lower-input model that works for every network** (`z11.py`, `lines.py`):

  P̂(c) = I₀(c) + Σ_p (E_p[I](c) − I₀(c)), over the 6 positions p outside the top 4, clamped to [0, 1].

- I₀ is the network's correctness at the **MF4 point**, where the 6 smallest inputs sit at μ(v₄).
- E_p[I] varies the input at position p alone over its law, with the other 5 at μ(v₄). This is first-order cut-HDMR around MF4.
- **Each E_p is exact.** Along one coordinate the network is piecewise affine, so `lines.trace_line` sums Φ masses over the pieces; it matches a 20k-point grid to 4e-5. All 6 lines cost about 65 forwards per configuration, or 97 with 16 equal-probability nodes per line.
- **Scored per network**, E_c[P̂] − true accuracy, in points, as mean ± SD across networks. Configurations were sampled here only to test the model, on the same inputs as the true accuracy.

  | model | networks | error (points) | notes |
  |---|---|---|---|
  | P̂, 16 nodes per line | all 1,024 | −0.013 ± 0.055 (RMS 0.057) | correlation 0.98 with each network's true leakage |
  | P̂, exact lines | every 8th | about −0.05 ± 0.04 | after removing scoring noise |
  | MF3 alone | all 1,024 | +0.25 ± 0.51 | |
  | MF4 alone | all 1,024 | +0.30 ± 0.28 | |

- **So if E_c[P̂] were computed exactly and for free, the estimate would be worth about 150k samples, averaged over the population.**
- **Mechanistically**, in every network the decision depends jointly on the top-4 values and positions, and each of the 6 smallest inputs acts on it independently.

**4. What the integrand looks like** (`z12.py`, `pop1k/z12_seed0.txt`). These are medians over every 16th network, with configurations sampled only to describe the integrand.
- **I₀ is a 0/1 indicator**, with variance 0.97·p(1−p) over configurations. P̂ is barely smoother, at 0.79·p(1−p). So sampling configurations gains only about 1.3× per configuration, while paying about 65 forwards for each.
- **The penalty P̂ − I₀ lives in a thin shell around the MF4 decision boundary.**
  - It is nonzero on 12% of configurations (7–19% across networks).
  - 80% of its square lies where the MF4 logit margin is below 0.5, which is 6% of configurations.
  - 92% of its square lies within margin 1.
- **Arrangements carry little of the variance.** The 5,040 arrangement means explain 6–8% of the variance of the truth, of I₀ and of P̂ (3–25% across networks), and 1% of the penalty's. The rest is the integral over the 4 values within each arrangement, that is, where each arrangement's decision boundary lies.

**5. What has been tried for the configuration integral.** Most of these were developed on the zoo models before the population existed.

| attempt | result |
|---|---|
| Exact linear-region walk for MF3 (`z9.py`): 720 arrangements; on each slice of fixed R the MF3 input is affine in (M, S), so the incorrect polygons are integrated exactly, with Gauss–Legendre in R | Converges to each network's MF3 value (run on three zoo models). But it costs 5–45M forwards per network. 72–80% of the cost is R < 0, where the regions are many. The error is spread over arrangements: 139 of 720 hold half of it. MF4 would add 7× the arrangements and one dimension. |
| Boundary-aware quadrature over (M, (M−S)/\|M\|, (S−R)/\|M\|) × 720 arrangements, in scale-adapted coordinates with prefix caching (REPORT §7.13, zoo models) | With honest randomized replicates, its SE is 6–25× sampling's at equal cost. Convergence is erratic because the integrand is discontinuous at the ties. |
| Gaussian blur of the mean-field boundary, with width derived from the weights (REPORT §7.15) | Converges to −2.6 points, because ReLU clipping makes the lower inputs' effect nonlinear |
| First-order lower-input model around MF3, on an ordered-Sobol top-3 design (`z8.py`, all 1,024) | 4–10× short of sampling. It also samples the configurations. |
| Exact integration along one input (the last input, or the 2nd-max value), averaged over sampled inputs (REPORT §7.15, `v1.py`) | 2–6× better than sampling, but a constant factor, and it samples |
| Predicting accuracy from neuron roles or weights across the population (`z2.py`, `z3.py`) | 1.0–1.2 points CV RMSE, worth about 700 samples. It is not an integral of the target network. |

## What we would like from you

1. **A deterministic method that computes E_c[P̂(c)] for any network in the population** (or a variant of the lower-input model with similar accuracy), **with MSE against compute at or below sampling's**, at least from about 10³ to 10⁵ forwards per network.
   - Score it as RMS error across networks at each compute level, with sampling's √(mean p(1−p)/C) beside it.
   - Develop on a subset, for example every 16th network. Give code that runs on all 1,024; we will run it on a GPU. Report the zoo models as an out-of-population check:

     | model | accuracy |
     |---|---|
     | mystery (`weights.npz`) | 0.953119 ± 0.000015 |
     | seed 2 | 0.94851 |
     | seed 0 | 0.91009 |
     | seeds 1, 3, 4 | about 0.880, 0.932, 0.890 (please compute a precise reference) |

   - If it can't reach the line, say how far off it is, for which networks, and why.
2. **Directions we think are worth testing.** Choose others if they are better.
   - **(a) Boundary first.** P̂ is an indicator except in a thin shell.
     - Find the MF4 decision boundary in the 4 values for each arrangement, or for each class of arrangements.
     - Integrate the Gaussian density on one side of it exactly. For example, integrate exactly along one coordinate, as `v1.py` does along the 2nd-max value, with the other coordinates on a structure-adapted grid.
     - Add the penalty as an integral over the shell only.
   - **(b) Share work across arrangements** through the causal structure.
     - Arrangements share prefixes, so they form a prefix tree.
     - Between events the MF4 input is the same μ(v₄) at every step, so the idle dynamics is one shared map.
     - The networks encode ages, and only 6–8% of the variance lies between arrangements.
   - **(c) Use homogeneity** to integrate out the overall scale analytically. That is exact for the true network. It is approximate for MF4, which is not homogeneous because μ(v₄) is nonlinear in v₄; one option is v₄ as the outer integral, with the rest as ratios.
   - **(d) Certificates for the penalty.** Skip line tracing wherever a cheap bound proves that no single lower input can flip the decision, for example interval bounds through the network. Then the 65-forward model is paid only in the shell.
   - **(e) A hierarchy across budgets.** At 10³ forwards sampling is only good to about 0.8 points, so a coarse MF3 integral (0.57 RMS bias) may already be on the line there. MF4 plus the penalty would take over at higher budgets.
   - **(f) Learn the shared structure from the population, then fix the method.** All 1,024 networks solve the same task with the same few mechanisms, so their decision boundaries may sit in similar places: near the ties, and in the handoff regions after late events. The population is free to study. It can decide in advance where to place grids, which arrangements can be merged, and how fine to resolve each region.
     - The method must then be applied unchanged to the target's own weights.
     - Validate it on networks held out from that study.
3. **Optional: the −0.05-point residual** of the exact first-order model. Is it interactions among the 6 smallest inputs, concentrated in the shell? Would a second-order term restricted to the shell remove it cheaply?
4. **An honest verdict.** If you think the top-k decomposition is the wrong frame, say so and show it. For example, the integral might be easier in ages and event values, or along the 2nd-max line.

## Constraints

- **Mechanism, not statistics.** The estimate for a network must be a deterministic computation on that network's own weights.
  - Not allowed:
    - random or quasi-random point sets over the input or configuration space, including unscrambled Sobol or lattice rules used in place of sampling;
    - control variates, and importance or stratified sampling;
    - any fitted prediction of accuracy or error from the population, such as a regression or a calibration of the final number.
  - Allowed:
    - points the method chooses from the weights, such as region vertices, line pieces, roots, and quadrature nodes for a smooth reduced integrand;
    - design choices learned from other networks in the population, as in (f).
- **Random sampling is fine for measuring**: scoring an estimate against the truth, or describing the integrand, as `z11.py` and `z12.py` do.
- **Please paste all code inline.** Sandbox files can expire before we download them. We will rerun everything on an RTX 2060 SUPER (8 GB) with 8 CPU cores.
- **Building blocks you can reuse:**
  - `lines.trace_line` (numba): the exact integral along one coordinate.
  - `z9.Slice`: the region walk on an MF3 slice.
  - `z11.expansion`: the MF4 point plus its q-node rows.
  - `batch_train.forward`: a forward pass batched over networks and inputs.
  - `z2.load('pop1k/seed0.npz')` returns the population's weights as torch tensors, and `seeds_clocks.load` loads a zoo model.

## Files in the zip

The zip holds the whole repository, minus the git history, a training checkpoint and some archives. The parts that matter here:

- **Population:**
  - `pop1k/seed0.npz`: the 1,024 networks, with keys `s1048576_{Wih,Whh,Wo}` and `acc_2e20`. It also holds training snapshots, `s<step>_*`, and the initializations, `init_*`.
  - `pop1k/seed0.roles.npz`, `pop1k/seed0.buckets.npz`: per-neuron features and roles.
  - `pop1k/z*_seed0.txt`: printed outputs of the population analyses.
- **Write-up:**
  - `REPORT.md` §7.16 covers the population and the lower-input model above. The earlier sections study the mystery model and the other zoo models in depth: §1–§2 and §7.9 cover the mechanism, §7.13 and §7.15 the earlier estimation attempts, and §9 is a ledger of everything tried.
  - `pop1k/notes_circuit_2e30.md` has the population numbers in detail.
  - `README.md` has setup and a script index.
- **Code:** `z9.py`, `z10.py` (leakage anatomy), `z11.py`, `z12.py`, `lines.py`, `v1.py`, `batch_train.py`, `z2.py`, `z3.py` (neuron roles), `z4.py` (error anatomy), `seeds_clocks.py`, `common.py` (NumPy forward pass).
- **Zoo:** `weights.npz` (mystery) and `weights_seed0..4.npz`.
- **Background only:** `QUESTION*_FOR_CHATGPT_PRO.md` and `chatgpt2/`, from earlier phases of the project. `REPORT.md` supersedes them where they differ.
