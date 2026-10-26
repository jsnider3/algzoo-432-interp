# Question 2: what can a *population* of trained networks precompute for per-network accuracy estimation?

## The goal in one paragraph

This follows up `QUESTION_FOR_CHATGPT_PRO.md` (the first brief, kept as sent; some of its numbers were later corrected, and REPORT.md is authoritative). The task is unchanged: estimate the **accuracy** of a tiny ReLU RNN that outputs the position of the 2nd-largest of 10 i.i.d. N(0,1) inputs, with **lower MSE than plain Monte Carlo sampling at equal per-network forward passes**. Expensive prework is allowed if it does not look at the target network's weights. **What is new is that we can now train thousands of independent networks with the same recipe** (about 65 networks per GPU-hour at the full budget, and 8× that at 1/8 of it). So the prework may use a large population of *other* trained networks drawn from the same training distribution. **What should that population be used to precompute, so that a newly seen network's accuracy can be estimated much faster than by sampling? How large can the gain be, and what limits it?** Please propose schemes, bound them where you can, and implement and test the best one on the population in the zip.

## Setup (same as before)

- **Network** (AlgZoo `DistRNN`; numpy in `common.py` / `seeds_clocks.py`, batched torch in `batch_train.py`):
  - h_t = ReLU(W_ih·x_t + W_hh·h_{t−1}), with h_0 = 0 and no biases; logits = W_o·h_T.
  - H = 16, T = 10, so 16 + 256 + 160 = 432 parameters.
- **Inputs and target:** x ~ N(0, I₁₀). The answer is argsort(x)[−2].
- **Accuracy:** P_x[argmax logits = answer].
- **Unit of cost:** one forward pass of the *target* network on one sequence. Plain sampling with n passes has SE √(p(1−p)/n).
- **Free (amortized) prework:** anything computed without the target's weights. That includes the task, the input distribution, the architecture, the training recipe, and any number of other networks trained with that recipe, plus their evaluations on any inputs.
- **Per-network cost:** forward passes of the target network. Any other computation on the target's weights is converted to forward-pass equivalents.
- **Honesty requirement:** the estimator must be unbiased, or carry a verified error bar, **for every network**, including ones unlike the population. The population may shape the design (strata, allocation, importance weights, control variates). It must not be able to bias the answer when the target is atypical. A defensive mixture or a difference estimator is the natural way to guarantee this.
- **Evaluation:**
  - Leave-one-out over the population: the target is never part of its own prior.
  - Out of population: the zoo models (`weights.npz`, the "mystery" model at 0.953119; `weights_seed0..4.npz` at 0.910, 0.880, 0.949, 0.932, 0.890) and their weight-perturbed variants. Multiply each weight by 1 + ε·N(0,1) with ε = 0.3% or 1%, which gives accuracies of about 0.94 and 0.83–0.90.
  - Report the variance or MSE ratio against plain sampling at equal per-network passes, with replicate-based error bars.

## Where the first round ended (REPORT 7.13, 7.15, 10)

- **Best task-only sampling design:** exact integration of the last input (P(correct | x₁..x₉) is a sum of at most 19 Φ-differences, at about 2.3 forwards per prefix) combined with an ordered scrambled-Sobol design over the top-3 order statistics. It reaches **about 2.5–4×** sampling's efficiency.
- **Stratification on task features** (sign of S, normalized gaps (M−S)/|M| and (S−R)/|M|, positions): even the Neyman *oracle* with each network's true stratum rates is capped at **about 2.3–5×** for zoo models. In practice, with a population-shrunk allocation, the gain is 1.4×.
  - The reason is that near-tie outcomes are "blurred": the error rate falls smoothly with the gap, so inside task strata the outcome stays near a coin flip.
  - A large gain needs strata in which the outcome is nearly certain, which means knowing *this network's* boundary.
- **Sampling-free quadrature** (mean field over the 7 lower inputs, a boundary-aware scan over (M, a, b) × 720 arrangements): with honest replicate error bars it is **6–25× worse** than sampling at equal cost. A weight-derived Gaussian blur converges to the wrong answer (−2.6 points) because ReLU clipping makes the lower inputs' effect nonlinear.
- **Mechanism** (REPORT 2, 7.9):
  - Good networks are online 2nd-max trackers: the state encodes the ages of the max and the 2nd max, a sparse write-pointer neuron fires one step after events, and the readout decodes the resulting impulse response.
  - Training forks stochastically between "memory" and "online selection" strategies (REPORT 6, 7.14). About 37–50% of runs switch to selection, and the outcome is not predictable from the initialization.
  - So a population is probably a **mixture of a few strategy types** (which neuron becomes the pointer, memory vs selector, and so on), with continuous variation within each type.

## The new resource: a population, and a first measurement

- **Trainer:** `batch_train.py` trains N independent networks as stacked tensors under one AdamW. AdamW is elementwise and the loss is a sum of per-model means, so this is exactly N independent runs. It follows AlgZoo's recipe: lr 0.04/√H, weight decay 0.001, batch 1024, 2¹⁴ warmup steps, exponential decay to 0.01× over the first half, PyTorch default init. It matches `nn.RNN` to 3e-7.
  - Hardware: one RTX 2060 SUPER. At N = 1024 it runs about 53 ms/step, so 1,024 networks at the full 2³⁰-sequence budget take about 15.5 h.
  - 10k networks would take about 150 GPU-h at 2³⁰, or about 19 h at 2²⁷.
- **Pilot run:** 1,024 networks at 2³⁰ sequences, about 10% done; the rest arrives in a follow-up message. Each saves its init, snapshots at 2²⁰, 2²², …, 2²⁸ sequences, and final weights.
- **In the zip now:** `chatgpt2/pop1k_2e24_2e26.npz` holds the same 1,024 networks at **2²⁶ sequences** (mid-training, accuracy 0.78 / 0.87 / 0.92 at the 5th / 50th / 95th percentiles), plus their 2²⁴ snapshot and their initializations.
  - Keys: `init_{Wih,Whh,Wo}` and `s{16384,65536}_{Wih,Whh,Wo}`, of shapes (1024,16), (1024,16,16) and (1024,10,16). Snapshot key s = steps of 1024 sequences, so s65536 = 2²⁶ sequences.
  - `acc_s65536_2e18` holds their accuracies on 2¹⁸ inputs.
  - **Prototype on this; the method should then transfer to the converged population.**
- **Older networks:** 16 more networks at 2²⁷ (`pop/p*.npz`), the 6 zoo models at 2³⁰, and zoo checkpoints `ckpt/s{seed}_e{k}.npz` after 2^k sequences.

**First measurement (`z1.py`).**

- **Method:** evaluate all networks on one fixed pool of inputs. For target i, the leave-one-out population score is p̄₋ᵢ(x), the fraction of *other* networks wrong on x. It is fully precomputable.
- **Design:** stratify on quantile bins of p̄₋ᵢ (20 bins, plus a separate stratum for inputs no other network gets wrong). The precomputable allocation uses the population's own stratum error rates with 10% defensive proportional allocation; the oracle is Neyman with the target's true rates.
- **Gains:** exact variance ratios on the pool. Median and [5%, 95%] across networks:

| population | precomputable p̄-strata | oracle p̄-strata | precomputable task strata | oracle task strata | corr(error, p̄₋ᵢ) |
|---|---|---|---|---|---|
| 1,024 nets at 2²⁶, pool 2¹⁸ | **1.98** [1.51, 2.33] | 2.05 [1.54, 2.83] | 1.52 [1.19, 1.89] | 1.54 [1.22, 2.19] | 0.51 [0.46, 0.55] |
| 22 older nets (mixed budgets), pool 2²² | 1.38 | 1.71 | 1.31 | 1.32 | 0.48 |

- Only 3.7% of inputs are never misclassified by any of the 1,024 mid-training networks.
- A correlation of 0.5 means that a single population-average control variate would give only 1/(1−ρ²) ≈ 1.3×.
- **Our reading:** a *single* population-average score is too blunt, because networks differ in *which* near-ties they get wrong. The structure, if it exists, is probably in the **mixture**: the target's error set may be well predicted by a *weighted subset* of population members, or by a low-dimensional family, not by the average.

## Questions

1. **Which kinds of population-level precomputation can beat the task-only ceiling of about 3–5×, by how much, and what bounds them?** Ideas to react to, improve or replace:
   - **(a) Population-basis control variates (regression estimator).** Precompute exactly (or to very high precision) each population member's accuracy and error indicators e_j(x) on a shared design. For the target, fit e_target(x) ≈ Σ β_j e_j(x) (or on features: p̄ within strategy clusters, soft-max-margin surrogates of the members' logits) using the target's own samples. The estimate is the sample mean of e_target − Σ β_j e_j plus Σ β_j·acc_j. This costs no extra target passes, and the gain is about 1/(1−R²). How large can R² get with K basis networks? How should the β be fit and regularized so that the error bar stays honest (e.g. cross-fitting)?
   - **(b) Two-stage, mixture-aware designs.** Spend a small pilot budget on a precomputed probe set of maximally discriminating inputs, identify which population members (or clusters) the target resembles, then stratify or importance-sample with the neighbours' error profile. The pilot passes are counted, and the second stage stays unbiased.
   - **(c) A low-dimensional model of error sets.** Do trained networks' error regions lie in a low-dimensional family, for example a few per-strategy boundary offsets and blur widths in normalized gap coordinates, times age-dependent factors? If so, the per-network work is to estimate a few parameters with targeted passes, read the accuracy off a precomputed map, and debias with a small residual sample.
   - **(d) Amortized predictors from weights or cheap probes.** Predict accuracy across the population from weight features, impulse responses, or a few hundred probe evaluations, then use the prediction only as a prior or control variate with an unbiased correction. How much variance can the correction remove?
   - **(e) Importance sampling with a population-predictive proposal.** For a 0/1 integrand, how close to zero variance can a proposal q(x) built from the population's predictive error probability get? Is there a clean bound on the achievable gain, given how well a network's errors can be predicted from the population, such as a Bayes-optimal or posterior-predictive ceiling that our oracle-Neyman numbers can be compared with?
2. **Theory.** For a 0/1 integrand where a prior over integrands comes from a population, what is the best achievable variance reduction at equal target evaluations? Ideally state it in terms of something we can measure on the population, so we know when to stop.
3. **Population design.** If the answer depends on population structure, what should we train? How many networks are needed before the leave-one-out quantities saturate? Is 2²⁷ training (8× cheaper) good enough, or does the converged population differ qualitatively? Which snapshots or statistics should we save? Is 10,000 networks worth about 150 GPU-hours, or would 3,000 do?
4. **Code.** Please give a script, numpy or torch, that implements your best scheme with the evaluation protocol above. We will run it on the GPU on the converged 1,024 and, if justified, on 10k networks. **Please paste all code inline in your answer:** last time the code in the sandbox was not recoverable. `batch_train.forward` and `z1.error_matrix` / `z1.strat_gain` are available to build on.

Please also tell us if you think that population-derived precomputation **cannot** give a large factor here, and why. A scheme that works only for converged networks of one strategy type is still interesting, but please say so.

## Files in the zip

- **Write-ups:**
  - `REPORT.md`: the full write-up. Sections 7.13 and 7.15 cover estimation, section 9 is a ledger of everything tried, and section 10 lists corrections.
  - `README.md`: setup and a script index.
- **Population tools:**
  - `batch_train.py`: the batched trainer, and `forward` / `target` / `accuracy` in torch.
  - `z1.py`: the population ceiling measurement above.
  - `chatgpt2/pop1k_2e24_2e26.npz`: the 1,024-network population.
- **Other weights:**
  - `weights.npz` and `weights_seed{0..4}.npz`: the zoo models (keys `rnn.weight_ih_l0` (16,1), `rnn.weight_hh_l0` (16,16), `linear.weight` (10,16)).
  - `pop/p{0..15}.npz`: 16 networks at 2²⁷ sequences.
  - `ckpt/`: the zoo training checkpoints.
- **First-round estimators:** `w1.py`, `w2.py`, `w5.py` (ordered Sobol, last-input integration), `w8.py`, `w9.py` (adaptive strata, Neyman bounds), `i8.py` (weight-perturbation protocol), and `r4.py`, `t2.py` (quadrature).
