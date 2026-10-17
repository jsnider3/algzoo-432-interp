# Question: can network-independent precomputation make per-network accuracy estimation beat random sampling?

## The goal in one paragraph

We have tiny ReLU RNNs trained on an algorithmic task: give the position of the 2nd-largest of 10 i.i.d. N(0,1) inputs. We want to estimate each network's **accuracy** (fraction of inputs where argmax of the logits equals the true 2nd-argmax) to high precision, using **less per-network compute than plain Monte Carlo sampling** for the same mean squared error. We are allowed to do expensive **prework once, before any particular network is given**: precomputation that depends only on the input distribution, the task and the architecture, not on the weights. That prework can then be reused for every network we are later handed. **What prework would make the per-network estimate substantially cheaper than sampling, and how would you use it?** Please propose concrete schemes, analyse their expected cost-versus-error scaling, and, if you can, implement and test one using the code and weights in this zip.

## Precise setup

- **Architecture** (ARC's AlgZoo `DistRNN`; numpy copy in `common.py` and `seeds_clocks.py`): one-layer RNN with no biases. Hidden size H (16 for the main models), input size 1, sequence length T = 10.
  - Recurrence: h_t = ReLU(W_ih·x_t + W_hh·h_{t−1}), with h_0 = 0.
  - Output: logits = W_o·h_T (10 logits).
  - Parameter count for H = 16: 16 + 256 + 160 = 432.
- **Input distribution:** x ∈ R¹⁰ with i.i.d. N(0,1) entries.
- **Correct answer:** argsort(x)[−2]. Ties have probability zero.
- **Accuracy:** P_x[argmax(W_o h_T(x)) = argsort(x)[−2]]. Typical values are 0.88–0.98.
- **Unit of compute:** one network forward pass on one 10-number sequence ("one evaluation"). Plain sampling with N evaluations has standard error √(p(1−p)/N); for p ≈ 0.95 that is about 0.006 percentage points at N = 14M.
- **The target:** at equal *per-network* evaluations, an estimator with lower MSE than sampling, ideally by a large factor, and with an honest error bar. Precomputation that does not touch the network's weights is free (amortized). If your scheme needs *weight-dependent* precomputation, count it as per-network cost.
- **Evaluation protocol** (as ARC suggests): test on many networks, e.g. the provided trained weights plus weight-perturbed variants with i.i.d. Gaussian noise scaled to each weight (0.3% and 1% noise give accuracies of about 0.94 and 0.83–0.90), and report MSE against a high-precision reference (we used 20M–200M samples).

## What is in the zip (all paths relative to the zip root)

- **`REPORT.md`:** the full write-up of our interpretability investigation. **Section 7.13 is the most relevant**: our attempts at sampling-free and low-variance accuracy estimation and how far they got. Section 9 is a ledger of everything tried.
- **Weights** (`.npz`, keys `rnn.weight_ih_l0` (H,1), `rnn.weight_hh_l0` (H,H), `linear.weight` (10,H)):
  - `weights.npz`: the 432-parameter "mystery" model (`alg_zoo.example_2nd_argmax()`), accuracy 0.953119 ± 0.000015 from 200M samples.
  - `weights_seed0.npz` … `weights_seed4.npz`: other 16-neuron trained models (accuracies 0.910, 0.880, 0.949, 0.932, 0.890).
  - `weights_h4/h6/h8/h32.npz`: other hidden sizes.
  - `ckpt/s{seed}_e{k}.npz`: training checkpoints after 2^k sequences, for k = 0 and 20–30; 60 files.
- **Key code:**
  - `common.py`, `seeds_clocks.py`: numpy RNN (`run`, `load`).
  - `r2.py`, `r4.py`: our best sampling-free estimator ("boundary-aware quadrature in scale-adapted coordinates", described below); `r4.py` adds prefix caching.
  - `cache.py`: prefix-trie caching of the 720 arrangements.
  - `t2.py`: randomized-replicate version that gives an honest error bar.
  - `s1.py`, `s6.py`: reference accuracies by sampling.
  - `o1.py`: control-variate experiment.
  - `g1.py`, `i7.py`, `i8.py`: stratified-sampling estimators.
  - `j5.py`: Gauss–Legendre helper `gl`.
- **Dependencies:** numpy, scipy; torch only for the training scripts. The AlgZoo library itself is not included and is not needed for estimation.

## What we already know (facts you can rely on, with evidence in REPORT.md)

1. **Positive homogeneity.** With no biases, logits(c·x) = c·logits(x) for c > 0. The decision depends only on the direction of x, not its scale. In practice the decision boundaries are close to fixed in *normalized* gap coordinates, (M−S)/|M| and (S−R)/|M| (M, S, R = 1st, 2nd and 3rd largest).
2. **Errors are concentrated** in near-tie regions: small (S−R)/M and small (M−S)/M. There is also a "blind spot" where the 2nd largest is negative (about 0.9% of inputs, 77% error). On "clean" inputs (S > 0, both gaps above 0.1·M) the error rate is about 0.2%.
3. **Mean-field reduction works for good networks.** Replace the 7 smallest inputs by their conditional mean given R, μ(R) = −φ(R)/Φ(R), keeping the true top-3 values and positions. The resulting accuracy equals the true accuracy within 0.01–0.07 points for the "online selector" networks (sampled check, 2M–20M samples). This works because these smaller inputs change 3–7% of individual outputs, but the outputs they break and the ones they rescue cancel.
   - For "memory-strategy" networks (e.g. seed 0) the mean-field bias is about +0.08 points, and making the 4th order statistic exact as well makes it worse (+0.27).
   - The bias seems to come from the *variance* of the smaller inputs feeding linear memory channels.
4. **With mean field, accuracy is a 3-dimensional integral over the top-3 order statistics,** times a sum over 720 arrangements of their positions. The density is φ(M)φ(S)φ(R)Φ(R)⁷ per arrangement, and the integrand is a 0/1 step function (decision correct or not).
5. **Our best estimator** (`r4.py`):
   - Outer Gauss–Legendre over M and a = (M−S)/|M|.
   - For each line, scan b = (S−R)/|M|, locate decision flips by bisection, and integrate the density exactly between flips.
   - Prefix caching: the 720 arrangements share prefixes, so 2,815 trie states replace 7,200 RNN steps per line.
6. **Results of that estimator:**
   - With fixed grids it converges to within 0.002–0.004 points for the selector networks at 13–27M evaluations. One lucky grid reached 0.36× sampling's error.
   - **But with an honest error bar** (randomized Gauss–Legendre replicates, `t2.py`), its standard error is **6–20× sampling's** at equal cost. Convergence is erratic because the integrand is discontinuous.
7. **Other things that did not beat sampling:**
   - A control variate using the mean-field indicator (0.4–0.7× efficiency, because each sample costs 2 evaluations and the correlation is only about 0.6).
   - Linear, additive or cut-HDMR models of the smaller inputs' leakage.
   - Plain stratified sampling by normalized gap strata does help, by **1.2–2.8×** lower MSE with a network-agnostic allocation, but it is still sampling.
8. **Mechanistic structure that might be exploitable** (sections 2 and 7.9 of the report):
   - Good networks implement an online, nearly time-translation-invariant "2nd-max tracker": the state encodes *how long ago* the max and the 2nd max appeared.
   - A sparse "write-pointer" neuron fires one step after events, and the readout decodes the resulting impulse response.
   - The readout is (nearly) the convex softmax-regression optimum given the recurrent features.

## The specific question

**Is there network-independent precomputation that makes per-network accuracy estimation much cheaper than sampling?** Ideas we have not tried, to react to, improve on or replace:

- **(a) Precomputed integration designs over input space.** Low-discrepancy, stratified or importance designs in the (M, a, b, positions) order-statistic space, or in the normalized "direction" space, optimised once for the *family* of networks. For example, concentrate points near the near-tie manifolds where all trained networks put their decision boundaries.
- **(b) Precomputed decision-boundary geometry.** All networks solving this task must put boundaries near the true task boundaries (ties between ranked values). Could we precompute a parametrization of inputs relative to the task's tie hyperplanes? Then a network's accuracy would be an integral over thin shells around them, and only a low-dimensional "offset" would need measuring per network.
- **(c) Exploiting piecewise linearity.** The network is piecewise linear and homogeneous, so the correct-region is a finite union of polyhedral cones intersected with the Gaussian. Is there a precomputable structure, such as Gaussian measures of cone families or tables for orthant- and polytope-probabilities, that turns per-network estimation into cheap lookups once the network's activation regions near the boundaries are found?
- **(d) Amortized or learned estimators across a family of networks.** Train a meta-model on many networks (the checkpoints and weight-perturbed copies give thousands): it predicts accuracy from cheap per-network measurements (e.g. a few thousand cleverly chosen evaluations), and acts as a learned control variate or regression estimator with a calibrated error bar. How do you keep it honest, i.e. unbiased, or with a verified error bar?
- **(e) Variance reduction specific to the step-function integrand.** Conditional Monte Carlo along one coordinate, which we partly do with boundary-aware inner integration: precompute the conditional densities so that per-network work is just root-finding. Antithetic or permutation symmetries are another option: the input distribution is exchangeable across positions.

For whichever schemes you think are most promising:
1. explain **why** the cost should scale better than sampling;
2. give the per-network cost and the expected MSE;
3. state what precomputation is needed and whether it is truly network-independent;
4. if possible, **implement and benchmark it** on `weights.npz`, `weights_seed2.npz` and `weights_seed0.npz` (references 0.953119, 0.94851, 0.91009), and on weight-perturbed variants, against sampling at equal per-network evaluations, with honest error bars.

An approach that only works for "selector" networks is still interesting, but please say so. Please also tell us if you think beating sampling by a large factor is impossible here, and why. ARC (who posed this model as a challenge) describe matching sampling as "a major research breakthrough".
