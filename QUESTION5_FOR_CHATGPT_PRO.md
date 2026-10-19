# Can a deterministic method match random sampling for the accuracy of tiny RNNs? A map of what fails, and why

The attached zip contains **1,024 independently trained networks** of one small architecture: ARC's AlgZoo 2nd-argmax RNN. It is a 432-parameter ReLU RNN that outputs the position of the 2nd-largest of 10 Gaussian inputs. The zip also has the research code and a long write-up (`REPORT.md`, section 7.16 is the relevant part).

We want either a method that estimates each network's accuracy **mechanistically, from its weights, at least as well as random sampling does for the same compute**, or a convincing argument that no method of a stated kind can. Below is a precise map of what has been measured. The question is what lies outside it.

## The goal

ARC's estimation challenge: they would "consider it a major research breakthrough" to produce a **mechanistic** estimate, one that "reasons deductively based on the structure of the model" rather than from sampled examples, that "matches the performance of random sampling in terms of mean squared error versus compute".

- Plain sampling with C forward passes has MSE p(1−p)/C for a network of accuracy p. Across the population p(1−p) averages about 0.070.
- At each compute level C, the estimate's MSE averaged over the 1,024 networks must be at most the average of p(1−p)/C.
- In points (percentage points of accuracy), sampling's RMS error is 1 at 700 forwards, 0.3 at 7,800, 0.1 at 70,000, and 0.03 at 780,000.
- A forward pass is 2,880 multiply-adds; other arithmetic is charged at the same rate.

## Setup

- Network: h_t = ReLU(W_ih x_t + W_hh h_{t−1}), with h_0 = 0 and no biases, 16 hidden units, T = 10 steps. The logits W_o h_T give 10 outputs, and the answer is their argmax. The target is the position of the 2nd-largest input, and x is iid N(0,1). The map is positively homogeneous.
- `pop1k/seed0.npz`, keys `s1048576_Wih` (1024,16), `s1048576_Whh` (1024,16,16), `s1048576_Wo` (1024,10,16), plus `acc_2e20`, each network's accuracy from 2²⁰ samples. Accuracy is 0.924 ± 3.4 points across the population.
- Six further networks (`weights.npz`, `weights_seed0..4.npz`) serve as an out-of-population check.

## Terms

- **Top-k configuration.** The positions and values of the k largest inputs. For k = 4 there are 10·9·8·7 = 5,040 position **arrangements**, and the ordered values M > S > R > Q.
- **MFk anchor (mean field).** The top k inputs are kept and the other 10 − k are set to their conditional mean below the k-th largest, μ(v) = −φ(v)/Φ(v). I₀ is the network's correctness at that point.
- **Lower-input model.** P(correct | top-k configuration), which accounts for the 10 − k smaller inputs.
- **Outer integral.** The integral over the top-k values (in CDF coordinates of the order statistics) and the arrangements.
- **S-line.** The 2nd-largest value varied along its range with the others fixed. The network is piecewise linear along it, so it can be traced exactly.

## What is established (all in `REPORT.md` 7.16 and the scripts `z11`–`z37`)

**Lower inputs (k = 4).**
- **Additive in the margin.** The margin m = L_target − max_other L is expanded to first order in the 6 lower inputs. The terms of inputs at most 2 positions apart are projected onto one-input terms. P(m > 0) is then an exact convolution over a Gauss–Legendre node grid.
  - On all 1,024 networks this gives +0.03 points bias and RMS 0.048, with local error 0.003·p(1−p), at about 240–410 forwards per configuration (`z18`).
  - The first-order version alone is −0.18. The bias came from gate flips shared by nearby inputs being counted once per input, mostly in clock and memory neurons (`z15`–`z19`).
- **The additive family has a floor.** Its ideal, the exact first-order ANOVA projection of the margin, is itself +0.063 points biased, RMS 0.074 (`z34`). Adjacent-triple terms do not help (`z32`). A joint product rule over all six inputs is 0.15 points off at 729 forwards per configuration (`z36`).
- **The correction is not local to the boundary.** 68% of the lower-input correction sits on S-lines where the MF4 anchor never changes its answer, from rare large effects (`z22`).

**The outer integral (k = 4), anchor only.**
- **Tensor Gauss–Jacobi in CDF coordinates.** With exact S-lines charged at about 9,300 forwards per outer node (one MF4 arrangement tree), the error is 3.6, 1.27, 0.18, 0.18, 0.039, 0.031, 0.032, 0.026, 0.022 and 0.011 points at orders 1–10 (`z21`). Order n costs about 9,300·n³ forwards; order 10 is 9.3M. The MSE ratio to sampling is 160, 160, 11, 25, 2.4, 2.7, 4.7, 4.6, 4.7 and 1.6, so it converges at sampling's rate from order 5.
- **Other rules are worse:**
  - Smolyak sparse grids do not converge (`z28`).
  - An adaptive box rule is 40× sampling's MSE at 864 nodes (`z31`).
  - Homogeneous coordinates (scale integrated out, an exact angle, a 2-D outer integral) converge erratically even with their geometric kinks removed (`z23`, `z27`).
- **Arrangements cannot be thinned.**
  - Each arrangement's anchor accuracy has an SD of 8 points across arrangements.
  - Additive position effects explain 18% of it, and all pairwise position interactions 48%.
  - Fixed subsets of arrangements are 4–10× sampling's RMS (`z24`).

**Closed forms from linear regions.**
- **One exact linear map per activation pattern.** One arrangement's value space holds hundreds of linear regions (a median of 560 seen in 2¹⁶ draws). The top 100 cover 95% of the mass (`z25`, `z26`).
- **With a free oracle.** If an oracle chooses the K patterns per arrangement greedily, for agreement on 2¹⁴ draws, the anchor is 0.33, 0.17, 0.11 and 0.04 points off at K = 3, 10, 30 and 100. That is 1.4–1.6× sampling's RMS in 10⁴–10⁵ forwards, but the oracle itself costs about 80M forwards (`z30`).
- **Cheaper selections fail.** Taking the patterns found on the S-lines at the outer nodes is 13–24× sampling's RMS (`z33`). A least-violated-region fallback is worse than nearest centroid (`z35`).
- **Clock-state simplification.** Making the usually-active (clock and memory) neurons linear biases the accuracy by 2.3 points and leaves about 230 regions per arrangement (`z29`).

**Other k** (`z37`). Here deterministic methods do beat sampling, but only on the mean-field anchor.

| exact inputs | arrangements | anchor bias against the truth (RMS, points) | deterministic anchor integral | sampling at that cost |
|---|---|---|---|---|
| top 2 (MF2) | 90 | 5.7 | 0.16 at 573 forwards, 0.03 at 1.7k | 1.16, 0.67 |
| top 3 (MF3) | 720 | 0.68 | 0.13 at 24k, 0.074 at 55k, 0.073 at 98k | 0.18, 0.12, 0.089 |

- For MF3, the deterministic integral is 1.5–2.6× better than sampling in MSE.
- But the correction from the anchor to the true accuracy is a 7–8-dimensional integral over the smaller inputs. It must be modelled per configuration inside the arrangement enumeration: for MF3, at least 720 × (S-points) × (rows) forwards per outer node, about 41k even for one node with the cheapest first-order model, which is itself 0.156 points off.
- Sampling the correction on top of the anchor would be a control variate, which is excluded.
- **End to end** (`z38`): the MF3 outer rule with the first-order correction at every configuration is 5.8, 2.1 and 0.34 points off at 84k, 668k and 10.5M forwards, where sampling gives 0.096, 0.034 and 0.009.

## The question

1. **Is there a decomposition, or a description of the network, in which both the integral and the correction are cheap?**
   - Every route above either integrates the anchor better than sampling but leaves a large, expensive correction (MF2, MF3), or is accurate but costs 10–160× sampling's MSE in 10³–10⁵ forwards (MF4). The crux is a deterministic way to do the correction: the 6–8-dimensional, discontinuous, non-local integral over the smaller inputs.
   - One alternative frame has been tried: decomposing by events (new max, new 2nd max, neither), whose sequence has a product law. Its mean-field anchor (records exact, each "neither" input at its mean below the running 2nd max) is 1.9 points off the truth (RMS 2.0), far worse than the top-4 anchor (`z39`). Integrals along the trajectory of the running max and 2nd max, or in ages, are untried.
   - If you propose one, please measure it the way the table above does: error across networks against √(mean p(1−p)/C) at the method's own cost.
2. **If you think no deterministic method of the allowed kind can match sampling here, give the strongest argument you can.**
   - For example, a lower bound on the cost of any weight-based method in terms of something measurable, such as the number of linear regions with non-negligible mass, or the non-locality of the lower-input correction.
   - In the black-box (membership-oracle) model, deterministic volume estimation is provably weaker than randomized (Elekes 1986; Bárány–Füredi 1987). The networks are white-box, though, so that alone does not settle it.
3. **An honest verdict**, either way.

## Constraints

- **Mechanism, not statistics.** The estimate for a network must be a deterministic computation on that network's own weights.
  - Not allowed:
    - random or quasi-random point sets over the input or configuration space, including Sobol or lattice rules used in place of sampling;
    - control variates, and importance or stratified sampling;
    - any fitted prediction of accuracy or error from the population.
  - Allowed:
    - points the method chooses from the weights (region vertices, line pieces, roots, quadrature nodes);
    - design choices learned from other networks in the population, if they are fixed before the method is applied to a network.
- **Random sampling is fine for measuring**, that is, for scoring an estimate against the truth.
- **Please paste all code inline.** Sandbox files can expire before we download them. We will rerun everything on an RTX 2060 SUPER (8 GB) with 8 CPU cores.
- **Building blocks you can reuse:**
  - `batch_train.forward`: a forward pass batched over networks and inputs.
  - `z2.load('pop1k/seed0.npz')`: the weights as torch tensors.
  - `lines.trace_line`: the exact integral along one coordinate.
  - `z21.py` and `z37.py`: the outer-integral harness.
  - `z18.py`: the lower-input margin model.

## Files in the zip

- `REPORT.md`: section 7.16 has all of the above, with tables.
- `pop1k/`: the 1,024 networks and every script's printed output (`z*_seed0*.txt`).
- `z*.py`: the scripts named above.
- `chatgpt2/` and the `QUESTION*.md` files: background only.
