# A cheap learned accuracy estimator for a tiny RNN beats sampling at low budgets: review it, then find features and optimizations that widen the margin

The attached zip holds a research project on ARC's AlgZoo 2nd-argmax RNN, a 432-parameter ReLU RNN that outputs the position of the 2nd-largest of 10 Gaussian inputs. It includes two populations of 1,024 independently trained networks, the code, the write-up (`REPORT.md`) and every script's printed output.

This document stands alone. It defines the task, describes the estimator that currently works and what has been tried around it, and then asks for three things (§7):
1. **A review** of the estimator, its scoring and its cost accounting.
2. **A brainstorm of features** that might carry the remaining error, each with its cost and the reason it could help.
3. **Optimizations**: ways to get the same accuracy for less compute, or more accuracy for the same compute.

Code for the most promising ideas, and tests of them where the time allows, are very welcome.

**Where things stand.** A deterministic estimator, G(1, K), runs each network on a fixed design of 1,440·K inputs and maps cheap summaries of the outputs plus 20 weight statistics to accuracy, through a kernel ridge readout fitted on the population. With every operation on the target network counted, it beats sampling's mean squared error at 619–2,152 forward-equivalents on both populations (0.30–0.79× sampling's MSE, §5). The goal now is to widen that margin and, if cheap, to push the budget where sampling overtakes it (about 4k forward-equivalents) higher.

## 1. The goal and the score

- **ARC's challenge.** Produce a *mechanistic* estimate of the network's accuracy, one computed from the network's structure rather than from sampled examples, that "matches the performance of random sampling in terms of mean squared error versus compute".
- **The comparator.** Sampling a network of accuracy p with C forward passes has MSE p(1−p)/C.
- **The score is an average over independently trained networks.** ARC suggests two ways to measure it: cheaply, over copies of one model with noise added to its weights, or (better, in ARC's words) "over random seeds used to train the model". Here it is the second: an estimate's MSE over networks trained from different seeds, against the mean of p(1−p)/C over the same networks, at the estimate's total cost C.
  - Weight-noise copies of a model are **not** a target here. Please do not train or select on them.
- **"Points"** are percentage points of accuracy. **FE** (forward-equivalent) is the unit of cost: one forward pass of the target network on one input, counted as 2,880 multiply-adds.

**What counts.**
- **On the network being estimated**, the estimate must be a deterministic computation on its own weights.
  - It may evaluate the network at fixed inputs, or at inputs chosen from its weights, and compute any statistics of its weights.
  - It may not use random or quasi-random samples of the network's inputs, or sampling-based variance reduction (control variates, importance or stratified sampling). Sampling is the baseline.
  - **All arithmetic on the target network is charged at 2,880 multiply-adds per FE**: forward rows, feature computation, weight statistics and the readout's own inference.
- **On the population, anything goes.** Fitting, learning, calibrating and sampling the other networks are all allowed; that is what the populations are for.
  - Whatever is learned must be fixed before it is applied to a network not used to fit it. Report errors cross-fitted over the population, or on held-out networks.
  - Offline work is not charged, but please report it.
- **Each target's training recipe is known**, so each population gets its own readout. Transfer between recipes is not a goal.
- **Mechanism is welcome.** Features with a meaning in terms of how the networks compute (§3) are preferred, and when a learned component helps, please say what it has learned.

**The budget range.** The estimator wins below about 4k FE. Results in roughly 10²–10⁴ FE are what matter now. Please report every result with its budget.

## 2. Setup

**Architecture** (AlgZoo `DistRNN`).
- h_t = ReLU(W_ih·x_t + W_hh·h_{t−1}), with h_0 = 0 and no biases; logits = W_o·h_T; the answer is the argmax (ties go to the lowest index).
- H = 16 hidden units and T = 10 steps: W_ih is 16 × 1, W_hh 16 × 16 and W_o 10 × 16, so 432 parameters.

**Task.** x ~ N(0, I₁₀). The target is the position of the 2nd-largest input. Accuracy is P_x[argmax logits = target].

**The mystery model** (`weights.npz`, accuracy 95.30%) is ARC's showcase network, `example_2nd_argmax()`. Zihao Chen's notebook (`refs/RNN_2nd_Argmax_notebook.ipynb`) gives its training recipe, which differs from the rest of AlgZoo.

**Population 1: the mystery model's recipe** (primary; `pop_unif/`, trained by `batch_train_unif.py`).
- **Recipe:** Unif(−1, 1) training inputs, Adam (lr 0.005, weight decay 10⁻¹⁴), learning rate × 0.995 per epoch, 1,000 epochs of 100,000 fresh sequences, cross-entropy, PyTorch default initialization.
  - The notebook does not give the batch size. At batch 256 and above the mystery model beats every network trained; at batch 64–128 it sits at the 97th percentile, so 64–128 is the plausible range.
- **The 1,024 networks:** `pop_unif/b128_n1024_seed4.npz`, batch 128, final weights under the keys `s1000_{Wih,Whh,Wo}`. Training them took 3.3 hours on one RTX 2060 SUPER, so more can be trained.
- **Labels:** `pop_unif/b128_n1024_seed4.acc22.npy`, each network's accuracy on one shared set of 2²² N(0, 1) inputs. SE about 0.014 points per network.
- **Accuracy:** 90.56% ± 3.70 across networks (range 70.8–96.3); mean p(1−p) = 0.0842.
- **Further groups**, labelled on the same inputs: 64 more batch-128 networks (`b128_seed0`), 64 at batch 64 (`b64_seed5`), and 64 each at batch 256, 512 and 1,024.

**Population 2: AlgZoo's recipe** (`pop1k/seed0.npz`, trained by `batch_train.py`).
- **Recipe:** N(0, 1) inputs, AdamW with weight decay 10⁻³, 2³⁰ sequences. Weights under `s1048576_{Wih,Whh,Wo}`.
- **Labels:** `pop1k/reference.npz` (`acc`, on one shared set of 2²⁰ inputs; SE about 0.025 points). Rows 0–1023 are the population and rows 1024–1029 ARC's six released models (mystery, then zoo seeds 0–4).
- **Accuracy:** 92.4% ± 3.4; mean p(1−p) = 0.0691.
- **ARC's zoo seeds 0–4** (`weights_seed0..4.npz`, loaded by `seeds_clocks.load`) were trained with this recipe, so this population's readout is the one for them.

**Sampling's RMS error** at the costs that matter (mystery-recipe population, mean p(1−p) = 0.0842):

| FE | 100 | 300 | 619 | 1,130 | 2,152 | 4,196 | 10,000 |
|---|---|---|---|---|---|---|---|
| sampling RMS (points) | 2.90 | 1.68 | 1.17 | 0.86 | 0.63 | 0.45 | 0.29 |

Label noise (0.014–0.025 points) is negligible at these budgets.

## 3. How the networks compute

The mechanism is studied in `REPORT.md` §1–§3 and §7.9 (mystery model) and §7.16 (AlgZoo-recipe population).

- **An online, time-invariant 2nd-max tracker.** The network never stores *where* the max and 2nd max are; it stores *how long ago* they appeared.
  - In the mystery model the two ages are the phases of two damped rotating "clocks" turning at the same speed (about 43° per step), with amplitudes scaled by the values they track.
  - **Comparator** neurons test each new input against thresholds that blend the running max M and running 2nd max S. A new max restarts the max clock and hands its old phase to the 2nd-max clock (the **handoff**). A new 2nd max restarts the 2nd-max clock. Anything else lets both rotate.
- **The underlying design.** A sparse **write-pointer** neuron fires on (or one step after) each new max or 2nd max. The clocks are the eigenmodes of W_hh's impulse response to that pulse. In the mystery model (pointer neuron 5), W_o·W_hh^a·e₅ decodes every age a from the weights alone, and the readout is a Fourier-like decoder of the 2nd-max clock's phase.
- **Variants.** About 54% of AlgZoo-recipe networks have a write pointer (38% fire on the event, 17% one step late); the rest are "memory" networks that keep a decaying record of past inputs. The two err in similar places.
- **Roles per network** (AlgZoo recipe, averages): 6.0 always-on neurons (clock and memory channels), 3.1 comparators, 0.9 max detectors, 5.3 others.
- **A subcircuit common under the mystery recipe** (`z56`): an isolated pair (a, b) computing the running max and its last increment, plus "leave-one-out max" units fed by it. A rescaling-invariant template match finds it in the mystery model (units 2, 4, then 6, 7) and a close copy in 69% of mystery-recipe networks, against 50% under AlgZoo's recipe.
- **Where errors come from.** Blur near ties (M ≈ S, or S ≈ R, the 3rd largest), and failed handoffs, above all when the new max arrives at the last step. A last-step handoff fails 12% of the time on average (SD 13.5 across networks, correlation −0.81 with accuracy): 38% with no comparators, 3% with five or more.
- **Structure.**
  - With no biases, h(cx) = c·h(x) for c > 0, so every linear region is a cone through the origin.
  - The logit margin is a small difference of large activities: about 1–4 against activities of 20–90. The network cancels lower-input noise in that difference, which is why moment closures and polynomial propagation fail (§6).
  - **Exact symmetries.** For any positive diagonal D, (D·W_ih, D·W_hh·D⁻¹, W_o·D⁻¹) computes the same logits, and scaling W_o by c > 0 changes no answer.

## 4. The estimator G(1, K) (`gfeat.py`)

**Terms.**
- **Top-k configuration:** the positions and values of the k largest inputs, M > S > R > Q for k = 4.
- **MFk row (mean field):** the top k inputs kept at given values, and every other input set to its conditional mean below the k-th largest, μ(v) = E[x | x < v] = −φ(v)/Φ(v).
- **Leakage:** a mean-field anchor's accuracy minus the true accuracy. Integrated over all configurations, MF4's is +0.31 points on average (RMS 0.42) on AlgZoo-recipe networks, and MF3's +0.25 (RMS 0.57).

**The design (one "box", K S-slices).**
- **Cells:** the 720 ordered position triples (M, S, R) × K slices of S. Each network sees the same 720·K cells.
- **Values:** M, R and Q at the conditional medians of their order-statistic laws, which with one box are the same in every cell: M = 1.388, R = 0.577, Q = 0.372. S is at the midpoints of K equal slices of its conditional CDF, uniform between Φ(R) and Φ(M): 0.907 for K = 1; 0.732 and 1.115 for K = 2; 0.653, 0.816, 1.006 and 1.240 for K = 4.
- **Q's position** is one representative per arrangement (`gfeat.QREP`), by a balanced pairing on R's rank among the 8 free positions: (0,7), (7,0), (1,3), (3,1), (2,5), (5,2), (4,6), (6,4).
- **Two rows per cell:** the MF4 row (the other six inputs at μ(Q) = −0.577) and the MF3 row (the other seven at μ(R) = −0.470). So 1,440·K rows.

**Cost of the rows: a prefix trie.** The design is fixed, so its distinct input prefixes are known offline. On the target, W_ih·x is computed once per distinct scalar, W_hh·h once per distinct prefix state, and W_o·h once per row, plus the adds, ReLUs, argmax and margin. That costs 494·K + 69.4 FE: 563.4, 1,057.4 and 2,045.4 FE for K = 1, 2, 4. Against the full forward pass (`z48`), its margins agree to 6×10⁻⁴ and its bits differ in 5 of 5.9M rows at K = 4, all at near-zero margins, and in none at K = 1 or 2.

**Features per network.**
- **Per cell:** the correctness bits I4 and I3 of the two rows, and their margins m4 and m3 (target logit minus the best other logit).
- **Ten cell features:** I4, I3, a = m4/s, b = m3/s, |a|, |b|, a/(0.1+|a|), b/(0.1+|b|), a/(0.03+|a|), b/(0.03+|b|), where s is the RMS of m4 over the network's cells.
- **Aggregation:** each of the ten is averaged over all cells and over each S slice (for K = 1 the two coincide; the layout keeps both).
- **Context:** s, log s, mean m4 and mean m3.
- **20 weight statistics:** the mean, SD, min and max over the 16 hidden units of W_ih, the self-loop diag(W_hh), the log row norm and log column norm of W_hh, and the log column norm of W_o.
- That is 44, 54 and 74 features for K = 1, 2, 4.
- **The weight statistics and s are not invariant** under the exact symmetries of §3. They carry a fingerprint of the training recipe. That is useful within a recipe (see the ablation in §5) and harmless since the recipe is known.

**Readout.**
- Features are standardized on the training networks.
- The model is either ridge or RBF kernel ridge, exp(−γ‖Δ‖²/d) with γ ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1} and penalty α ∈ {10⁻⁵, …, 10³}. Family, γ and α are chosen by 4 inner folds inside each of 8 outer folds (folds by a random permutation).
- Networks outside the training population are scored by a model refitted on all of it.

**Total cost (`gfeat.cost`).** Rows + 40 scalar operations per cell for the features + a 5,000-operation allowance (weight statistics, standardization) + the RBF readout over n = 1,024 supports with d features (2·n·d + 34·n + 2·d + 3 operations), all divided by 2,880. That is 618.5, 1,129.7 and 2,151.9 FE.

## 5. Results

**Mystery recipe** (`z55`, `z58`; readout trained on the 1,024 batch-128 networks):

| K | total FE | cross-fit RMS on the 1,024 (MSE / sampling) | 64 more batch-128 networks: RMS (MSE / sampling) | 64 batch-64 networks: RMS, mean (MSE / sampling) | mystery model error |
|---|---|---|---|---|---|
| 1 | 618.5 | 0.95 (0.67) | 0.88 (0.56) | 1.27, +0.80 (1.20) | −0.55 |
| 2 | 1,129.7 | 0.65 (0.57) | 0.56 (0.42) | 0.93, +0.49 (1.17) | −0.37 |
| 4 | 2,151.9 | 0.56 (0.79) | 0.53 (0.71) | 0.64, +0.25 (1.04) | −0.07 |

Cross-fit RMS moves by about 0.01 between fold schemes (0.947 / 0.550 for K = 1 / 4 with folds over the population alone, `z59`).

**AlgZoo recipe** (`z49`, `z52`, `z55`; readout trained on its 1,024):

| design | total FE | cross-fit RMS (MSE / sampling) | zoo seeds 0–4 errors |
|---|---|---|---|
| G(1,1) | 618.5 | 0.58–0.61 (0.30–0.33) | +0.51 −0.04 −0.18 −0.16 +0.12 |
| G(1,2) | 1,129.7 | 0.49–0.51 (0.40–0.42) | −0.49 +0.48 −0.01 −0.41 −0.10 |
| G(1,4) | 2,151.9 | 0.43 (0.58) | −0.51 +0.81 −0.19 −0.64 −0.45 |
| G(1,8) | 4,196 | 0.36–0.38 (0.80–0.88) | |
| G(8,2), 8 outer boxes | 8,792 | 0.35–0.38 (1.6) | |
| G(125,8) | 514,896 | 0.20–0.21 (31) | |

**Other points measured on the AlgZoo recipe** (from an earlier external analysis in `chatgpt2/round6/`; not rechecked here):
- A population-learned subset of 16 cells with a linear readout: 0.95 points at about 28 FE. 64 cells with RBF: 0.91 at 237 FE.
- Lower-input "screening" rows (each lower input moved to its extreme quadrature nodes) on every 8th cell of K = 4: 0.38 at 3,558 FE (0.75×).

**Ablation** (`z49`, AlgZoo recipe, K = 1, cross-fit RMS in points):

| features | RMS |
|---|---|
| cells only | 1.36 |
| cells + context (s, means) | 1.12 |
| weights only | 1.17 |
| cells + weights | 0.87 |
| cells + weights in a canonical gauge (symmetries removed) | 1.05 |
| all (G) | 0.61 |

**Open puzzles.**
- **Why is the mystery-recipe fit weaker at K = 1** (0.67× sampling's MSE, against 0.30× under AlgZoo's recipe), while at K = 4 the two are close?
- **Where is the residual?** On AlgZoo-recipe networks a cell-level learned model showed the residual is network-level: a per-network rate at which the 6 smaller inputs break answers the anchors get right, concentrated in the extreme S slices (S next to R, S next to M) and shared across all position patterns. Two anchor rows per cell do not reveal it.

## 6. Tried, and did not help within a population

Please don't repeat these unless with a new angle.

- **Gauge-fixed weight statistics** (`z56`), in two gauges, replacing the raw ones. One balances each unit's incoming and outgoing norms; ARC's standardization sets |W_ih| = 1. Within ±0.05 points or worse.
- **Gauge-invariant readouts:** 1.12 at K = 1 on AlgZoo's recipe, against 0.61 (`chatgpt2/round6/`).
- **The subcircuit template** of §3 as 10 features (`z56`): no gain.
- **Covariance propagation** of N(0, I) through the network with an exact bivariate ReLU closure, 13 features, +167 FE (`z56`): at most −0.02 points, worse per FE. Moment closures as estimators fail outright (`z8`), because they break the margin's cancellation.
- **Hermite (low-degree polynomial) propagation** (`z57`): even an oracle projection at degree 4 gives 12–40% accuracy against 86–96%.
- **Individually selected screening rows**, chosen for their value in predicting G's residual: worse, at 16–64 rows (`chatgpt2/round6/`).
- **An exact integral over the last input** given the state after nine inputs, as features (`chatgpt2/round6/`, checked by `z53`): dominated by cheaper global summaries.
- **A cell-level MLP** (`z47`): its inference cost (about 9 FE per cell for a 128-128-64 network) outweighs its accuracy. Small ones reach 1.10–1.13× sampling's MSE.
- **One readout over both recipes:** worse on each than its own.
- **Neuron-role features** from 4,096 probe inputs: 1.0–1.1 points alone, and the probes alone cost 4,096 FE.
- **Every deterministic integration route** (`REPORT.md` §7.16: mean-field anchors with quadrature, lower-input models, closed forms over linear regions, region walks): 1.5–200× sampling's MSE at every budget.

## 7. What we would like

1. **Review (short, but complete).**
   - **Leakage:** standardization and hyperparameter choice inside the folds; the shared label inputs across networks; anything else.
   - **Cost:** does `gfeat.cost` count everything done on the target network (weight statistics, standardization, the exponentials of the RBF kernel)? Is the trie cost right?
   - **Scoring:** the sampling comparator at the total cost.
   - **The numbers in §5.**
2. **Features: a ranked brainstorm.** For each idea, give:
   - what it measures in terms of §3;
   - its cost in FE;
   - why it might carry the residual of §5;
   - a quick test, if possible.

   Some directions we have not tried, as a starting point only:
   - **Other aggregates of the cell outputs:** quantiles or histograms of margins, counts of near-zero margins, cells where I4 ≠ I3, means over mechanistically meaningful groups of arrangements (max at the last step, S last, M right after S, gaps between M, S and R).
   - **Mechanistic features from the weights alone,** which cost almost nothing:
     - W_hh's spectrum (the clocks' damping and rotation speed);
     - how well W_o·W_hh^a·W_ih decodes ages a;
     - comparator thresholds;
     - the write pointer's strength;
     - handoff margins computed from the weights.
   - **Sensitivities at the anchor rows:** derivatives of the margin with respect to the lower inputs, where one backward pass costs about one forward.
   - **A few designed rows off the anchors:** joint, temporally adjacent lower-input perturbations near the extreme S slices, aimed at the network-level breaking rate.
   - **Permutation-invariant encoders over the 16 hidden units** (set networks over per-unit weight descriptors).
3. **Optimizations.**
   - **Cost:**
     - fewer cells or a learned subset;
     - only MF4 or only MF3 rows;
     - different values per cell, at what cost to prefix sharing;
     - a cheaper readout: fewer supports, a low-rank or compiled kernel, or a linear model on good features.
   - **Accuracy:**
     - the readout model (Gaussian processes with per-feature length scales, boosting, ensembles, stacking across K);
     - feature selection;
     - target transforms such as the logit of accuracy;
     - using the extra batch-128 and batch-64 networks;
     - a learning curve in the number of training networks: would 2,048 or 4,096 networks help?
   - **Training more networks is easy:** 1,024 mystery-recipe networks take about 3.3 hours.
4. **Code** for the top ideas, runnable on the caches below, and results where the time allows.
5. **A short ranked plan:** the experiments most likely to lower the MSE ratio at 10²–10⁴ FE, and what each would show.

**Reporting.** For each method, give:
- forward rows, other arithmetic and the total, in FE;
- cross-fit RMS and mean error on the 1,024 mystery-recipe networks, and the MSE ratio against sampling at the total cost;
- the errors on the 64 further batch-128 networks, the 64 batch-64 networks and the mystery model;
- where relevant, the same on AlgZoo's recipe and zoo seeds 0–4.

Please say how each setting was chosen: nested inside the folds, or chosen on development networks and reported on held-out ones.

## 8. Files and building blocks

**No GPU needed.** The caches below hold every network's cell outputs and summaries.

- **`pop_unif/cells_k{1,2,4}.npz`** (`z59.py`) cover 1,153 mystery-recipe networks, in this order: the 1,024, then `b128_seed0` (64), `b64_seed5` (64), then the mystery model. Per network they hold:
  - `I4`, `I3` (bool) and `m4`, `m3` (float32), each of shape (N, 720·K), for cell c = a·K + k, where a indexes `arr` (the 720 (M, S, R) position triples, with Q at `qrep[a]`) and k the S slice;
  - `F`, G's summary features;
  - `stats`, the 20 weight statistics;
  - `acc` (labels), `group` and `net_fe`.

  G's feature matrix is `np.c_[stats, F]`.
- **`pop1k/seed0_z46_cells_b1_k{1,2,4}.npz`** has the same cell layout for the AlgZoo-recipe population plus ARC's six models (1,030 networks): `I4`, `I3`, `m4`, `m3`, plus `tgt`, each cell's true accuracy measured by sampling inside the cell.
- **`gfeat.py`:**
  - `rows(NB, K)`: the design;
  - `trie(x)`: the prefix trie;
  - `summary(W, NB, K, cells=True)`: runs any networks through the trie and returns the summaries, the row cost and the cells;
  - `stats`, `cost`;
  - the readout: `crossfit(X, y, idx, fold)` and `fit_predict(X, y, train, test)`.

  It runs on the CPU if there is no GPU.
- **Loaders and forward passes:**
  - `z2.load(path)` returns the weights of a population file as float32 torch tensors.
  - `seeds_clocks.load(path)` returns one released model's (W_ih, W_hh, W_o).
  - `batch_train.forward(Wih, Whh, Wo, x)` gives the logits batched over networks.
  - `common.py` has an exact NumPy forward pass and Jacobian (`run`, `run_jac`).
- **Scripts:**
  - `z55.py`: readouts trained on each recipe;
  - `z56.py`: the gauges, the template and covariance features;
  - `z57.py`: the Hermite oracle;
  - `z58.py`: batch 64;
  - `z49.py`: the ablation;
  - `z52.py`: larger designs.

  Printed outputs are in `pop_unif/z5*.txt` and `pop1k/z4*_seed0.txt` / `pop1k/z5*_seed0.txt`.
- **`REPORT.md` §7.17** covers all of §4–§6 in more detail, and **§9** is a ledger of every idea tried in the project. **`README.md`** has a script index.
- **`refs/`** holds ARC's AlgZoo post, Chen's notebook and write-up, and related notes.
- **Background only:** the other `QUESTION*_FOR_CHATGPT_PRO.md` files and `chatgpt2/`, from earlier phases. `REPORT.md` supersedes them where they differ.

**Compute.** We rerun everything on an RTX 2060 SUPER (8 GB) with 8 CPU cores and 31 GB of RAM; torch code is welcome. **Please paste all code inline**, because sandbox files can expire before we download them.
