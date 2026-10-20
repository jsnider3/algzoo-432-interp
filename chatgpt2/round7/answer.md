**The low-budget advantage survives the review, and I found substantial improvements.** The strongest changes are to use MF4 alone, add two cheap signed-pulse responses, and retain summaries that distinguish temporal arrangements.

On the primary population, the most useful new operating points are:

- **176 FE: 0.951 percentage points RMS**, with **0.189× sampling’s MSE**.
- **About 598 FE: 0.621 points RMS**, with **0.274× sampling’s MSE**.
- **651 FE: 0.606 points RMS**, with **0.284× sampling’s MSE**.
- **1,216 FE: 0.515 points RMS**, with **0.384× sampling’s MSE**.

The last result improves on the paired G(1,4) rerun—0.583 points at 2,152 FE—while using about **43% less compute**. These are exploratory, nested cross-fit results; the feature-development process has now examined these populations repeatedly. The exact numbers and their source files are recorded in [headline_summary.json](sandbox:/workspace/scratch/0aa5efd25118/deliverables/headline_summary.json).

I also found **two implementation discrepancies that matter to the advertised cost**. Both are fixed in the delivered estimator. I did **not** establish an improved crossover above 4,000 FE.

## Deliverables

- [ROUND7_REPORT.pdf](sandbox:/workspace/scratch/0aa5efd25118/deliverables/ROUND7_REPORT.pdf) — six-page review, results, feature analysis, and next experiments.
- [round7_estimator.py](sandbox:/workspace/scratch/0aa5efd25118/deliverables/round7_estimator.py) — complete standalone implementation, also pasted inline below.
- [algzoo_round7_results.zip](sandbox:/workspace/scratch/0aa5efd25118/deliverables/algzoo_round7_results.zip) — all experiment scripts, numerical results, per-network predictions, validation records, and a fitted 651-FE model. It includes a concatenated `ROUND7_ALL_CODE.txt` for the supplementary research scripts.

:codex-file-citation{path="sandbox:/workspace/scratch/0aa5efd25118/deliverables/ROUND7_REPORT.pdf" purpose="output"}

![Accuracy-estimation error against target compute for the two populations](sandbox:/workspace/scratch/0aa5efd25118/deliverables/round7_frontier.png)

# 1. Review: what holds up, and what needs fixing

### Validation and leakage

The basic statistical setup is sound: networks are the units of generalization, each recipe receives its own readout, and standardization, response centering, and readout selection use training networks only.

The shared label-input bank is **not direct leakage of a held-out network’s weights or label into its fit**. It does, however, correlate the reference-label integration errors across networks. Consequently, one should not treat those errors as independent noise and simply subtract their individual variances from the observed MSE. Their scale—roughly 0.014–0.026 points—is small relative to the residuals here. A second, independent reference bank would be useful for the final confirmation.

There is a reproducibility issue in `gfeat.pick`: the inner-fold construction uses `% len(y)`. Appending unused external networks can therefore change the inner folds and selected hyperparameters. I reproduced these primary-population G(1,4) results:

| Inner-fold context | RMS, points |
|---|---:|
| Historical `z55` layout | 0.5571 |
| Full 1,153-row layout | 0.5499 |
| Stable IDs based on the 1,024 training-population networks | 0.5831 |

Thus, the historical numerical claims are reproducible, but the fold sensitivity is sometimes closer to **0.03 points** than 0.01. All paired comparisons below use the stable 1,024-ID construction.

Hyperparameters are nested within each experiment. The later choice of feature combinations and designs was informed by earlier experiments on these same populations. That additional research selection is why these results should be confirmed on a fresh cohort.

### Cost accounting: two concrete fixes

**First, cache input products by scalar value.** The advertised trie ledger computes \(W_{ih}x\) once per distinct scalar. The supplied `summary` implementation recomputes it across prefix occurrences. Relative to the advertised budget, that adds approximately:

| K | Uncached extra input-product cost |
|---|---:|
| 1 | 33.38 FE |
| 2 | 62.93 FE |
| 4 | 122.03 FE |

The delivered implementation precomputes the scalar-input table and indexes it.

**Second, compile one selected readout before querying a target.** The evaluation helper applies all readout candidates to the target and performs target-dependent kernel–eigenvector products. At 1,024 supports, the six dense \(n\times n\) products alone cost approximately **2,184.5 FE per target**.

Those products belong offline. After selection, contract the eigensystem and ridge coefficients into one vector. A target then needs only its feature vector, kernel distances, exponentials, and one dot product. The delivered model does this for both full and Nyström kernels.

With those fixes, the advertised trie expression is consistent:

\[
C_{\rm rows}=
\frac{
256\sum_{t=1}^{9}n_t
+16v
+160n_{10}
+16\sum_{t=2}^{10}n_t
+16\sum_{t=1}^{10}n_t
+20n_{10}
}{2880},
\]

where \(n_t\) is the number of distinct length-\(t\) prefixes and \(v\) is the number of distinct input values.

The RBF ledger **does include exponentials**, through its \(34n\) allowance. The feature and weight-statistic allowances remain an operation-count convention, rather than a hardware timing measurement.

### Scoring

For a common target cost \(C\), the reported ratio is correctly

\[
\rho=
\frac{\frac1N\sum_i(\hat p_i-p_i)^2}
{\frac1N\sum_i p_i(1-p_i)/C}.
\]

If a future adaptive method has costs \(C_i\), the denominator should instead be

\[
\frac1N\sum_i\frac{p_i(1-p_i)}{C_i}.
\]

Do not replace that expression with a formula using only the mean cost.

I retained the conservative 1,024-support budget when scoring cross-fitted full-kernel models, even though each outer training fold has 896 networks. Correctness bits also remain separate from margins: with the stipulated tie rule, `margin > 0` cannot replace the exact correctness bit.

The detailed audit and implementation checks are included in the [results archive](sandbox:/workspace/scratch/0aa5efd25118/deliverables/algzoo_round7_results.zip).

# 2. Main primary-population results

Notation:

- **P:** ten features from two signed input-pulse trajectories.
- **H4:** temporal-group summaries using MF4 correctness and normalized soft margins.
- **128:** a Nyström readout with 128 training-population landmarks.
- **Full:** the full kernel, charged at 1,024 supports.
- **Compact:** all 90 ordered \((M,S)\) position pairs, but only the first and last available temporal positions for \(R\). This leaves 180 arrangements. It is a fixed structural design.

“Other FE” includes feature arithmetic, weight statistics, pulses where present, and the compiled readout. Anchor rows exclude the pulse trajectories. The two pulses produce **20 age-output evaluations in total**, with their complete extraction conservatively charged at six FE.

### Cross-fit accuracy and cost

| Method | Anchor rows | Anchor FE | Other FE | Total FE | RMS, points | Mean error, points | MSE / sampling |
|---|---:|---:|---:|---:|---:|---:|---:|
| G(1,1) | 1,440 | 563.40 | 55.15 | 618.55 | 0.9636 | +0.0007 | 0.6824 |
| G(1,2) | 2,880 | 1,057.40 | 72.26 | 1,129.66 | 0.6780 | +0.0171 | 0.6171 |
| G(1,4) | 5,760 | 2,045.40 | 106.50 | 2,151.90 | 0.5831 | +0.0098 | 0.8692 |
| Compact, K=1, P, 128 | 360 | 159.41 | 16.59 | **175.99** | **0.9508** | −0.0089 | **0.1891** |
| MF4, K=2, full | 1,440 | 566.90 | 51.87 | **618.77** | **0.6809** | +0.0073 | **0.3409** |
| MF4, K=2, P, 128 | 1,440 | 566.90 | 24.55 | **591.45** | **0.6506** | +0.0013 | **0.2975** |
| MF4, K=2, P, fixed H4, 128 | 1,440 | 566.90 | 31.34 | **598.24** | **0.6212** | +0.0058 | **0.2743** |
| MF4, K=2, P, tuned H4, full | 1,440 | 566.90 | 83.74 | **650.64** | **0.6060** | +0.0045 | **0.2839** |
| MF4, K=4, P, tuned H4, full | 2,880 | 1,095.90 | 120.60 | **1,216.50** | **0.5155** | +0.0026 | **0.3841** |

The table preserves the conservative experimental ledgers. The consolidated implementation reuses normalization and costs **650.62 FE** for the full K=2 model and **595.24 FE** for the fixed-H4/128 model. This changes the cost slightly, without changing their features or predictions.

The most revealing ablation is **MF4-only K=2**: it gives essentially the paired G(1,2) accuracy at almost half the compute. At approximately the G(1,1) budget, spending the rows on two MF4 slices is much more informative than spending them on an MF3/MF4 pair at one slice.

### External networks

Each batch column reports **RMS / mean error**, in percentage points.

| Method | Extra batch-128 networks | Batch-64 networks | Mystery error |
|---|---:|---:|---:|
| G(1,1) | 0.875 / +0.143 | 1.269 / +0.804 | −0.553 |
| G(1,2) | 0.563 / +0.093 | 0.929 / +0.490 | −0.367 |
| G(1,4) | 0.529 / +0.080 | 0.635 / +0.254 | −0.066 |
| Compact K=1, P, 128 | 0.851 / +0.194 | 1.095 / +0.673 | −0.889 |
| MF4 K=2, full | 0.597 / +0.047 | 1.011 / +0.509 | −0.290 |
| MF4 K=2, P, 128 | 0.613 / +0.061 | 0.925 / +0.498 | −0.390 |
| MF4 K=2, P, fixed H4, 128 | 0.538 / +0.009 | 0.854 / +0.432 | −0.247 |
| MF4 K=2, P, tuned H4, full | **0.543 / +0.032** | **0.831 / +0.424** | −0.445 |
| MF4 K=4, P, tuned H4, full | **0.483 / −0.003** | **0.585 / +0.234** | ≈0.000 |

For the 651-FE model, the external MSE ratios are **0.227** on batch 128 and **0.539** on batch 64. For the 1,216-FE model they are **0.336** and **0.500**, respectively.

The near-zero mystery error of the last model is a fortunate match to this reference label; it is far below the label’s integration uncertainty and should not drive model selection.

There is one protocol distinction in these external results. The original/full temporal experiments select the final refit using index-mod-8 CV; the compact/128 experiments use candidate losses from the fixed outer folds. Both use only the 1,024 training networks. The standalone program exposes this distinction as `--refit index` versus `--refit outer`.

# 3. Ranked features: what they measure and what happened

## 1. Signed input-pulse responses — best cheap addition

Start from

\[
h_0^+=\operatorname{ReLU}(W_{ih}),\qquad
h_0^-=\operatorname{ReLU}(-W_{ih}),
\]

and propagate nine zero-input steps:

\[
h_{a+1}^{\pm}=\operatorname{ReLU}(W_{hh}h_a^\pm),\qquad
L_a^\pm=W_oh_a^\pm.
\]

For each sign, summarize:

1. Mean normalized margin for the pulse’s age-aligned output location.
2. Fraction of ages whose winning output matches that location.
3. Mean normalized margin at ages 5–9.
4. Mean change in log centered-output amplitude.
5. Standard deviation of that change.

**Mechanistic meaning.** This tests whether a reachable input-induced state preserves a decodable temporal trace, including actual ReLU gates. It measures retention, decoding quality, and irregular temporal evolution.

The age-alignment score is **not second-argmax correctness**: a zero-padded pulse sequence contains tied inputs. It is a diagnostic of the network’s temporal representation.

**Cost:** six FE for extraction; ten more coordinates add about 7.12 FE to a 1,024-support kernel.

On the primary population, adding P to G(1,1) improves RMS from **0.9636 to 0.8411** at **631.66 FE**, reducing MSE by about **24%**. Its mean error is +0.0061 points. External errors are:

- Batch 128: **0.808 RMS, +0.214 mean**.
- Batch 64: **1.032 RMS, +0.451 mean**.
- Mystery: **−0.416 points**.

The benefit becomes smaller for primary G(1,4): 0.5831 to 0.5765 at 2,165 FE. That small improvement is weak relative to the variability in the paired squared-error differences.

The normalized features are invariant to positive hidden-unit gauge transformations and positive output scaling when the numerical floors are inactive. The implementation checks confirmed this to numerical precision.

## 2. Temporal-group summaries — strongest inexpensive residual information

For each S slice, retain means over six groups:

- M is last.
- S is last.
- M immediately follows S.
- M occurs after S.
- R occurs after S.
- M and S are at least five positions apart.

For H4, the channels are MF4 correctness and its normalized soft margin. This adds **\(12K\) coordinates**.

**Mechanistic meaning.** These groups separate late handoffs, direct writes, subsequent distractors, and long retention intervals. A single global mean can hide which of those computations is failing.

**Extraction cost:** approximately **\(0.821K\) FE**, reusing the base normalization and softsign. The additional kernel coordinates cost about **\(8.54K\) FE** with 1,024 supports.

For the full kernel, I nested the additional block’s weight over \(\{0,0.1,1\}\), alongside width and ridge penalty. The metric is

\[
D(x,x')=
\frac{\lVert x_b-x_b'\rVert^2}{d_b}
+\lambda\frac{\lVert x_H-x_H'\rVert^2}{d_H}.
\]

This produces the **0.6060 at 651 FE** and **0.5155 at 1,216 FE** primary results above.

There is an important recipe-specific failure below; temporal summaries should not be appended indiscriminately.

## 3. Anchor-margin sensitivities — informative, but more expensive

I differentiated the target-minus-largest-rival margin through the actual gates on a fixed panel of **68 MF4 rows at K=4**. The panel covers extreme S slices and temporal patterns associated with late handoffs.

Let \(g_j=\partial m/\partial x_j\) for the six lower-input positions. Summaries include individual sensitivity, coherent changes, adjacent directions, and margin relative to a local perturbation scale. For example,

\[
v_{\rm local}=\sigma_Q^2\sum_{j\in L}g_j^2
\]

measures the variance predicted by a local linear approximation.

**Mechanistic meaning.** These features ask whether the network cancels lower-input variation in the decision margin, and which temporal directions threaten that cancellation.

They are features, not certified error probabilities: gates and the winning rival can change away from the anchor.

**Cost:** the tested standalone implementation adds **150.89 FE**, including repeated forwards. Retaining the necessary forward gates would reduce the addition to **77.51 FE**.

At K=4:

| Recipe | Baseline RMS | With gradients | Tested total FE | MSE / sampling |
|---|---:|---:|---:|---:|
| Primary | 0.5831 | **0.5163** | 2,302.79 | 0.7293 |
| AlgZoo | 0.4345 | **0.4056** | 2,302.79 | 0.5483 |

The primary mean error is +0.0020 points. External primary errors are 0.503 / +0.069 on batch 128, 0.588 / +0.183 on batch 64, and −0.151 on the mystery model.

The derivatives passed Euler-homogeneity checks and 192 deterministic central-difference checks. K=1 and K=2 did not give attractive gains per FE.

## 4. Margin tails and MF3/MF4 disagreement — useful mainly at K=4

I tested normalized-margin CDF counts around zero, together with the magnitude of the MF3/MF4 margin change and correctness disagreement.

**Mechanistic meaning.** These expose a fragile tail that can be hidden by average margins.

**Cost:** \(15K\) added coordinates, about \(8K\) FE for extraction, plus approximately \(10.68K\) FE for a full-kernel query.

For primary G(1,4), tails reduce RMS from 0.5831 to **0.5285 at about 2,227 FE**. Combining tails and temporal groups reaches **0.5125 at about 2,302 FE**. At K=1, tails do not help.

That pattern supports the idea that additional S slices reveal relevant relative-gap structure. It does not establish that near ties explain the whole residual.

## 5. Joint lower-input perturbations — next new information source

A small fixed panel of adjacent or coherent lower-input perturbations could reveal gate and rival switches that a derivative cannot.

I would start with **32–64 rows**, selected offline within training folds and concentrated on extreme S slices and late handoffs. A preliminary budget is roughly **30–70 FE before prefix sharing**, plus summaries and readout coordinates.

This was **not tested** in this round. It has a clearer route to genuinely new information than simply increasing K again.

## 6. A small hidden-unit encoder — plausible, lower priority

A shared encoder over 16 units could combine input, readout, recurrence, and finite-horizon response descriptors, then pool them into a compact representation.

A deliberately small design could fit within approximately **5–20 FE**, but it should include relational information from \(W_{hh}\); pooling unordered scalar unit descriptors alone loses much of the circuit. With only 1,024 training networks, model capacity and validation would need care. This was not tested.

### Lower priority: bare spectrum and arbitrary hidden-state impulses

The raw \(W_{hh}\) spectrum omits changing ReLU gates. Likewise, an arbitrary hidden basis-vector injection may test states that ordinary inputs do not reach.

The tested basis-impulse summaries worsened primary RMS to **0.9739 at 664.93 FE** for K=1 and **0.6818 at 1,176.05 FE** for K=2. I would prioritize the finite signed-input responses over these alternatives.

# 4. What changes on the AlgZoo recipe

The cheaper designs also work on AlgZoo, but the most accurate feature combinations differ.

| Method | Anchor rows | Total FE | RMS, points | Mean error, points | MSE / sampling |
|---|---:|---:|---:|---:|---:|
| G(1,1) | 1,440 | 618.55 | 0.5967 | +0.0090 | 0.3187 |
| Compact K=1, P, 128 | 360 | **175.99** | **0.6338** | +0.0032 | **0.1023** |
| MF4 K=2, P, 128 | 1,440 | **591.45** | **0.5476** | +0.0075 | **0.2566** |
| MF4 K=4, P, 128 | 2,880 | **1,132.34** | **0.4998** | +0.0105 | **0.4094** |
| G(1,4) | 5,760 | 2,151.90 | 0.4345 | +0.0057 | 0.5879 |
| G(1,4) + P | 5,760 | **2,165.02** | **0.4052** | +0.0004 | **0.5144** |

The corresponding released seed errors are:

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| Compact K=1, P, 128 | +0.738 | +0.162 | −0.384 | −0.070 | +0.272 |
| MF4 K=2, P, 128 | −0.182 | +0.436 | +0.027 | −0.262 | +0.176 |
| MF4 K=4, P, 128 | −0.218 | +0.407 | −0.213 | −0.165 | −0.479 |
| G(1,4) + P | −0.264 | +0.610 | −0.435 | −0.331 | −0.208 |

### The failure that should guide readout development

At AlgZoo K=1, adding temporal groups increased RMS from **0.5967 to 0.7177**. Network 994, whose reference accuracy is **78.605%**, accounts for essentially all the added MSE: its error rises from **+4.210 to +13.694 points**. It remains in every reported score.

Appending distances to an RBF kernel multiplies similarities. An unusual temporal signature can therefore remove a poor network’s useful neighbors and pull its prediction toward the population mean.

This motivates an **additive kernel**,

\[
k=k_{\rm base}+\eta k_{\rm mechanism},
\]

or a carefully cross-fitted residual correction. Either can add a mechanistic signal while preserving the base predictor’s information. Neither was tested here.

The primary K=1 gap also cannot be explained merely by some anchor values exceeding the Unif training interval. Positive homogeneity allows every input direction to be scaled into that cube. The recipes differ in directional weighting, optimization, and learned mechanisms; amplitude support alone is insufficient.

# 5. Optimizations and the remaining limits

### Fewer anchors and fewer supports give the largest savings

The compact design preserves every ordered \((M,S)\) pair. That temporal coverage appears valuable: its K=1 accuracy nearly matches G(1,1) at approximately **one quarter of the rows and 28% of the compute**.

For further cell selection, optimize the **incremental cost of the union of trie prefixes**. Two candidate rows can have very different marginal costs despite each counting as one row.

Nyström compression is also effective. All whitening and coefficient contraction happen offline, so the target sees only 128 supports. However, compression is not uniformly free: full-G AlgZoo K=2 worsens from approximately **0.515 to 0.567 RMS** with the tested 128-support approximation. Supports, features, and anchor design should be selected jointly against the total FE budget.

MF3-only variants were tested, but MF4 was the more useful primary-population allocation.

### Exact, inexpensive implementation improvements remain

At K=1, ten global and per-slice means are duplicates. They can be merged **without changing the RBF metric** by weighting each retained duplicate coordinate by \(\sqrt2\) and retaining the original gamma denominator. This saves about **7.1 FE** with 1,024 supports. Simply deleting one copy changes the kernel.

Clipping a final estimate to \([0,1]\) cannot increase squared error against an accuracy in that interval. Logit or error-rate transforms are less automatic: select them using squared error **after inverse transformation**, because that is the actual objective. These transforms were not tested.

For richer readouts, I would test a few block-specific length scales before one length scale per feature. Small boosted models are also plausible cheap readouts, but their accuracy needs to be compared at their compiled inference cost. Stacking across K must charge the union of required anchors and produce its training predictions out of fold.

### More S slices did not extend the crossover

The larger deterministic designs gave these negative results:

| Method | Total FE | Primary RMS | Primary MSE / sampling |
|---|---:|---:|---:|
| MF4 K=8, P, 128 | 2,214.14 | 0.5619 | 0.8306 |
| MF4 K=16, P, 128 | 4,377.72 | 0.5731 | 1.7086 |
| MF4 K=16, P, fixed H4, 128 | 4,432.05 | 0.5214 | 1.4316 |

AlgZoo MF4 K=16 with P and 128 supports reaches **0.4257 RMS at 4,377.72 FE**, or **1.148× sampling’s MSE**.

These tests reject those particular high-budget designs as improvements. They do not identify whether the remaining limitation is the features, the restricted readout, or a broader estimation floor.

### Would 2,048 or 4,096 training networks help?

I measured a learning curve for the fixed **MF4 K=2 + P, 128-support** estimator, keeping target cost at **591.45 FE**. Every row scores all 1,024 networks out of fold.

| Networks in each outer training set | Cross-fit RMS | Mean error | MSE / sampling |
|---:|---:|---:|---:|
| 224 | 0.9019 | −0.0568 | 0.5717 |
| 448 | 0.7721 | −0.0049 | 0.4189 |
| 672 | 0.6512 | −0.0055 | 0.2980 |
| 896 | 0.6506 | +0.0013 | 0.2975 |

The last interval is flat in this nested subset sequence. It does **not** justify a power-law forecast to 4,096 networks. The extra batch-128 test improves somewhat with the final increase in training size, so it also does not prove that more data is useless.

I would repeat the learning curve for the richer temporal model and several nested subset orders before training thousands more networks. Once the validation design is frozen, the extra batch-128 networks can join the final fit. Batch-64 networks should receive explicit recipe conditioning or a separate calibration rather than being pooled automatically.

# 6. Ranked next experiment plan

1. **Freeze and confirm three operating points:** compact + P at 176 FE, MF4 + P + H4 around 600–650 FE, and the full K=4 version at 1,216 FE. Use a fresh batch-128 seed cohort and an independent reference-input bank. This separates reproducible improvement from accumulated research selection.

2. **Optimize the probe budget using trie costs.** Select additional R/Q placements or S slices inside training folds, measuring MSE reduction per additional prefix operation. Preserve the useful coverage of all \((M,S)\) pairs.

3. **Add a small nonlinear perturbation panel.** Use the gradient diagnostics to define a fixed family of joint lower-input perturbations around late handoffs and extreme S slices. This tests whether gate switches carry the missing network-level breaking rate.

4. **Test additive kernels or proper out-of-fold residual correction.** Track the worst networks as well as overall MSE. Compare 64, 128, and 256 supports at fixed total budgets. The AlgZoo network-994 failure makes this a concrete priority.

5. **Revisit population size after choosing the representation.** If several learning curves still fall, train 2,048 before 4,096 networks. Otherwise, invest in probes that expose new behavior.

# 7. Reproduction and complete standalone code

No new RNNs were trained, no GPU was needed, and no random or quasi-random target inputs or weight-noise copies were used. Population-level randomness was limited to permitted offline fold and landmark selection.

Representative measured offline times in this environment were approximately **91 seconds** for the default full-kernel fit, **2–4 seconds** for 128-support fits, and **6.8 seconds** for the four-setting learning curve. These are offline wall times, separate from the target FE ledger.

The implementation was checked against the original caches and independent computations:

- Compiled full and Nyström readouts agree with their selected candidate predictions to about \(10^{-11}\) or better.
- Nine direct float64 forward comparisons matched the optimized prefix logits and argmaxes exactly.
- Five target-weight queries had no correctness-bit disagreements with the supplied caches; float32/float64 margin differences produced accuracy-estimate differences below the reported error scale.

Run these commands after saving the code below as `round7_estimator.py`. Replace `PROJECT` with the extracted `algzoo-432-interp` directory.

```bash
# Default: primary recipe, MF4 K=2, pulses, tuned temporal block, full kernel.
python round7_estimator.py --root PROJECT --out mf4_k2

# Approximately 1,216 FE.
python round7_estimator.py --root PROJECT --K 4 --out mf4_k4

# Approximately 176 FE.
python round7_estimator.py --root PROJECT --K 1 --kind both \
  --subset ends --groups none --supports 128 --refit outer --out compact

# Fixed temporal block and 128 supports.
python round7_estimator.py --root PROJECT --K 2 --groups fixed \
  --supports 128 --refit outer --out mf4_k2_small

# AlgZoo recipe, approximately 591 FE.
python round7_estimator.py --root PROJECT --recipe zoo --K 2 \
  --groups none --supports 128 --refit outer --out zoo_k2

# Evaluate a target directly; no fitting happens in this command.
python round7_estimator.py --predict mf4_k2.model.npz \
  --weights PROJECT/weights.npz
```

The following is the **entire main implementation**, with no dependency on another new script. It covers the original estimator, MF4/MF3 ablations, compact designs, pulses, temporal groups, full and compressed readouts, nested evaluation, deterministic cache generation, model serialization, and actual target-weight inference.

```python
"""Round 7 deterministic accuracy estimator; NumPy, SciPy, threadpoolctl only.

Run from the extracted project (or pass --root PROJECT):
  python round7_estimator.py --out mf4_k2
  python round7_estimator.py --K 4 --out mf4_k4
  python round7_estimator.py --K 1 --kind both --subset ends --groups none \
      --supports 128 --refit outer --out compact
  python round7_estimator.py --kind both --no-pulse --groups none --out G1 --K 1
  python round7_estimator.py --K 16 --groups fixed --supports 128 \
      --refit outer --generate --out mf4_k16
  python round7_estimator.py --predict mf4_k2.model.npz --weights weights.npz

Fit writes a portable model, cross-fit/external predictions, and JSON metrics.
Only the first 1024 labels enter fitting. New deterministic cell caches can be
made with --generate, or supplied with --cells. No input sampling is used.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import betaincinv, ndtr, ndtri
from threadpoolctl import threadpool_limits

GAM = [.003, .01, .03, .1, .3, 1.]
ALP = np.logspace(-5, 3, 9)
ARR = np.array(list(itertools.permutations(range(10), 3)))
FREE = [[i for i in range(10) if i not in a[:2]] for a in ARR]
RANK = np.array([f.index(a[2]) for f, a in zip(FREE, ARR)])
PAIR = {0: 7, 7: 0, 1: 3, 3: 1, 2: 5, 5: 2, 4: 6, 6: 4}
QREP = np.array([f[PAIR[r]] for f, r in zip(FREE, RANK)])
FOLD = np.random.default_rng(0).permutation(1024) % 8


def read_weights(path):
    with np.load(path, allow_pickle=False) as z:
        if 'rnn.weight_ih_l0' in z:
            return [z['rnn.weight_ih_l0'][:, 0].astype(float),
                    z['rnn.weight_hh_l0'].astype(float), z['linear.weight'].astype(float)]
        if 'Wih' in z:
            return [z[k].astype(float) for k in ('Wih', 'Whh', 'Wo')]
        s = max(int(k[1:-4]) for k in z.files if k.startswith('s') and k.endswith('_Wih'))
        return [z[f's{s}_{k}'].astype(float) for k in ('Wih', 'Whh', 'Wo')]


def population(root, recipe):
    if recipe == 'unif':
        paths = ['b128_n1024_seed4', 'b128_seed0', 'b64_seed5']
        groups = [read_weights(root/'pop_unif'/f'{p}.npz') for p in paths]
        groups += [[w[None] for w in read_weights(root/'weights.npz')]]
        y = np.concatenate([np.load(root/'pop_unif'/f'{p}.acc22.npy') for p in paths]
                           + [np.array([.95304775])])
    else:
        groups = [read_weights(root/'pop1k/seed0.npz')]
        groups += [[w[None] for w in read_weights(root/p)]
                   for p in ['weights.npz']+[f'weights_seed{i}.npz' for i in range(5)]]
        y = np.load(root/'pop1k/reference.npz')['acc']
    return [np.concatenate([w[i] for w in groups]) for i in range(3)], y.astype(float)


def weight_stats(wi, wh, wo):
    f = np.stack([wi, np.diagonal(wh, axis1=1, axis2=2),
                  np.log(np.linalg.norm(wh, axis=2)+1e-9),
                  np.log(np.linalg.norm(wh, axis=1)+1e-9),
                  np.log(np.linalg.norm(wo, axis=1)+1e-9)], 2)
    return np.concatenate([f.mean(1), f.std(1), f.min(1), f.max(1)], 1)


def pulse_features(wi, wh, wo):
    """Two signed impulse responses, all ages; ten features, budget six FE."""
    h = np.maximum(np.stack([wi, -wi], 2), 0)
    logits = []
    for age in range(10):
        logits.append(wo@h)
        if age < 9:
            h = np.maximum(wh@h, 0)
    lg = np.stack(logits, 1)
    centered = lg-lg.mean(2, keepdims=True)
    amp = np.maximum(np.sqrt((centered*centered).mean(2)), 1e-12)
    age, target = np.arange(10), 9-np.arange(10)
    correct = lg.argmax(2) == target[None, :, None]
    chosen = lg[:, age, target, :].copy()
    lg[:, age, target, :] = -np.inf
    margin = (chosen-lg.max(2))/amp
    growth = np.diff(np.log(amp), axis=1)
    return np.concatenate([margin.mean(1), correct.mean(1), margin[:, 5:].mean(1),
                           growth.mean(1), growth.std(1)], 1)


def arrangement_ids(spec):
    return (np.arange(720) if spec['subset'] == 'all'
            else np.flatnonzero(np.isin(RANK, [0, 7])))


def temporal_masks(ar):
    M, S, R = ARR[ar].T
    return [M == 9, S == 9, M == S+1, M > S, R > S, abs(M-S) >= 5]


def make_plan(spec, ar):
    """All geometry, input values, and trie bookkeeping are prepared offline."""
    k, kind = spec['K'], spec['kind']
    fq = betaincinv(7, 4, .5)
    fr = fq+(1-fq)*betaincinv(1, 3, .5)
    fm = fr+(1-fr)*betaincinv(2, 1, .5)
    q, r, m = ndtri([fq, fr, fm])
    ss = ndtri(fr+(fm-fr)*(np.arange(k)+.5)/k)
    mu = lambda v: -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
    types = [4, 3] if kind == 'both' else [int(kind)]
    rows, target = [], []
    for a in ar:
        for s in ss:
            for typ in types:
                x = np.full(10, mu(q if typ == 4 else r))
                x[ARR[a]] = [m, s, r]
                if typ == 4:
                    x[QREP[a]] = q
                rows.append(x)
                target.append(ARR[a, 1])
    x = np.array(rows, dtype=np.float32)
    vals, tok = np.unique(x, return_inverse=True)
    tok = tok.reshape(x.shape)
    par = np.zeros(len(x), int)
    plan = {'values': vals, 'target': np.array(target), 'ar': ar}
    for t in range(10):
        key, par = np.unique(par*len(vals)+tok[:, t], return_inverse=True)
        plan[f'parent{t}'], plan[f'token{t}'] = key//len(vals), key % len(vals)
    plan['leaves'] = par
    plan.update({f'group{i}': mask for i, mask in enumerate(temporal_masks(ar))})
    return plan


def prefix_cells(w, plan, dtype=np.float64):
    """Batched targets; input products cached once per distinct scalar."""
    wi, wh, wo = [np.asarray(a, dtype=dtype) for a in w]
    inp = wi[:, None, :]*plan['values'].astype(dtype)[None, :, None]
    h = None
    for t in range(10):
        par, tok = plan[f'parent{t}'], plan[f'token{t}']
        h = (np.maximum(inp[:, tok], 0) if h is None
             else np.maximum((h@wh.transpose(0, 2, 1))[:, par]+inp[:, tok], 0))
    lg = (h@wo.transpose(0, 2, 1))[:, plan['leaves']]
    target = plan['target']
    bits = lg.argmax(2) == target[None]
    chosen = lg[:, np.arange(len(target)), target].copy()
    lg[:, np.arange(len(target)), target] = -np.inf
    return bits, chosen-lg.max(2)


def summarize(bits, margins, stats, spec, ar, pulse=None, masks=None):
    """One shared normalization/softsign pass; H requires no extra network rows."""
    n, k = len(bits), spec['K']
    nr = 2 if spec['kind'] == 'both' else 1
    b, m = bits.reshape(n, -1, nr), margins.reshape(n, -1, nr).astype(float)
    s = np.maximum(np.sqrt((m[:, :, 0]**2).mean(1, keepdims=True)), 1e-15)
    a = m/s[:, :, None]
    aa = abs(a)
    soft = a/(.1+aa)
    f = np.concatenate([b, a, aa, soft, a/(.03+aa)], 2)
    base = np.c_[stats, s, np.log(s), m.mean(1), f.mean(1),
                 f.reshape(n, len(ar), k, -1).mean(1).reshape(n, -1)]
    if pulse is not None:
        base = np.c_[base, pulse]
    if spec['groups'] == 'none':
        return base, base.shape[1]
    masks = temporal_masks(ar) if masks is None else masks
    v = np.concatenate([b, soft], 2).reshape(n, len(ar), k, 2*nr)
    groups = np.concatenate([v[:, mask].mean(1).reshape(n, -1)
                             for mask in masks], 1)
    return np.c_[base, groups], base.shape[1]


def cached_features(root, spec, cells_path=None, generate=False, cache_out=None):
    w, y = population(root, spec['recipe'])
    ar, k = arrangement_ids(spec), spec['K']
    default = (root/f'pop_unif/cells_k{k}.npz' if spec['recipe'] == 'unif'
               else root/f'pop1k/seed0_z46_cells_b1_k{k}.npz')
    path = Path(cells_path) if cells_path else default
    types = [4, 3] if spec['kind'] == 'both' else [int(spec['kind'])]
    if path.exists():
        z = dict(np.load(path, allow_pickle=False))
    else:
        if not generate:
            raise FileNotFoundError(f'{path}; use --cells PATH or --generate')
        plan = make_plan(spec, np.arange(720))
        rows = [prefix_cells([a[i:i+24] for a in w], plan, np.float32)
                for i in range(0, len(y), 24)]
        b, m = [np.concatenate([r[j] for r in rows]) for j in range(2)]
        z = {f'{prefix}{typ}': value[:, j::len(types)]
             for j, typ in enumerate(types)
             for prefix, value in [('I', b), ('m', m)]}
        np.savez_compressed(cache_out, **z)
    for typ in types:
        for prefix in ('I', 'm'):
            if z[f'{prefix}{typ}'].shape != (len(y), 720*k):
                shape = z[f'{prefix}{typ}'].shape
                raise ValueError(
                    f'Wrong recipe/K or incomplete cache: {prefix}{typ} has shape {shape}')
    ci = (ar[:, None]*k+np.arange(k)).ravel()
    bits = np.stack([z[f'I{t}'][:, ci] for t in types], 2).reshape(len(y), -1)
    margins = np.stack([z[f'm{t}'][:, ci] for t in types], 2).reshape(len(y), -1)
    p = pulse_features(*w) if spec['pulse'] else None
    x, db = summarize(bits, margins, weight_stats(*w), spec, ar, p)
    return x, y, db, make_plan(spec, ar)


def candidates(spec):
    out = [('linear', 0., 0., float(a)) for a in ALP]
    weights = [0., .1, 1.] if spec['groups'] == 'tune' else [0.]
    return out + [('rbf', w, g, float(a))
                  for w in weights for g in GAM for a in ALP]


def metric(a, b, db, mode, weight):
    if mode != 'tune':
        return cdist(a, b, 'sqeuclidean')/a.shape[1]
    return (cdist(a[:, :db], b[:, :db], 'sqeuclidean')/db
            + weight*cdist(a[:, db:], b[:, db:], 'sqeuclidean')/(a.shape[1]-db))


def kernel_fit(a, y, db, spec, weight, gamma):
    """Offline eigensystems; Nystrom uses only a fixed subset of training rows."""
    ids = (np.random.default_rng(1701).permutation(len(a))
           [:min(spec['supports'], len(a))]
           if spec['supports'] else np.arange(len(a)))
    centers = a[ids]
    km = np.exp(-gamma*metric(centers, centers, db, spec['groups'], weight))
    e, v = eigh(km)
    if not spec['supports']:
        return centers, v@((v.T@y)[:, None]/(np.maximum(e, 0)[:, None]+ALP))
    keep = e > max(1e-12, e[-1]*1e-12)
    whitening = v[:, keep]/np.sqrt(e[keep])
    phi = np.exp(-gamma*metric(a, centers, db, spec['groups'], weight))@whitening
    e, v = eigh(phi.T@phi)
    coef = whitening@v@(
        (v.T@(phi.T@y))[:, None]/(np.maximum(e, 0)[:, None]+ALP))
    return centers, coef


def candidate_predictions(x, y, tr, te, db, spec):
    mean, sd = x[tr].mean(0), np.maximum(x[tr].std(0), 1e-6)
    a, b = (x[tr]-mean)/sd, (x[te]-mean)/sd
    ym = y[tr].mean()
    yc = y[tr]-ym
    dl = db if spec['groups'] == 'tune' else x.shape[1]
    e, v = eigh(a[:, :dl].T@a[:, :dl])
    coef = v@((v.T@(a[:, :dl].T@yc))[:, None]
              /(np.maximum(e, 0)[:, None]+len(tr)*ALP))
    out = [b[:, :dl]@coef]
    for weight in ([0., .1, 1.] if spec['groups'] == 'tune' else [0.]):
        for gamma in GAM:
            centers, coef = kernel_fit(a, yc, db, spec, weight, gamma)
            out.append(np.exp(
                -gamma*metric(b, centers, db, spec['groups'], weight))@coef)
    return ym+np.concatenate(out, 1)


def crossfit(x, y, db, spec, refit):
    pop = np.arange(1024)
    pred = np.empty(len(y))
    picks = []
    outer_loss = 0
    for f in range(8):
        tr, te = pop[FOLD != f], pop[FOLD == f]
        inner = (np.arange(len(tr))*4//len(tr))[
            np.argsort(np.argsort(tr*7919 % 1024))]
        loss = sum(
            ((candidate_predictions(
                x, y, tr[inner != j], tr[inner == j], db, spec)
              - y[tr[inner == j], None])**2).sum(0)
            for j in range(4))
        pr = candidate_predictions(x, y, tr, te, db, spec)
        pick = int(loss.argmin())
        pred[te] = pr[:, pick]
        picks.append(pick)
        outer_loss = outer_loss+((pr-y[te, None])**2).sum(0)
    if refit == 'index':
        outer_loss = sum(
            ((candidate_predictions(
                x, y, pop[pop % 8 != j], pop[pop % 8 == j], db, spec)
              - y[pop[pop % 8 == j], None])**2).sum(0)
            for j in range(8))
    choice = int(outer_loss.argmin())
    pred[1024:] = candidate_predictions(
        x, y, pop, np.arange(1024, len(y)), db, spec)[:, choice]
    return pred, picks, choice


def compile_model(x, y, db, spec, choice):
    """One selected readout is compiled; query-time prediction never fits."""
    family, weight, gamma, alpha = candidates(spec)[choice]
    mean, sd = x.mean(0), np.maximum(x.std(0), 1e-6)
    a, yc, ym = (x-mean)/sd, y-y.mean(), y.mean()
    if family == 'linear':
        dl = db if spec['groups'] == 'tune' else x.shape[1]
        e, v = eigh(a[:, :dl].T@a[:, :dl])
        beta = v@((v.T@(a[:, :dl].T@yc))/(np.maximum(e, 0)+len(y)*alpha))
        coef = np.r_[beta/sd[:dl], np.zeros(x.shape[1]-dl)]
        return {'family': np.array(family), 'coef': coef,
                'intercept': ym-mean@coef}
    centers, coef = kernel_fit(a, yc, db, spec, weight, gamma)
    coord = np.full(x.shape[1], 1/np.sqrt(x.shape[1]))
    if spec['groups'] == 'tune':
        coord = np.r_[
            np.full(db, 1/np.sqrt(db)),
            np.full(x.shape[1]-db, np.sqrt(weight/(x.shape[1]-db)))]
    return {'family': np.array(family), 'mean': mean, 'scale': coord/sd,
            'centers': centers*coord,
            'coef': coef[:, int(np.argmin(abs(ALP-alpha)))],
            'gamma': gamma, 'intercept': ym}


def predict_features(model, x):
    if str(model['family']) == 'linear':
        return np.asarray(x)@model['coef']+model['intercept']
    a = (np.atleast_2d(x)-model['mean'])*model['scale']
    return model['intercept']+np.exp(
        -model['gamma']*cdist(a, model['centers'], 'sqeuclidean'))@model['coef']


def predict_weights(model, weights):
    """Actual deterministic target query: cached prefix inputs + one readout."""
    spec = json.loads(str(model['spec']))
    shapes = [(16,), (16, 16), (10, 16)]
    if len(weights) != 3 or any(
            np.asarray(w).shape != shape for w, shape in zip(weights, shapes)):
        raise ValueError('One target requires weight shapes (16,), (16,16), (10,16)')
    w = [np.asarray(v)[None] for v in weights]
    b, m = prefix_cells(w, model)
    p = pulse_features(*w) if spec['pulse'] else None
    masks = [model[f'group{i}'] for i in range(6)]
    x, _ = summarize(b, m, weight_stats(*w), spec, model['ar'], p, masks)
    return float(predict_features(model, x)[0])


def costs(spec, plan, d, db, model):
    n = np.array([len(plan[f'parent{t}']) for t in range(10)])
    net = (256*n[:-1].sum()+16*len(plan['values'])+160*n[-1]
           +16*n[1:].sum()+16*n.sum()+20*n[-1])/2880
    k, ar = spec['K'], plan['ar']
    cells = k*len(ar)
    ops = (40 if spec['kind'] == 'both' else 22)*cells+5000
    if spec['groups'] != 'none':
        masks = [plan[f'group{i}'] for i in range(6)]
        channels = 4 if spec['kind'] == 'both' else 2
        ops += channels*k*sum(int(m.sum()) for m in masks)+channels*6*k
    feature = ops/2880
    pulse = 6. if spec['pulse'] else 0.
    ns = spec['supports'] or 1024
    budget = (2*ns*d+34*ns+2*d+3)/2880
    actual = (d+1)/2880 if str(model['family']) == 'linear' else budget
    return dict(
        anchor_rows=len(plan['target']),
        pulse_rows=2*int(spec['pulse']),
        prefix_fe=float(net), feature_fe=feature, pulse_fe=pulse,
        readout_budget_fe=budget, compiled_readout_fe=actual,
        total_fe=net+feature+pulse+budget,
        compiled_total_fe=net+feature+pulse+actual)


def metrics(pred, y, cost, recipe):
    groups = {'crossfit': slice(0, 1024)}
    groups.update(
        {'b128': slice(1024, 1088), 'b64': slice(1088, 1152),
         'mystery': slice(1152, 1153)}
        if recipe == 'unif'
        else {'mystery': slice(1024, 1025), 'seeds': slice(1025, 1030)})
    out = {}
    for name, sl in groups.items():
        e = pred[sl]-y[sl]
        mse = np.mean(e*e)
        out[name] = dict(
            rms_points=float(100*np.sqrt(mse)),
            mean_points=float(100*e.mean()),
            mse_ratio=float(mse*cost/np.mean(y[sl]*(1-y[sl]))))
        if name in ('mystery', 'seeds'):
            out[name]['errors_points'] = (100*e).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, default=Path('round7_result'))
    ap.add_argument('--recipe', choices=['unif', 'zoo'], default='unif')
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--kind', choices=['both', '4', '3'], default='4')
    ap.add_argument('--subset', choices=['all', 'ends'], default='all')
    ap.add_argument('--pulse', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--groups', choices=['none', 'fixed', 'tune'], default='tune')
    ap.add_argument('--supports', type=int, choices=[0, 128], default=0)
    ap.add_argument('--refit', choices=['outer', 'index'], default='index')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--cells', type=Path)
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--predict', type=Path)
    ap.add_argument('--weights', type=Path, default=Path('weights.npz'))
    args = ap.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.predict:
            with np.load(args.predict, allow_pickle=False) as model:
                print(json.dumps({
                    'estimated_accuracy':
                    predict_weights(model, read_weights(args.weights))}))
            return
        if args.K < 1:
            ap.error('--K must be positive')
        spec = {k: getattr(args, k) for k in [
            'recipe', 'K', 'kind', 'subset', 'pulse', 'groups', 'supports']}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        x, y, db, plan = cached_features(
            args.root, spec, args.cells, args.generate,
            args.out.with_suffix('.cells.npz'))
        pred, picks, choice = crossfit(x, y, db, spec, args.refit)
        model = compile_model(x[:1024], y[:1024], db, spec, choice)
        # Compare compilation with the selected candidate's external predictions.
        discrepancy = float(np.max(abs(
            predict_features(model, x[1024:])-pred[1024:])))
        if discrepancy > 1e-8:
            raise ArithmeticError(f'Compiled readout mismatch: {discrepancy}')
        ledger = costs(spec, plan, x.shape[1], db, model)
        result = dict(
            spec=spec, refit=args.refit,
            dimensions=x.shape[1], base_dimensions=db,
            costs=ledger,
            metrics=metrics(pred, y, ledger['total_fe'], args.recipe),
            outer_choices=[candidates(spec)[i] for i in picks],
            refit_choice=candidates(spec)[choice],
            compile_max_error=discrepancy,
            offline_seconds=time.monotonic()-started)
        np.savez_compressed(
            args.out.with_suffix('.model.npz'),
            **model, **plan, spec=np.array(json.dumps(spec)))
        np.savez_compressed(
            args.out.with_suffix('.predictions.npz'), pred=pred, y=y, X=x)
        args.out.with_suffix('.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
```