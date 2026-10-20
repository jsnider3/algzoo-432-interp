## Result

**The cheap end can beat sampling with all target-dependent arithmetic included. I did not reach the sampling benchmark at \(10^5\) forward-equivalents.**

The strongest practical improvement is **exact sharing of recurrent prefixes**, combined with a much cheaper network-level learned readout. For the original one-box designs:

| S slices \(K\) | Logical forward rows | Total cost, including features and readout | Population CV RMS, points | Sampling RMS at that cost | MSE / sampling |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,440 | **618.55 FE** | **0.58711** | 1.05698 | **0.309** |
| 2 | 2,880 | **1,129.66 FE** | **0.49214** | 0.78213 | **0.396** |
| 4 | 5,760 | **2,151.90 FE** | **0.43032** | 0.56669 | **0.577** |
| 8 | 11,520 | **4,196.37 FE** | **0.38087** | 0.40580 | **0.881** |

A learned fixed subset is useful below this range: **16 deterministic rows give 0.9514-point RMS at at most 15.875 FE** across the fitted folds and final deployment model.

The best measured envelope stays below sampling’s **point-estimate** error from 100 to about **4,760 FE**. At an available budget of \(10^5\), the best tested estimator has **8.89× sampling’s MSE**; at \(10^6\), **65.15×**. These are results against the supplied finite reference, with the selection qualifications discussed below.

The six zoo models transfer unevenly. The \(K=1\) global estimator also beats sampling **averaged over those six**, at **0.766× its MSE**, but it overestimates the mystery model by **2.266 points**. I would therefore describe this as an improvement in deterministic population prediction, with exact computational components—not a completed mechanistic solution.

### Deliverables

- [Q6_report.md](sandbox:/workspace/scratch/d2058543327b/algzoo_q6_results/Q6_report.md) — complete audit, findings, numerical appendices, and standalone code.
- [q6_evidence.zip](sandbox:/workspace/scratch/d2058543327b/algzoo_q6_results/q6_evidence.zip) — 74 MB; source drivers, fitted models, cross-fitted predictions, compact feature caches, diagnostics, and a checksum manifest.
- [q6_methods.py](sandbox:/workspace/scratch/d2058543327b/algzoo_q6_results/q6_methods.py) — standalone NumPy/SciPy implementation.
- [q6_results.json](sandbox:/workspace/scratch/d2058543327b/algzoo_q6_results/q6_results.json) and [complete_ledger.md](sandbox:/workspace/scratch/d2058543327b/algzoo_q6_results/complete_ledger.md) — full-precision scores, costs, validation protocols, timing scopes, and all six zoo errors for every retained variant.

I consolidated the operative new estimator algorithms into the complete program pasted at the end. The archive additionally preserves the exploratory and audit drivers.

---

## 1. Audit: the main numbers reproduce, but several qualifications matter

### z45–z47

The saved population predictions reproduce the headline **1.102800×** and **1.129390×** ratios for the 32–32 and 16–16 MLPs under the original logical-row and nominal inference accounting.

There are four material corrections:

1. **Preprocessing crosses the fold boundary.** z47 excludes the held-out network’s accuracy label, but computes context means and standard deviations on all 1,024 networks before splitting. z45 does this too, and chooses its ridge penalty using the same CV scores it reports. The new fits use training-fold preprocessing; selection is nested where stated.

2. **The optimized first-layer cost is prospective.** The advertised cost assumes precomputed embedding projections and one context projection per target. That optimization is feasible, but the archived `G.forward` repeats a full dense concatenated first layer per cell. Weight statistics, normalization, reductions, and activations also require arithmetic.

3. **Zoo inference is an eight-model ensemble.** Its RNN outputs can be reused, but all eight learned readouts must be charged. The zoo also needs its own comparator, \(\overline{p(1-p)}=0.073747335\), rather than the population’s \(0.069104512\).

4. **Cross-fitting does not erase research selection.** Widths, layouts, seeds, features, and screening choices were compared using the same population. That qualification also applies to choosing among my new pipelines after seeing their results. The displayed best-at-budget envelope is retrospective.

The ensemble correction alone changes the nominal zoo comparison substantially:

| Original MLP | Population total FE | Population MSE ratio | Zoo total FE, eight readouts | Corrected zoo MSE ratio |
|---|---:|---:|---:|---:|
| 128–128–64 | 7,892 | 4.680 | 53,056 | **64.894** |
| 32–32 | 1,780 | 1.103 | 4,160 | **4.978** |
| 16–16 | 1,548 | 1.129 | 2,304 | **2.502** |
| 8 | 1,464 | 1.483 | 1,632 | **1.702** |

These remain nominal original-implementation comparisons; they do not incorporate the new prefix evaluator.

### The cell law is correct; the targets have shared noise

The three independent Beta coordinates and the conditional \(S\) slices do produce \(720n^3K\) equal-probability cells. The target generator correctly randomizes \(Q\)’s position. The fixed representative \(Q\) used for a feature row is a permissible deterministic design, but it is not exact averaging over the seven possible positions.

For \(K=1,2,4\), the mean cell-target integral exceeds the reference by **0.08756, 0.05502, and 0.04387 points**, respectively. The corresponding RMS discrepancies are **0.11427, 0.08031, and 0.06190**. The fractions with denominator 128 are exactly representable in float16, so target-storage quantization is not the issue.

The K1 MLP mean errors, roughly \(+0.08\) to \(+0.10\) points, track that shared target offset. This is evidence of a common training-target effect, rather than a proof that it explains every residual.

### Coverage of the other §5 components

| Component | Audit finding and measured replacement status |
|---|---|
| Reference and scoring | Rescored supplied arrays and audited z40. Did **not** regenerate the million-input reference. Variable-cost comparisons should use \(\operatorname{mean}_i[p_i(1-p_i)/C_i]\). |
| Lower-input models | Source and logs are coherent, but several small reported errors are noise-subtracted: 0.052/0.065/0.160 points come from observed 0.059/0.071/0.169; the pair result 0.048 comes from 0.064. These are conditional-model errors evaluated using sampled outer configurations, not complete deterministic estimators at the local row costs. No new lower-input model improvement was established here. |
| Anchors and outer rules | Probability transforms check out. Historical **9,300/1,528** optimized exact-S costs remain unreplicated. z37 uses 256 S midpoints while charging an exact-line implementation; its MF2 path also changes \(\mu(S)\), so it is not affine. The new prefix savings were executed and verified for the actual cell designs. |
| Arrangements | z43’s definitions and logged group variances are consistent. Failure of the tested additive arrangement fits does not exclude learned summaries or shared computation. |
| Regions and gate histories | z44’s full-sample history count and half-sample unseen-mass calculation are internally consistent. They do not prove necessary integration complexity. Missing mass limits the tested certificate, not every estimator. |
| Gating and screening | The approximately \(-2.3/+1.9\)-point flows are parts of **the proxy \(F_4-I_4\)**, not measurements of the true joint lower-input law. Reported screen coverage is proxy coverage. To realize the projected savings, expensive rows must actually be skipped after screening. |
| Two-level estimator | Rescoring saved z41 quantities reproduces the dev/held RMS **0.2254/0.2493** and ratios **170.4/219.9**, using each network’s tracer cost. Root, sorting, and CDF work is extra. The expensive integrator was not regenerated. |
| Other frames | The archived event anchor has 2.041-point RMS. Keeping the last one/two “neither” inputs exact already improves it to 1.409/0.995; their complete deterministic integration costs remain missing. |
| Calibration and roles | z45 population regressions reproduce. The 4,096 role probes come from `torch.randn`, so those target-side role-feature estimators violate this question’s restriction even with a fixed seed. z45 does not report zoo errors for its best feature-augmented rows. |
| Lower bounds | The membership-oracle cone bound does not apply to weight-reading population estimators. Independent bounded cell residuals alone give \(C^{-1/2}\) RMS; a faster exponent needs additional variance decay or boundary regularity. |

The detailed report distinguishes rescoring saved arrays, auditing source/logs, and executing replacements. I did not retrain the archived MLPs or rerun every historical simulation.

---

## 2. What improved

### A. Exact recurrent-prefix sharing

There are two levels of reuse:

- Compute \(W_{ih}x\) once for each distinct scalar in the fixed design.
- Compute \(W_{hh}h\) once for each distinct parent state, then reuse it among that parent’s children.

For the one-box \(K\)-slice design, the resulting network cost is

\[
\boxed{C_{\mathrm{network}}(K)=494K+69.4\ \mathrm{FE}.}
\]

This includes state additions, ReLU comparisons, winner/rival reductions, and margins. It gives **563.4, 1,057.4, and 2,045.4 FE** for K1/K2/K4.

Across **10,382,400 supplied feature rows over all 1,030 networks**, float32 correctness bits and margins match the stored data exactly. Float64 changes only three near-zero-margin correctness bits in K4, none on the zoo.

This validates these designs’ savings. It does not validate the older exact-S constants, which also presumed reuse between traced segments.

### B. Train the network integral directly with cheap summaries

For each cell I used:

\[
I_4,\ I_3,\ a,\ b,\ |a|,\ |b|,\ 
\frac{a}{0.1+|a|},\frac{b}{0.1+|b|},
\frac{a}{0.03+|a|},\frac{b}{0.03+|b|},
\]

where \(a,b\) are margins normalized by the target’s already-computed MF4 margin RMS.

I aggregate these globally, by S slice, and—on finer grids—by each outer box digit. I append the 20 weight summaries, margin scale/log scale, and raw mean margins. The readout predicts **network accuracy directly**, replacing cellwise cross-entropy.

Each of eight outer folds contains four-fold selection of ridge versus RBF, regularization, and kernel scale. The final zoo predictions use one model refitted on all 1,024 population networks.

A separate experiment also nests selection among **global, position, pair, and full-cell grouping families**. Its final linear coefficients compile offline into a fixed per-cell coefficient table, so a large intermediate feature vector is unnecessary at inference.

### C. Learn a small fixed set of probes

Group orthogonal matching pursuit selects among the 720 K1 cell pairs, conditional on the weight summaries. Each outer fold relearns the selected cells and their scales.

Crucially, the scale of a selected cell is fitted across **training networks**. A future target evaluates only the selected rows; it does not evaluate all 720 cells to obtain an uncharged normalization.

The linear results for 8/16/32/64/128 selected cells are **0.9514/0.9499/0.9928/1.0221/0.9806 points**. More selected probes do not consistently improve this readout. RBF improves the larger subsets modestly, reaching **0.8823 points with 128 cells**.

### Main online ledger

Here \(G(B,K)\) means the global summary estimator with \(B=n^3\) outer boxes and \(K\) S slices; \(L(K)\) is the strictly nested compiled linear estimator. Costs use a conservative common cap across fitted folds and final deployment, including up to 1,024 kernel supports.

“Rows” are logical rows. They are **not** added again to the shared network FE column.

| Method | Rows | Network FE | Other FE | Total FE | CV RMS | Mean error | MC RMS | MSE ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weight RBF¹ | 0 | 0 | 21.55 | 21.55 | 1.1749 | +0.0127 | 5.6624 | 0.043 |
| 8-cell linear subset | 16 | 14.08 | 1.79 | 15.88 | 0.9514 | +0.0115 | 6.5978 | 0.021 |
| 16-cell linear subset | 32 | 26.36 | 1.85 | 28.20 | 0.9499 | +0.0018 | 4.9500 | 0.037 |
| 64-cell RBF subset | 128 | 77.02 | 159.90 | 236.92 | 0.9109 | +0.0693 | 1.7079 | 0.284 |
| 128-cell RBF subset | 256 | 128.84 | 297.41 | 426.25 | 0.8823 | +0.0584 | 1.2733 | 0.480 |
| \(L(1)\) | 1,440 | 563.40 | 11.74 | 575.14 | 0.6913 | −0.0062 | 1.0961 | 0.398 |
| \(L(2)\) | 2,880 | 1,057.40 | 21.74 | 1,079.14 | 0.5577 | +0.0021 | 0.8002 | 0.486 |
| \(L(4)\) | 5,760 | 2,045.40 | 41.74 | 2,087.14 | 0.4536 | −0.0013 | 0.5754 | 0.621 |
| \(G(1,1)\) | 1,440 | 563.40 | 55.15 | 618.55 | 0.5871 | +0.0054 | 1.0570 | 0.309 |
| \(G(1,2)\) | 2,880 | 1,057.40 | 72.26 | 1,129.66 | 0.4921 | +0.0028 | 0.7821 | 0.396 |
| \(G(1,4)\) | 5,760 | 2,045.40 | 106.50 | 2,151.90 | 0.4303 | +0.0034 | 0.5667 | 0.577 |
| \(G(1,8)\) | 11,520 | 4,021.40 | 174.97 | 4,196.37 | 0.3809 | +0.0008 | 0.4058 | 0.881 |
| \(G(8,2)\) | 23,040 | 8,417.20 | 374.97 | 8,792.17 | 0.3559 | +0.0019 | 0.2804 | 1.612 |
| \(G(8,8)\) | 92,160 | 32,129.20 | 1,257.68 | 33,386.88 | 0.2666 | −0.0001 | 0.1439 | 3.433 |
| \(G(27,4)\) | 155,520 | 55,042.20 | 2,020.56 | 57,062.76 | 0.2478 | −0.0001 | 0.1100 | 5.071 |
| \(G(27,8)\) | 311,040 | 108,394.20 | 3,939.03 | 112,333.23 | 0.2273 | +0.0021 | 0.0784 | 8.396 |
| \(G(64,8)\) | 737,280 | 256,887.20 | 9,140.39 | 266,027.59 | 0.2164 | +0.0025 | 0.0510 | 18.023 |
| \(G(125,8)\) | 1,440,000 | 497,194.00 | 17,701.74 | 514,895.74 | 0.2122 | +0.0012 | 0.0366 | 33.545 |

¹ The fixed raw20 RBF cross-fit is conditional on its earlier development selection. Its primary untouched 64-network heldout RMS is **1.113792**, mean **−0.006764**, at **18.975 FE** with 896 supports. A broader nested weight-family search gives 1.183031-point RMS, using different feature/model combinations and different costs.

The complete ledger includes the additional slice counts, all subset sizes, all screening variants, and their individual timing fields.

### Zoo errors for the principal variants

All entries are **estimated accuracy minus reference accuracy**, in percentage points. Zoo models were not used to fit these readouts.

| Method | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---|---:|---:|---:|---:|---:|---:|
| Weight RBF | +1.528 | +0.658 | +0.069 | +0.332 | −0.141 | +0.370 |
| 8-cell linear subset | +1.726 | +0.371 | +0.566 | −0.337 | −0.576 | +1.067 |
| 16-cell linear subset | +1.688 | −0.502 | +0.258 | −0.757 | −0.733 | −0.155 |
| 128-cell RBF subset | +1.232 | −0.433 | +0.241 | −0.452 | −0.640 | +0.187 |
| \(L(1)\) | +1.966 | −0.078 | +0.905 | −1.536 | −0.442 | +0.349 |
| \(G(1,1)\) | +2.266 | +0.509 | −0.016 | −0.188 | −0.154 | +0.162 |
| \(G(1,2)\) | +1.862 | −0.482 | +0.471 | +0.001 | −0.414 | −0.071 |
| \(G(1,4)\) | +1.496 | −0.360 | +0.481 | −0.183 | −0.546 | −0.242 |
| \(G(1,8)\) | +0.567 | −0.462 | +0.456 | −0.150 | −0.529 | −0.805 |
| \(G(8,8)\) | +0.912 | −0.217 | +0.206 | +0.132 | −0.354 | −0.165 |
| \(G(27,4)\) | +1.303 | −0.205 | +0.201 | +0.110 | −0.340 | −0.036 |
| \(G(125,8)\) | +0.595 | −0.218 | +0.189 | +0.077 | −0.296 | −0.202 |

---

## 3. Other approaches I executed

### Screening helps one intermediate point, but the residual-selected experiment failed

Adding all 12 extreme lower-input rows to every K4 cell reduces RMS to **0.35001**, but costs **12,206.37 FE**, giving **2.164×** sampling MSE.

Using those rows on every eighth K4 cell gives a more useful point: **0.38133 RMS at 3,558.29 FE**, ratio **0.749**. Its zoo errors are:

\[
[+1.630,-0.779,+0.849,-0.390,-0.341,-0.058].
\]

I also tested the more targeted proposal: choose individual screen rows for their value in predicting the residual after the cheap global features. Each outer fold relearned the selection.

| Added selected rows | Total FE cap | CV RMS | Mean error | Zoo RMS |
|---:|---:|---:|---:|---:|
| 0, fixed ridge10 baseline | 575.136 | **0.72611** | −0.00135 | **0.93582** |
| 16 | 585.981 | 0.74651 | +0.01121 | 1.09786 |
| 32 | 596.525 | 0.80509 | +0.01911 | 1.37885 |
| 64 | 615.614 | 0.85301 | +0.03454 | 1.41747 |

The selected 16 rows elicit more individual anchor breaks than the candidate average—**0.446% versus 0.215%**—yet accuracy prediction deteriorates. Finding sensitive individual probes is therefore insufficient for this selection/readout family.

### Symmetry-preserving features exposed a limitation rather than fixing the curve

For positive diagonal \(D\),

\[
W_{ih}'=DW_{ih},\qquad
W_{hh}'=DW_{hh}D^{-1},\qquad
W_o'=W_oD^{-1}
\]

preserves the logits exactly. Doubling \(W_o\) also preserves every answer.

Nevertheless, on 64 population networks:

| Function-preserving change | RMS change in the raw20 RBF estimate | Maximum change |
|---|---:|---:|
| Hidden scales alternating 0.5 and 2 | **2.39284 points** | 5.97150 |
| Readout multiplied by 2 | **3.97617 points** | 7.70248 |

The raw weight predictor is learning useful associations within the training recipe. Those associations are not intrinsic properties of the computed function.

I constructed a readout gauge using centered readout-column norms and removed raw scale-dependent summaries. The resulting global estimators have:

| K | Total FE | Population RMS | Mean error | MSE ratio |
|---:|---:|---:|---:|---:|
| 1 | 642.85 | 1.12191 | +0.00368 | 1.171 |
| 2 | 1,153.97 | 0.92791 | +0.00445 | 1.438 |
| 4 | 2,176.21 | 0.61202 | −0.00128 | 1.180 |

Their zoo errors are:

| K | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.281 | +0.698 | +0.394 | −0.494 | −0.188 | +0.404 |
| 2 | −0.716 | −0.712 | +0.082 | −0.568 | −0.915 | +0.203 |
| 4 | −0.990 | −0.915 | +0.654 | −0.477 | −0.688 | −0.606 |

The construction has the stated invariances on the nondegenerate domain. A zero centered-readout column needs special treatment: that neuron may still influence later states through recurrence.

### Exact last-input integration works, but the preceding state approximation remains expensive

Given the exact state after nine inputs, the last step has at most 16 ReLU gate roots, plus the two values where the target changes. On each of at most 19 intervals, the logits are affine. Intersect the target’s nine linear inequalities and integrate the resulting intervals with Gaussian CDF differences.

The sweep passed deterministic high-resolution checks, including persistent-tie cases. The maximum discrepancy in the 18-prefix midpoint checks was \(5.14\times10^{-6}\), within the interval-count bound.

For the earlier nine inputs I used a new MF3 anchor with all 504 position triples and a \(q^3\) Beta rule:

| \(q\) | Nine-input prefixes | Mean heldout FE | Raw heldout RMS | Raw mean error | Dev-calibrated heldout RMS | Calibrated mean | MSE ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 504 | 1,303.44 | 5.2264 | +5.0265 | 1.1056 | −0.0460 | 2.283 |
| 2 | 4,032 | 10,324.80 | 3.5654 | +3.4774 | 0.6400 | +0.0560 | 6.057 |
| 3 | 13,608 | 34,655.07 | 1.2212 | +1.1294 | 0.4371 | −0.0015 | 9.484 |

Calibration uses only the 64 development networks. Calibrated zoo errors are:

| \(q\) | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.994 | +1.074 | +1.092 | −0.406 | −0.356 | +2.125 |
| 2 | −0.172 | +1.136 | +0.905 | −0.460 | +0.111 | +0.440 |
| 3 | −0.334 | +0.357 | +0.751 | −0.608 | −0.568 | −0.311 |

Adding the q1 probability and segment count to K1/K2/K4 global summaries gives nested RMS **0.5590/0.4886/0.4200**, but cheaper global-only alternatives dominate them. The q3+K4 combination gives **0.30346** at a conservative cap of **37,294 FE**, with zoo errors

\[
[+0.836,-0.002,+0.246,+0.003,-0.548,-0.621].
\]

The attractive 0.222-point development-selected heldout result for that combination should not be substituted for its full-population nested score.

---

## 4. The best measured end-to-end curve

![Deterministic estimator cost and error curve](sandbox:/workspace/scratch/d2058543327b/algzoo_q6_results/q6_cost_error_curve.png)

The table compares sampling at the **whole available budget**, even when the selected estimator spends less.

| Available FE | Best tested choice | FE spent | RMS | Sampling RMS | MSE ratio |
|---:|---|---:|---:|---:|---:|
| 100 | 16-cell linear subset | 28.20 | 0.94985 | 2.62877 | **0.131** |
| 300 | 64-cell RBF subset | 236.92 | 0.91090 | 1.51772 | **0.360** |
| 1,000 | \(G(1,1)\) | 618.55 | 0.58711 | 0.83129 | **0.499** |
| 1,500 | \(G(1,2)\) | 1,129.66 | 0.49214 | 0.67875 | **0.526** |
| 3,000 | \(G(1,4)\) | 2,151.90 | 0.43032 | 0.47995 | **0.804** |
| 4,000 | K4 screens on every eighth cell | 3,558.29 | 0.38133 | 0.41565 | **0.842** |
| 5,000 | \(G(1,8)\) | 4,196.37 | 0.38087 | 0.37176 | **1.050** |
| 10,000 | \(G(8,2)\) | 8,792.17 | 0.35593 | 0.26288 | **1.833** |
| 100,000 | \(G(27,4)\) | 57,062.76 | 0.24782 | 0.08313 | **8.887** |
| 300,000 | \(G(64,8)\) | 266,027.59 | 0.21638 | 0.04799 | **20.325** |
| 1,000,000 | \(G(125,8)\) | 514,895.74 | 0.21218 | 0.02629 | **65.149** |

This is a **retrospective envelope over explored pipelines**, not an independently validated rule for selecting the winning pipeline. The approximately 4,760-FE crossing is a point estimate, not a confidence guarantee.

### Cost and offline-work qualifications

I used the requested unit, **1 FE = 2,880 nominal MACs**. All target feature extraction and readouts are charged. Ordinary arithmetic, comparisons, square roots, and logs are scalar primitives; kernel exponentials receive a 32-operation allowance and the last-input sweep reserves 100 per Gaussian CDF. These are declared arithmetic budgets, not hardware timing equivalences.

The consolidated formulas include:

\[
\begin{aligned}
\text{Global direct-distance kernel: }&2Nd+34N+2d+1,\\
\text{Optimized weight kernel: }&2681+58N,\\
\text{Subset kernel addition: }&N(d+38)+4d+1.
\end{aligned}
\]

The last expression deliberately retains the base allowance for an unused linear readout, making it conservative.

All new experiments ran on CPU. Offline costs included:

| Work | Recorded elapsed time |
|---|---:|
| Fourteen deterministic cache builds, all 1,030 networks | **965.36 s total** |
| Five sparse linear sizes, including cross-fitting | 16.6 s |
| Five sparse RBF sizes, including cross-fitting | 18.1 s |
| Three residual-selected screen sizes | 37.4 s |
| K1/K2/K4 nested global readouts | 70.6 s |
| Six uniform/partial-screen readouts | 121.0 s |
| Strict nested four-family linear K1/K2/K4 | about 81 s |
| Broader nested weight-family selection | 293.9 s |
| Exact-last features, all 1,030: q1/q2/q3 | 5.28 / 36.97 / 128.60 s |

The cache builds represent **3.046 billion logical rows**, reduced by sharing to **1.055 billion network FE**. Some builds repeat rows; these totals describe recorded work, not unique designs. Fitting timings generally use two BLAS threads and exclude inherited reference generation.

The supplied reference itself represents about **1.074 billion population forward evaluations**, which were not regenerated.

---

## 5. Mechanism, uncertainty, and the next experiment

### What the learned components appear to capture

The selected probes favor late extrema. Across the 16-cell fold designs, the maximum is last **22.66%** of the time versus **10%** among candidates, and follows the second-largest input **67.19%** versus **50%**. This is compatible with the handoff and clock/pointer failures described in your mechanism study.

The weight predictor reproduces the population association between comparator count and accuracy: approximately **0.530** correlation with predicted accuracy versus **0.533** with true accuracy. Those role measurements were used only for offline interpretation.

Neither observation identifies a deductive, network-specific breaking-rate parameter. The failed individual-screen experiment is evidence that such a parameter is not easily recovered by the tested coordinate probes and readout.

### The mystery model is not an obvious feature-support outlier

In raw20 space, its largest standardized coordinate is **1.272**, and its shrinkage-Mahalanobis distance is at the **53.4th population percentile**. Its 32 nearest raw20 neighbors average **95.466%** accuracy, near its **95.305%** reference accuracy, while the weight kernel predicts about **96.833%**.

That weakens the simple explanation that the mystery model lies outside the population’s feature range. An imperfect learned response surface or a missing circuit property remains plausible; I did not identify which.

### Reference noise matters differently at the two ends

The reference RMS standard error is about **0.02567 points**. Under independent scoring noise, its variance relative to sampling’s is approximately

\[
\frac{C}{2^{20}},
\]

or **9.54% at \(10^5\)** and **95.37% at \(10^6\)**.

Training and scoring labels share an input pool, so holding out networks does not remove all reference-noise covariance. Blindly subtracting the reported SE is unjustified.

As a check, replacing the reference with the independent stored `acc_2e20` changes fixed global-ridge K1/K2/K4 RMS from **0.68911/0.55247/0.44155** to **0.68757/0.55095/0.43985**. Reference noise cannot explain the cheap improvement or rescue the present high-budget gap.

For a high-precision confirmation, freeze the estimator and use an independent scoring pool. To keep reference variance below 1% of the sampling comparator, use \(R\ge100C\), for example \(2^{24}\) inputs at \(10^5\) FE and \(2^{27}\) at \(10^6\).

### The useful theoretical statement

For a fixed feature map \(F_C\), conditional on independent offline training,

\[
\mathbb E[(g(F_C)-p)^2]
=
\mathbb E[\operatorname{Var}(p\mid F_C)]
+
\mathbb E[(g(F_C)-\mathbb E[p\mid F_C])^2].
\]

Matching sampling requires their sum to be at most \(\mathbb E[p(1-p)]/C\); beating its asymptotic RMS rate requires \(o(1/C)\).

For cell residuals \(e\) and weights \(w\),

\[
\mathbb E[(w^\top e)^2]
=
w^\top\operatorname{Cov}(e)w
+
(w^\top\mathbb E[e])^2.
\]

A shared network-dependent component survives averaging. More cells help only if they also resolve that component, or otherwise reduce the integrated covariance and bias.

This is a limitation of a chosen feature/predictor class. With the full weights and unrestricted computation, accuracy is a deterministic function of the features, so the conditional-variance term is zero. It is not a general white-box lower bound.

### Verdict and the single experiment I would run next

The cheap end is materially better. The \(10^5\)-FE target remains unmet, and the strongest cheap predictor is not a parameterization-independent explanation of the circuit.

**The next experiment should learn a small fixed set of joint, temporally adjacent lower-input perturbations near the extreme S slices, conditional on the cheap global estimator’s residual.** Charge the union of their recurrent prefixes with the base design, learn selection inside the outer folds, then freeze it for a fresh network/reference test.

The archive already contains pairwise lower-input models. The new part would be **population-selected joint probe placement for the network-integral residual**. It directly tests whether shared gate changes, invisible to individual perturbations, contain the missing information.

I did not implement continuous probe optimization, exact S-integration of the clipped correction, adaptive refinement, affine-state merging, finite cone covers, exact-integrable distilled surrogates, or new zoo-like training populations. They remain open routes; the measured failures do not exclude them.

---

## 6. Complete standalone estimator code

This is the implementation in `q6_methods.py`. It needs only NumPy and SciPy. It includes the deterministic designs, shared-prefix evaluator, global and compiled readouts, learned subsets and screens, gauges, exact-last integration, fitting, deployment, and scoring.

Saved-model equivalence checks passed: global K1 predictions agreed within \(2\times10^{-13}\); selected linear and screen predictions were exact; selected RBF differed by at most \(6\times10^{-15}\); weight predictions were exact across all 1,030; compiled K1/K2/K4 zoo predictions agreed within \(4\times10^{-15}\).

```python
"""Q6 deterministic accuracy estimators: NumPy/SciPy, one standalone file.

Training reads the ORIGINAL archive's NPZ files; deployment needs only a
weight NPZ and fitted model NPZ. Target inputs are deterministic quadrature
or population-learned fixed probes. No target samples or quasi-random points.

Examples (set OPENBLAS_NUM_THREADS=1 before running):
 python q6_methods.py weight REPO weight_model.npz --cv
 python q6_methods.py summary REPO global_k1.npz --k 1 --cv
 python q6_methods.py subset REPO subset16.npz --count 16 --cv
 python q6_methods.py subset REPO subset16_rbf.npz --count 16 --kernel rbf --cv
 python q6_methods.py cache REPO screen.npz --k 1 --screen all
 python q6_methods.py screens REPO screens32.npz --cache screen.npz --count 32 --cv
 python q6_methods.py cache REPO large.npz --boxes 5 --k 8 --summary-only
 python q6_methods.py summary REPO large_model.npz --cache large.npz --cv
 python q6_methods.py compiled REPO compiled_k1.npz --k 1
 python q6_methods.py last REPO last_q1.npz --q 1
 python q6_methods.py summary REPO combined.npz --k 1 --last last_q1.npz --cv
 python q6_methods.py predict weights.npz global_k1.npz

The last command also loads models produced by the individual Q6 scripts;
provide --k/--boxes/--screen/--context when an older model lacks metadata.
Cost unit: 2,880 multiply-adds. Prefix arithmetic is counted explicitly.
Other helpers use stated conservative scalar-operation allowances. Exp is
charged 32 operations and Gaussian CDF 100; these are accounting assumptions.
CV for global models nests family/regularization selection. Subset/screen
CV relearns the design in every outer fold with fixed stated hyperparameters.
"""
import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import betaincinv, ndtr, ndtri, roots_jacobi

ALPHAS = np.logspace(-3, 3, 9)
GAMMAS = [.03, .1, .3, 1., 3.]
FEATURE_COST = dict(raw20=2600, readoutgauge=7000, impulse=44000,
                    raw_readout=9600, raw_impulse=46600)


# ---------- Deterministic quadrature design and exact shared prefixes ----------

def design(k=1, boxes=1, screen=False):
    """Return z46's exact float32 input rows, targets, and cell count."""
    arr = np.array(list(itertools.permutations(range(10), 3)))
    qof = {0: 7, 7: 0, 1: 3, 3: 1, 2: 5, 5: 2, 4: 6, 6: 4}
    free = [[p for p in range(10) if p not in a[:2]] for a in arr]
    qrep = np.array([f[qof[f.index(a[2])]] for f, a in zip(free, arr)])
    low6 = np.array([[p for p in range(10) if p not in a and p != q]
                     for a, q in zip(arr, qrep)])
    un = (np.array([-.9602898564975363, .9602898564975363])+1)/2
    rows = 14 if screen else 2
    ai, ki = np.arange(720)[:, None], np.arange(k)[None, :]
    beta = [(7, 4), (1, 3), (2, 1)]
    all_x = []
    for b in range(boxes**3):
        half = [(b // boxes**i) % boxes for i in range(3)]
        u = [betaincinv(*beta[i], (half[i] + .5) / boxes) for i in range(3)]
        fq = u[0]
        fr = fq + (1 - fq) * u[1]
        fm = fr + (1 - fr) * u[2]
        q, r, m = ndtri([fq, fr, fm])
        s = ndtri(fr + (fm - fr) * (np.arange(k) + .5) / k)
        mu = lambda v: -np.exp(-v*v/2) / np.sqrt(2*np.pi) / ndtr(v)
        x = np.empty((720, k, rows, 10))
        for row, lo in [(0, q), (1, r)]:
            x[:, :, row] = mu(lo)
            x[ai, ki, row, arr[:, 0, None]] = m
            x[ai, ki, row, arr[:, 1, None]] = s
            x[ai, ki, row, arr[:, 2, None]] = r
        x[ai, ki, 0, qrep[:, None]] = q
        if screen:
            for j in range(6):
                for h in range(2):
                    row = 2+2*j+h
                    x[:, :, row] = x[:, :, 0]
                    x[ai, ki, row, low6[:, j, None]] = ndtri(fq*un[h])
        all_x.append(x.astype(np.float32).reshape(-1, 10))
    x = np.concatenate(all_x)
    target = np.tile(np.repeat(arr[:, 1], rows*k), boxes**3)
    return x, target, len(x) // rows


class PrefixEvaluator:
    def __init__(self, x):
        self.values, token = np.unique(x, return_inverse=True)
        token = token.reshape(x.shape)
        parent = np.zeros(len(x), dtype=np.int64)
        self.levels = []
        for t in range(x.shape[1]):
            nodes, parent = np.unique(
                np.column_stack([parent, token[:, t]]), axis=0,
                return_inverse=True)
            self.levels.append((nodes[:, 0], nodes[:, 1]))
        self.leaf = parent
        self.counts = [len(p) for p, _ in self.levels]

    def __call__(self, wih, whh, wo, dtype=np.float64):
        wih, whh, wo = [np.asarray(w, dtype=dtype) for w in (wih, whh, wo)]
        # Reuse W_ih*x across occurrences of the same design scalar.
        inp = self.values.astype(dtype)[:, None] * wih
        p, v = self.levels[0]
        h = np.maximum(inp[v], 0)  # h_0 = 0.
        for p, v in self.levels[1:]:
            # Reuse W_hh*h among every child of the same parent.
            recurrent = h @ whh.T
            h = np.maximum(recurrent[p] + inp[v], 0)
        return (h @ wo.T)[self.leaf]

    def cost(self):
        n = np.array(self.counts)
        recurrent_mac = 256 * int(n[:-1].sum())
        input_mult = 16 * len(self.values)
        output_mac = 160 * int(n[-1])
        state_add = 16 * int(n[1:].sum())
        relu_compare = 16 * int(n.sum())
        output_scalar = 20 * int(n[-1])
        mac = recurrent_mac + input_mult + output_mac
        other = state_add + relu_compare + output_scalar
        return dict(logical_rows=len(self.leaf), prefix_counts=self.counts,
                    recurrent_mac=recurrent_mac, input_mult=input_mult,
                    output_mac=output_mac, mac_forward_equiv=mac/2880,
                    scalar_ops=other, scalar_forward_equiv=other/2880,
                    total_forward_equiv=(mac+other)/2880,
                    saving=len(self.leaf)*2880/(mac+other))


def load_models(repo):
    d = np.load(repo / 'pop1k/seed0.npz')
    p = [d['s1048576_' + key] for key in ('Wih', 'Whh', 'Wo')]
    zoo = []
    for fname in ['weights.npz'] + [f'weights_seed{i}.npz' for i in range(5)]:
        w = np.load(repo / fname)
        zoo.append([w['rnn.weight_ih_l0'][:, 0], w['rnn.weight_hh_l0'],
                    w['linear.weight']])
    return [np.concatenate([p[i], np.stack([w[i] for w in zoo])])
            for i in range(3)]


def load_weights(path):
    d = np.load(path)
    return [d['rnn.weight_ih_l0'][:, 0],
            d['rnn.weight_hh_l0'], d['linear.weight']]


def bits_margins(logits, target):
    bits = logits.argmax(-1) == target
    margin = logits[np.arange(len(target)), target].copy()
    logits[np.arange(len(target)), target] = -np.inf
    return bits, margin-logits.max(-1)


# ---------- Weight summaries and two gauge constructions ----------

def raw20(wi, A, O):
    wi, A, O = [np.asarray(w, dtype=float) for w in (wi, A, O)]
    if wi.ndim == 1:
        wi, A, O = wi[None], A[None], O[None]
    F = np.stack([
        wi, np.diagonal(A, axis1=1, axis2=2),
        np.log(np.linalg.norm(A, axis=2)+1e-9),
        np.log(np.linalg.norm(A, axis=1)+1e-9),
        np.log(np.linalg.norm(O, axis=1)+1e-9)
    ], axis=2)
    return np.concatenate(
        [F.mean(1), F.std(1), F.min(1), F.max(1)], axis=1)


def pool(x):
    return np.concatenate(
        [x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)


def readout_gauge(wi, A, O):
    eps = 1e-30
    Z = O-O.mean(1, keepdims=True)
    rn = np.linalg.norm(Z, axis=1)
    rn = np.where(rn > 0, rn, 1.)
    R = A*rn[:, :, None]/rn[:, None, :]
    U = Z/rn[:, None, :]
    b = wi*rn
    bn = np.linalg.norm(b, axis=1)
    b /= np.where(bn > 0, bn, 1.)[:, None]
    nodes = [
        b, np.diagonal(R, axis1=1, axis2=2),
        np.log(np.linalg.norm(R, axis=2)+eps),
        np.log(np.linalg.norm(R, axis=1)+eps)
    ]
    nodes.extend(U[:, j] for j in range(10))
    return pool(np.stack(nodes, axis=2))


def impulse_gauge(wi, A, O):
    eps = 1e-30
    v = np.stack([np.maximum(wi, 0), np.maximum(-wi, 0)], axis=2)
    g2 = np.sum(v*v, axis=2)
    for _ in range(9):
        v = A@v
        g2 += np.sum(v*v, axis=2)
    g = np.sqrt(np.maximum(g2, eps))
    C = A*g[:, None, :]/g[:, :, None]
    B = (O-O.mean(1, keepdims=True))*g[:, None, :]
    B /= np.maximum(np.linalg.norm(B, axis=(1, 2)), eps)[:, None, None]
    V = B.copy()
    f = []
    for age in range(10):
        target = 9-age
        rival = np.max(V[:, np.arange(10) != target], axis=1)
        scale = np.maximum(np.linalg.norm(V, axis=1), eps)
        f.extend([(V[:, target]-rival)/scale,
                  (V.argmax(1) == target).astype(float)])
        if age < 9:
            V = V@C
    q = np.stack(f, axis=2)
    q = np.concatenate([
        q, q[:, :, 0::2].mean(2, keepdims=True),
        q[:, :, 0::2].min(2, keepdims=True),
        q[:, :, 1::2].mean(2, keepdims=True)
    ], axis=2)
    return pool(q)


def feature_set(wi, A, O, name='raw_readout'):
    wi, A, O = [np.asarray(w, dtype=float) for w in (wi, A, O)]
    if wi.ndim == 1:
        wi, A, O = wi[None], A[None], O[None]
    if name == 'raw20':
        return raw20(wi, A, O)
    if name == 'readoutgauge':
        return readout_gauge(wi, A, O)
    if name == 'impulse':
        return impulse_gauge(wi, A, O)
    q = raw20(wi, A, O)
    if name == 'raw_readout':
        return np.concatenate([q, readout_gauge(wi, A, O)], axis=1)
    if name == 'raw_impulse':
        return np.concatenate([q, impulse_gauge(wi, A, O)], axis=1)
    raise ValueError(name)


# Gauge guarantees assume the normalizing energies/norms are nonzero.
# Fallbacks define finite features for degenerate networks.
# Mixed raw+gauge features retain raw20's rescaling dependence.

def weight_fit(wi, A, O, accuracy, alpha=.01, gamma=.01):
    X = raw20(wi, A, O)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    Z = (X-mu)/sd
    n = len(Z)
    zz = np.sum(Z*Z, axis=1)
    D = np.maximum(zz[:, None]+zz[None]-2*Z@Z.T, 0)
    K = np.exp(-gamma*D)
    K.flat[::n+1] += alpha
    y = np.asarray(accuracy, dtype=float)
    offset = y.mean()
    return dict(
        mean=mu, scale=sd, train=Z, train_norm=zz,
        coefficient=np.linalg.solve(K, y-offset),
        offset=np.array(offset), gamma=np.array(gamma),
        alpha=np.array(alpha))


def weight_predict(model, wi, A, O):
    Z = (raw20(wi, A, O)-model['mean'])/model['scale']
    D = np.maximum(
        np.sum(Z*Z, axis=1)[:, None]+model['train_norm'][None]
        -2*(Z@model['train'].T), 0)
    return np.exp(-model['gamma']*D)@model['coefficient']+model['offset']


# ---------- Global summaries and train-only standardized regression ----------

def summarize(d, screen='none', outer=True):
    """Works on one-network dictionaries for streaming large designs."""
    k, nb = int(d['K']), int(d['NB'])
    m4, m3 = np.asarray(d['m4'], float), np.asarray(d['m3'], float)
    n, c = m4.shape
    scale = np.sqrt((m4*m4).mean(1, keepdims=True))
    scale = np.where(scale > 0, scale, 1.)
    a, b = m4/scale, m3/scale
    i4, i3 = np.asarray(d['I4'], float), np.asarray(d['I3'], float)
    f = np.stack([
        i4, i3, a, b, abs(a), abs(b),
        a/(.1+abs(a)), b/(.1+abs(b)),
        a/(.03+abs(a)), b/(.03+abs(b))
    ], -1)
    ids = np.arange(c)
    box, sl = ids//(720*k), ids % k
    groups = [np.zeros(c, int), sl]
    if outer and nb > 1:
        groups += [(box//nb**j) % nb for j in range(3)]
    out = [scale, np.log(scale),
           m4.mean(1, keepdims=True), m3.mean(1, keepdims=True)]
    for g in groups:
        out += [f[:, g == j].mean(1) for j in np.unique(g)]
    selected = np.zeros(c, bool)
    if screen != 'none':
        arr = np.array(list(itertools.permutations(range(10), 3)))
        arr = arr[(ids//k) % 720]
        if screen == 'all':
            selected[:] = True
        elif screen == 'late':
            selected = arr[:, 0] == 9
        elif screen.startswith('stride'):
            selected = ids % int(screen[6:]) == 0
        else:
            raise ValueError(screen)
        fl = np.asarray(d['fl'][:, selected], float)
        fm = np.asarray(d['fm'][:, selected], float)
        anchor = i4[:, selected, None]
        sm = fm/scale[:, :, None]
        g = np.stack(
            [(anchor*(1-fl[:, :, h::2])).mean(2) for h in range(2)] +
            [((1-anchor)*fl[:, :, h::2]).mean(2) for h in range(2)] +
            [(sm[:, :, h::2]/(.1+abs(sm[:, :, h::2]))).mean(2)
             for h in range(2)] +
            [(sm[:, :, h::2]-a[:, selected, None]).mean(2)
             for h in range(2)], -1)
        out.append(g.mean(1))
        for j in range(k):
            take = sl[selected] == j
            if take.any():
                out.append(g[:, take].mean(1))
    return np.concatenate(out, 1), selected


def ridge_grid(x, y, tr, va, alphas):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a, b = (x[tr]-mean)/std, (x[va]-mean)/std
    ym = y[tr].mean()
    if a.shape[1] <= len(tr):
        ev, v = eigh(a.T@a, check_finite=False)
        q = v.T@(a.T@(y[tr]-ym))
        left = b@v
    else:
        ev, v = eigh(a@a.T, check_finite=False)
        q = v.T@(y[tr]-ym)
        left = (b@a.T)@v
    return ym+(
        (left*q[None, :])[:, :, None] /
        (np.maximum(ev, 0)[:, None] +
         np.asarray(alphas)[None, :])[None]
    ).sum(1)


def fit_ridge(x, y, tr, alpha):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a = (x[tr]-mean)/std
    ym = y[tr].mean()
    coef = np.linalg.solve(
        a.T@a+alpha*np.eye(x.shape[1]), a.T@(y[tr]-ym))/std
    return coef, float(ym-mean@coef)


def rbf_grid(x, y, tr, va, alphas, gamma):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a, b = (x[tr]-mean)/std, (x[va]-mean)/std
    g = gamma/x.shape[1]
    kt = np.exp(-g*cdist(a, a, 'sqeuclidean'))
    kv = np.exp(-g*cdist(b, a, 'sqeuclidean'))
    ev, v = eigh(kt, check_finite=False)
    q = v.T@(y[tr]-y[tr].mean())
    return y[tr].mean()+((kv@v)*q) @ (
        1/(np.maximum(ev, 0)[:, None]+np.asarray(alphas)))


def fit_rbf(x, y, tr, alpha, gamma):
    mean = x[tr].mean(0)
    std = np.maximum(x[tr].std(0), .03)
    a = (x[tr]-mean)/std
    g = gamma/x.shape[1]
    kernel = np.exp(-g*cdist(a, a, 'sqeuclidean'))
    coef = np.linalg.solve(
        kernel+alpha*np.eye(len(tr)), y[tr]-y[tr].mean())
    return dict(mean=mean, std=std, centres=a, coef=coef,
                gamma=g, intercept=y[tr].mean())


def predict_rbf(model, x):
    z = (x-model['mean'])/model['std']
    return (
        np.exp(-model['gamma'] *
               cdist(z, model['centres'], 'sqeuclidean'))
        @ model['coef'] + model['intercept'])


def search(x, y, tr, va):
    options = [('ridge', 0.)]+[('rbf', g) for g in GAMMAS]
    values = []
    for family, gamma in options:
        values.append(
            ridge_grid(x, y, tr, va, ALPHAS) if family == 'ridge'
            else rbf_grid(x, y, tr, va, ALPHAS, gamma))
    return np.concatenate(values, 1), [
        (family, g, float(a))
        for family, g in options for a in ALPHAS]


def one(x, y, tr, va, setting):
    family, gamma, alpha = setting
    return (
        ridge_grid(x, y, tr, va, [alpha]) if family == 'ridge'
        else rbf_grid(x, y, tr, va, [alpha], gamma))[:, 0]


def nested(x, y):
    pop = np.arange(1024)
    pred = np.zeros(1030)
    settings = []
    for fold in range(8):
        tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
        loss = None
        for f in range(4):
            it, iv = tr[(tr//8) % 4 != f], tr[(tr//8) % 4 == f]
            p, opt = search(x, y, it, iv)
            e = ((p-y[iv, None])**2).sum(0)
            loss = e if loss is None else loss+e
        setting = opt[int(loss.argmin())]
        settings.append(setting)
        pred[va] = one(x, y, tr, va, setting)
    loss = None
    for f in range(8):
        tr, va = pop[pop % 8 != f], pop[pop % 8 == f]
        p, opt = search(x, y, tr, va)
        e = ((p-y[va, None])**2).sum(0)
        loss = e if loss is None else loss+e
    final = opt[int(loss.argmin())]
    pred[1024:] = one(x, y, pop, np.arange(1024, 1030), final)
    if final[0] == 'ridge':
        coef, intercept = fit_ridge(x, y, pop, final[2])
        model = dict(coef=coef, intercept=intercept)
    else:
        model = fit_rbf(x, y, pop, final[2], final[1])
    return pred, dict(outer=settings, final=final), model


def fit_readout(x, y, train, kernel='ridge', alpha=10., gamma=.1):
    if kernel == 'rbf':
        return fit_rbf(x, y, train, alpha, gamma)
    coef, intercept = fit_ridge(x, y, train, alpha)
    return dict(coef=coef, intercept=intercept)


def fast_rbf(model, x):
    z = (np.atleast_2d(x)-model['mean'])/model['std']
    a = model['centres']
    if 'centre_norm2' not in model:
        model['centre_norm2'] = np.sum(a*a, axis=1)
    norm = model['centre_norm2']
    distance = np.maximum(
        np.sum(z*z, axis=1)[:, None]+norm[None]-2*z@a.T, 0)
    return model['intercept']+np.exp(
        -model['gamma']*distance)@model['coef']


def readout(model, x, fast=False):
    if 'centres' in model:
        return (fast_rbf(model, x) if fast
                else predict_rbf(model, np.atleast_2d(x)))
    return np.atleast_2d(x)@model['coef']+model['intercept']


# ---------- Strict nested group selection and offline linear compilation ----------

def group_labels(arr, k, family='global'):
    arr = np.repeat(arr, k, axis=0)
    c = len(arr)
    groups = [np.zeros(c, int), np.tile(np.arange(k), 720)]
    if family in ('positions', 'pairs', 'cells'):
        order = np.argsort(arr, axis=1)
        groups += [arr[:, j] for j in range(3)]
        groups += [order[:, 0]*9+order[:, 1],
                   arr[:, 0]-arr[:, 1]+9]
    if family == 'pairs':
        groups += [arr[:, 0]*10+arr[:, 1],
                   arr[:, 0]*10+arr[:, 2],
                   arr[:, 1]*10+arr[:, 2]]
    if family == 'cells':
        groups += [np.arange(c)]
    return groups


def grouped_parts(d, w):
    m4, m3 = np.asarray(d['m4'], float), np.asarray(d['m3'], float)
    scale = np.maximum(
        np.sqrt((m4*m4).mean(1, keepdims=True)), 1e-12)
    a, b = m4/scale, m3/scale
    f = np.stack([
        d['I4'], d['I3'], a, b, abs(a), abs(b),
        a/(.1+abs(a)), b/(.1+abs(b)),
        a/(.03+abs(a)), b/(.03+abs(b))
    ], -1)
    return np.c_[w, scale, np.log(scale), m4.mean(1), m3.mean(1)], f


def grouped_features(d, w, family='global'):
    context, f = grouped_parts(d, w)
    arr = d.get('arr', np.array(
        list(itertools.permutations(range(10), 3))))
    out = [context]
    for group in group_labels(arr, int(d['K']), family):
        out.append(np.concatenate(
            [f[:, group == j].mean(1) for j in np.unique(group)], 1))
    return np.concatenate(out, 1)


def compile_groups(coef, arr, k, family='global'):
    """Fold all group means into one fixed cell coefficient table."""
    table = np.zeros((len(arr)*k, 10))
    pos = 24
    for group in group_labels(arr, k, family):
        for label in np.unique(group):
            ix = group == label
            table[ix] += coef[pos:pos+10]/ix.sum()
            pos += 10
    assert pos == len(coef)
    return table, np.asarray(coef[:24])


def compiled_predict(model, d, w):
    context, f = grouped_parts(d, w)
    return (
        model['intercept']+context@model['context_coef']+
        f.reshape(len(w), -1)@model['table'].ravel())


def nested_compiled(d, w, y):
    """Eight outer folds; four inner folds select all families and alpha."""
    pop = np.arange(1024)
    families = ['global', 'positions', 'pairs', 'cells']
    alphas = np.logspace(-1, 5, 13)
    losses = np.empty((8, 4, 13))
    grids = np.empty((4, 1024, 13))
    dims = {}
    for fi, family in enumerate(families):
        x = grouped_features(d, w, family)
        dims[family] = x.shape[1]
        for fold in range(8):
            tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
            loss = np.zeros(13)
            for inner in range(4):
                it = tr[(tr//8) % 4 != inner]
                iv = tr[(tr//8) % 4 == inner]
                pp = ridge_grid(x, y, it, iv, alphas)
                loss += np.sum((pp-y[iv, None])**2, axis=0)
            losses[fold, fi] = loss
            grids[fi, va] = ridge_grid(x, y, tr, va, alphas)
        print(f'Finished nested linear family {family}', flush=True)

    pred = np.empty(len(y))
    choices = []
    for fold in range(8):
        fi, ai = np.unravel_index(np.argmin(losses[fold]), (4, 13))
        va = pop[pop % 8 == fold]
        pred[va] = grids[fi, va, ai]
        choices.append(dict(
            fold=fold, family=families[fi], alpha=float(alphas[ai])))

    errors = ((grids-y[None, :1024, None])**2).mean(1)
    fi, ai = np.unravel_index(np.argmin(errors), (4, 13))
    family, alpha = families[fi], float(alphas[ai])
    x = grouped_features(d, w, family)
    pred[1024:] = ridge_grid(
        x, y, pop, np.arange(1024, len(y)), [alpha])[:, 0]
    if x.shape[1] <= 1024:
        coef, intercept = fit_ridge(x, y, pop, alpha)
    else:
        # Dual ridge avoids a huge feature Gram matrix.
        mean = x[pop].mean(0)
        sd = np.maximum(x[pop].std(0), .03)
        a = (x[pop]-mean)/sd
        ym = y[pop].mean()
        coef = (
            a.T@np.linalg.solve(
                a@a.T+alpha*np.eye(1024), y[pop]-ym))/sd
        intercept = float(ym-mean@coef)

    arr = d.get('arr', np.array(
        list(itertools.permutations(range(10), 3))))
    table, context = compile_groups(coef, arr, int(d['K']), family)
    model = dict(
        kind='compiled', k=int(d['K']), family=family, alpha=alpha,
        table=table, context_coef=context, intercept=intercept)
    return pred, dict(
        outer=choices, final_family=family, final_alpha=alpha,
        feature_dimensions=dims), model


# ---------- Fixed population-learned probe subsets and screening rows ----------

def six_features(I4, I3, m4, m3, scale):
    a, b = m4/scale, m3/scale
    return np.stack([
        I4, I3, a, b, a/(.1+abs(a)), b/(.1+abs(b))], -1)


def select_groups(f, base, y, train, n=8):
    A, G = f.shape[1:]
    v = f[train].reshape(len(train), -1)
    v = (v-v.mean(0))/np.maximum(v.std(0), .03)
    wx = base[train]-base[train].mean(0)
    wx /= np.maximum(wx.std(0), .03)
    u, s, _ = np.linalg.svd(wx, full_matrices=False)
    q = u[:, s > s.max()*1e-10]
    r = y[train]-y[train].mean()
    r -= q@(q.T@r)
    v -= q@(q.T@v)
    gram = np.einsum(
        'ncj,nck->cjk', v.reshape(-1, A, G),
        v.reshape(-1, A, G), optimize=True)
    corr = (v.T@r).reshape(A, G)
    cells = []
    for step in range(n):
        score = (
            corr*np.linalg.solve(
                gram+np.eye(G), corr[..., None])[..., 0]).sum(1)
        score[cells] = -np.inf
        j = int(score.argmax())
        cells.append(j)
        z = v[:, G*j:G*j+G]
        ev, vec = np.linalg.eigh(z.T@z)
        keep = ev > max(ev[-1]*1e-10, 1e-9)
        if not keep.any():
            break
        q = z@vec[:, keep]/np.sqrt(ev[keep])
        b, u = q.T@v, q.T@r
        r -= q@u
        v -= q@b
        bg = b.reshape(len(u), A, G)
        gram -= np.einsum('gcj,gck->cjk', bg, bg, optimize=True)
        corr -= (b.T@u).reshape(A, G)
    return np.array(cells)


def legacy_global(d, w):
    """Exact K1 global feature order for the learned-screen experiment."""
    m4, m3 = np.asarray(d['m4'], float), np.asarray(d['m3'], float)
    scale = np.maximum(
        np.sqrt((m4*m4).mean(1, keepdims=True)), 1e-12)
    a, b = m4/scale, m3/scale
    f = np.stack([
        d['I4'], d['I3'], a, b, abs(a), abs(b),
        a/(.1+abs(a)), b/(.1+abs(b)),
        a/(.03+abs(a)), b/(.03+abs(b))
    ], -1)
    return np.c_[
        w, scale, np.log(scale), m4.mean(1), m3.mean(1),
        f.mean(1), f.mean(1)]


def learn_selected(d, w, y, train, count=16, screens=False,
                   kernel='ridge', alpha=10., gamma=.1):
    if screens:
        base = legacy_global(d, w)
        scale = np.maximum(
            np.sqrt(np.mean(np.asarray(d['m4'], float)**2, 1)),
            1e-12)
        fm = np.asarray(d['fm'], float).reshape(len(w), -1)/scale[:, None]
        f = np.stack([
            d['fl'].reshape(len(w), -1),
            fm, fm/(.1+abs(fm))], -1)
    else:
        base = w
        scale = np.maximum(
            np.sqrt(np.mean(np.asarray(d['m4'][train], float)**2, 0)),
            .05)
        f = six_features(
            d['I4'], d['I3'], np.asarray(d['m4'], float),
            np.asarray(d['m3'], float), scale)
    chosen = select_groups(f, base, y, train, count)
    x = np.c_[base, f[:, chosen].reshape(len(w), -1)]
    model = fit_readout(x, y, train, kernel, alpha, gamma)
    if screens:
        model.update(selected=chosen)
    else:
        model.update(cells=chosen, scale=scale[chosen])
    model['kind'] = 'screens' if screens else 'subset'
    return model, x


def selected_cv(d, w, y, count=16, screens=False,
                kernel='ridge', alpha=10., gamma=.1):
    pop = np.arange(1024)
    pred, costs = np.empty(1030), np.empty(1030)
    for fold in range(8):
        tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
        model, x = learn_selected(
            d, w, y, tr, count, screens, kernel, alpha, gamma)
        pred[va] = np.clip(readout(model, x[va], fast=True), 0, 1)
        costs[va] = selected_cost(model)['total_with_readout']
    model, x = learn_selected(
        d, w, y, pop, count, screens, kernel, alpha, gamma)
    pred[1024:] = np.clip(readout(model, x[1024:], fast=True), 0, 1)
    costs[1024:] = selected_cost(model)['total_with_readout']
    return pred, costs, model


# ---------- Exact integration of the final Gaussian input ----------

def final_integral(h, wih, whh, wo, second, maximum, ps, pm):
    """P(correct | first nine inputs), using at most 18 breakpoints.

    h is the exact state after those inputs. ps/pm give their second/max
    positions, and second/maximum their values. Persistent ties use the
    lowest index. The final Gaussian input is integrated without samples.
    """
    h, wih, whh, wo = [
        np.asarray(v, np.float64) for v in (h, wih, whh, wo)]
    n = len(h)
    b = h@whh.T
    gate = np.divide(
        -b, wih, out=np.full_like(b, np.inf), where=wih != 0)
    events = np.column_stack([gate, second, maximum])
    order = events.argsort(-1, kind='stable')
    edges = np.column_stack([
        np.full(n, -np.inf),
        np.take_along_axis(events, order, 1),
        np.full(n, np.inf)])
    active = (
        (wih < 0)[None, :] |
        ((wih == 0)[None, :] & (b > 0)))
    slope = np.broadcast_to(
        np.minimum(wih, 0)@wo.T, (n, 10)).copy()
    bias = (active*b)@wo.T
    delta = (wo*np.abs(wih)).T
    ids = np.arange(n)
    target = np.asarray(ps).copy()
    start = np.full(n, np.nan)
    end = np.full(n, np.nan)
    answer = np.zeros(n)
    segments = np.zeros(n, int)
    for step in range(19):
        a = slope[ids, target, None]-slope
        c = bias[ids, target, None]-bias
        root = np.divide(-c, a, out=np.zeros_like(c), where=a != 0)
        lo = np.maximum(
            edges[:, step], np.where(a > 0, root, -np.inf).max(1))
        hi = np.minimum(
            edges[:, step+1], np.where(a < 0, root, np.inf).min(1))
        tied_loss = (
            (c == 0) &
            (np.arange(10)[None, :] < target[:, None]))
        impossible = (
            (a == 0) & ((c < 0) | tied_loss)).any(1)
        valid = (lo < hi) & ~impossible
        old = ~np.isnan(start)
        disjoint = valid & old & (lo > end)
        answer[disjoint] += (
            ndtr(end[disjoint])-ndtr(start[disjoint]))
        segments[disjoint] += 1
        reset = valid & (~old | disjoint)
        start[reset], end[reset] = lo[reset], hi[reset]
        join = valid & old & ~disjoint
        end[join] = np.maximum(end[join], hi[join])
        if step == 18:
            break
        event = order[:, step]
        target[event == 16] = 9
        target[event == 17] = np.asarray(pm)[event == 17]
        use = event < 16
        ix, g = ids[use], event[use]
        sign = np.sign(wih[g])
        slope[ix] += delta[g]
        bias[ix] += (sign*b[ix, g])[:, None]*wo[:, g].T

    use = ~np.isnan(start)
    answer[use] += ndtr(end[use])-ndtr(start[use])
    segments[use] += 1
    return answer, segments


def beta_rule(q, a, b):
    x, w = roots_jacobi(q, b-1, a-1)
    return (x+1)/2, w/w.sum()


def prefix_rule(q=1):
    """First-nine MF3 anchor with tensor Beta quadrature."""
    arr = np.array(list(itertools.permutations(range(9), 3)))
    rules = [
        beta_rule(q, 7, 3),
        beta_rule(q, 2, 1),
        beta_rule(q, 1, 1)]
    rows, weights = [], []
    for i, j, k in itertools.product(range(q), repeat=3):
        ur, vm, vs = rules[0][0][i], rules[1][0][j], rules[2][0][k]
        um = ur+(1-ur)*vm
        r, m, s = ndtri([ur, um, ur+(um-ur)*vs])
        mu = -np.exp(-r*r/2)/np.sqrt(2*np.pi)/ur
        x = np.full((504, 9), mu)
        x[np.arange(504)[:, None], arr] = np.array([m, s, r])
        rows.append(x)
        weights.extend([
            rules[0][1][i]*rules[1][1][j]*rules[2][1][k]/504
        ]*504)
    return np.concatenate(rows), np.array(weights)


def last_estimate(x, weight, wih, whh, wo, trie=None):
    if trie is None:
        trie = PrefixEvaluator(x)
    inp = trie.values[:, None]*wih
    _, v = trie.levels[0]
    h = np.maximum(inp[v], 0)
    for p, v in trie.levels[1:]:
        h = np.maximum((h@whh.T)[p]+inp[v], 0)
    h = h[trie.leaf]
    ranks = x.argsort(-1)
    ps, pm = ranks[:, -2], ranks[:, -1]
    second = x[np.arange(len(x)), ps]
    maximum = x[np.arange(len(x)), pm]
    probability, segments = final_integral(
        h, wih, whh, wo, second, maximum, ps, pm)
    n = len(x)
    prefix_mac = 256*sum(trie.counts[:-1])+16*len(trie.values)
    prefix_scalar = 16*(2*sum(trie.counts)-trie.counts[0])
    # Explicit allowance: normal CDF costs 100 scalar/MAC units each.
    sweep_ops = 6000*n+203*int(segments.sum())+336
    cost = (prefix_mac+prefix_scalar+sweep_ops+n)/2880
    return float(weight@probability), cost, float(segments.mean())


@lru_cache(maxsize=8)
def last_plan(q):
    x, weights = prefix_rule(q)
    return x, weights, PrefixEvaluator(x)


def last_features(w, q=1):
    x, weights, trie = last_plan(q)
    return last_estimate(
        x, weights, *[np.asarray(v, float) for v in w], trie)


# ---------- Target deployment and explicit operation allowances ----------

@lru_cache(maxsize=32)
def cell_plan(k=1, boxes=1, screen='none'):
    x, target, c = design(k, boxes, screen != 'none')
    r = 2 if screen == 'none' else 14
    mask = np.zeros((c, r), bool)
    mask[:, :2] = True
    if screen != 'none':
        ids = np.arange(c)
        arr = np.array(list(itertools.permutations(range(10), 3)))
        arr = arr[(ids//k) % 720]
        take = (
            arr[:, 0] == 9 if screen == 'late'
            else ids % int(screen[6:]) == 0
            if screen.startswith('stride')
            else np.ones(c, bool))
        mask[take, 2:] = True
    return (
        PrefixEvaluator(x.reshape(c, r, 10)[mask]),
        target.reshape(c, r)[mask], mask)


def target_cells(w, k=1, boxes=1, screen='none'):
    tree, target, mask = cell_plan(k, boxes, screen)
    ok, mg = bits_margins(tree(*w, dtype=np.float32), target)
    bits = np.zeros(mask.shape, np.uint8)
    margins = np.zeros(mask.shape, np.float32)
    bits[mask], margins[mask] = ok, mg
    d = dict(
        NB=boxes, K=k,
        I4=bits[None, :, 0], I3=bits[None, :, 1],
        m4=margins[None, :, 0], m3=margins[None, :, 1])
    if screen != 'none':
        d.update(fl=bits[None, :, 2:], fm=margins[None, :, 2:])
    return d, tree.cost()


@lru_cache(maxsize=32)
def selected_plan(indices, screens=False):
    x, target, _ = design(1, 1, screens)
    ids = np.asarray(indices)
    if screens:
        rows = np.r_[
            (14*np.arange(720)[:, None]+np.arange(2)).ravel(),
            14*(ids//12)+2+ids % 12]
    else:
        rows = (2*ids[:, None]+np.arange(2)).ravel()
    return PrefixEvaluator(x[rows]), target[rows]


def selected_online(model, w):
    screens = 'selected' in model
    ids = model['selected'] if screens else model['cells']
    tree, target = selected_plan(tuple(map(int, ids)), screens)
    ok, mg = bits_margins(tree(*w, dtype=np.float32), target)
    weights = raw20(*w)
    if screens:
        bits = ok[:1440].reshape(720, 2)
        margins = mg[:1440].reshape(720, 2)
        d = dict(
            I4=bits[None, :, 0], I3=bits[None, :, 1],
            m4=margins[None, :, 0], m3=margins[None, :, 1])
        base = legacy_global(d, weights)
        scale = max(
            np.sqrt(np.mean(margins[:, 0].astype(float)**2)),
            1e-12)
        sm = mg[1440:].astype(float)/scale
        f = np.stack(
            [ok[1440:], sm, sm/(.1+abs(sm))], -1)
    else:
        bits, margins = ok.reshape(-1, 2), mg.reshape(-1, 2)
        base = weights
        f = six_features(
            bits[:, 0], bits[:, 1],
            margins[:, 0], margins[:, 1], model['scale'])
    return np.c_[base, f.reshape(1, -1)]


def selected_cost(model):
    screens = 'selected' in model
    ids = model['selected'] if screens else model['cells']
    tree, _ = selected_plan(tuple(map(int, ids)), screens)
    cost = tree.cost()
    ops = (
        40*720+5000+12*len(ids) if screens
        else 20*len(ids)+5000)
    if 'centres' in model:
        n, dim = model['centres'].shape
        ops += n*(dim+38)+4*dim+1
    cost['other_readout_equiv'] = ops/2880
    cost['total_with_readout'] = cost['total_forward_equiv']+ops/2880
    return cost


def predict_network(w, model, k=1, boxes=1, screen='none',
                    context='raw20', last_q=0):
    kind = str(model.get(
        'kind',
        'subset' if 'cells' in model else
        'screens' if 'selected' in model else
        'weight' if 'train' in model else
        'compiled' if 'table' in model else 'summary'))

    if kind in ('subset', 'screens'):
        value = readout(model, selected_online(model, w), fast=True)[0]
        return float(np.clip(value, 0, 1)), selected_cost(model)

    if kind == 'weight':
        value = float(weight_predict(model, *w)[0])
        return value, dict(
            logical_rows=0,
            total_with_readout=(2681+58*len(model['train']))/2880)

    if kind == 'compiled':
        d, cost = target_cells(w, int(model.get('k', k)))
        extra = (40*len(d['m4'][0])+5000)/2880
        cost.update(
            other_readout_equiv=extra,
            total_with_readout=cost['total_forward_equiv']+extra)
        return float(compiled_predict(model, d, raw20(*w))[0]), cost

    k = int(model.get('k', k))
    boxes = int(model.get('boxes', boxes))
    screen = str(model.get('screen', screen))
    context = str(model.get('context', context))
    d, cost = target_cells(w, k, boxes, screen)
    f, selected = summarize(d, screen)
    name = 'readoutgauge' if context == 'function_invariant' else context
    feat = np.c_[
        feature_set(*w, name),
        f[:, 4:] if context == 'function_invariant' else f]
    last_q = int(model.get('last_q', last_q))
    if last_q:
        probability, c, segments = last_features(w, last_q)
        feat = np.c_[feat, probability, segments]
        cost['last_input_forward_equiv'] = c

    extra = (
        (40+30*(boxes > 1))*len(d['m4'][0])+
        240*int(selected.sum())+5000+FEATURE_COST[name]-2600)
    if 'centres' in model:
        n, dim = model['centres'].shape
        ops = 2*n*dim+34*n+2*dim+1
    else:
        ops = feat.shape[1]+1
    cost.update(
        feature_forward_equiv=extra/2880,
        learned_readout_forward_equiv=(ops+2)/2880)
    cost['total_with_readout'] = (
        cost['total_forward_equiv']+(extra+ops+2)/2880+
        cost.get('last_input_forward_equiv', 0))
    return float(np.clip(readout(model, feat)[0], 0, 1)), cost


# ---------- Command-line fitting, regeneration, deployment, and scoring ----------

def save_model(path, model):
    np.savez(path, **model)


def score(pred, truth, indices, costs=None):
    e = pred[indices]-truth[indices]
    result = dict(
        rms_pp=float(100*np.sqrt(np.mean(e*e))),
        mean_pp=float(100*e.mean()))
    if costs is not None:
        c = np.broadcast_to(costs, truth.shape)[indices]
        sampling = np.mean(truth[indices]*(1-truth[indices])/c)
        result.update(
            mean_cost=float(c.mean()),
            sampling_rms_pp=float(100*np.sqrt(sampling)),
            mse_ratio=float(np.mean(e*e)/sampling))
    return result


def generate_cache(w, k, boxes, screen, summary_only=False):
    rows = []
    for n in range(len(w[0])):
        d, cost = target_cells([v[n] for v in w], k, boxes, screen)
        rows.append(summarize(d, screen)[0] if summary_only else d)
        if n % 128 == 127:
            print(f'Computed {n+1}/{len(w[0])} networks', flush=True)
    if summary_only:
        return dict(
            NB=boxes, K=k, summary=np.concatenate(rows)), cost
    keys = [key for key in rows[0] if key not in ('NB', 'K')]
    return dict(
        NB=boxes, K=k,
        **{key: np.concatenate([d[key] for d in rows])
           for key in keys}), cost


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=[
        'predict', 'cache', 'summary', 'compiled',
        'subset', 'screens', 'weight', 'last'])
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--k', type=int, default=1)
    p.add_argument('--boxes', type=int, default=1)
    p.add_argument('--screen', default='none')
    p.add_argument('--context', default='raw20')
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--kernel', choices=['ridge', 'rbf'], default='ridge')
    p.add_argument('--alpha', type=float)
    p.add_argument('--gamma', type=float)
    p.add_argument('--cv', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    p.add_argument('--q', type=int, default=1)
    p.add_argument('--last', type=Path)
    p.add_argument('--last-q', type=int, default=0)
    a = p.parse_args()

    if a.command == 'predict':
        value, cost = predict_network(
            load_weights(a.source), dict(np.load(a.output)),
            a.k, a.boxes, a.screen, a.context, a.last_q)
        print(json.dumps(
            dict(estimated_accuracy=value, cost=cost), indent=2))
        return

    a.output.parent.mkdir(parents=True, exist_ok=True)
    w = load_models(a.source)
    y = np.load(a.source/'pop1k/reference.npz')['acc']
    pop, zoo = np.arange(1024), np.arange(1024, 1030)

    if a.command == 'cache':
        data, cost = generate_cache(
            w, a.k, a.boxes, a.screen, a.summary_only)
        np.savez(a.output, **data)
        print(json.dumps(cost, indent=2))
        return

    if a.command == 'last':
        values = np.array([
            last_features([v[n] for v in w], a.q)
            for n in range(1030)])
        np.savez(
            a.output, q=a.q, pred=values[:, 0],
            cost=values[:, 1], segments=values[:, 2])
        return

    costs = None
    if a.command == 'weight':
        alpha, gamma = a.alpha or .01, a.gamma or .01
        pred = np.empty(1030)
        if a.cv:
            for fold in range(8):
                tr, va = pop[pop % 8 != fold], pop[pop % 8 == fold]
                m = weight_fit(
                    *(v[tr] for v in w), y[tr], alpha, gamma)
                pred[va] = weight_predict(
                    m, *(v[va] for v in w))
        model = weight_fit(
            *(v[pop] for v in w), y[pop], alpha, gamma)
        pred[zoo] = weight_predict(
            model, *(v[zoo] for v in w))
        if not a.cv:
            pred[pop] = weight_predict(
                model, *(v[pop] for v in w))
        model['kind'] = 'weight'
        costs = np.r_[
            np.full(
                1024, (2681+58*(896 if a.cv else 1024))/2880),
            np.full(6, (2681+58*1024)/2880)]
    else:
        cache = a.cache or (
            a.source /
            f'pop1k/seed0_z46_cells_b{a.boxes}_k{a.k}.npz')
        d = dict(np.load(cache))
        if a.command == 'compiled':
            if int(d['NB']) != 1:
                raise ValueError(
                    'The four-family compiled design requires one outer box')
            pred, settings, model = nested_compiled(d, raw20(*w), y)
            costs = predict_network(
                [v[0] for v in w], model)[1]['total_with_readout']
            a.cv = True
        elif a.command in ('subset', 'screens'):
            if int(d['K']) != 1 or int(d['NB']) != 1:
                raise ValueError(
                    'Learned subsets/screens require the K1 one-box design')
            base = raw20(*w)
            alpha = a.alpha or (
                10. if a.kernel == 'ridge' else .01)
            gamma = a.gamma or .1
            if a.cv:
                pred, costs, model = selected_cv(
                    d, base, y, a.count, a.command == 'screens',
                    a.kernel, alpha, gamma)
            else:
                model, x = learn_selected(
                    d, base, y, pop, a.count, a.command == 'screens',
                    a.kernel, alpha, gamma)
                pred = np.clip(readout(model, x, fast=True), 0, 1)
                costs = selected_cost(model)['total_with_readout']
        else:
            f = (
                d['summary'] if 'summary' in d
                else summarize(d, a.screen)[0])
            name = (
                'readoutgauge'
                if a.context == 'function_invariant' else a.context)
            x = np.c_[
                feature_set(*w, name),
                f[:, 4:] if a.context == 'function_invariant' else f]
            if a.last:
                last = np.load(a.last)
                x = np.c_[x, last['pred'], last['segments']]
            if a.cv:
                pred, settings, model = nested(x, y)
            else:
                tr = pop[pop % 8 != 0]
                dv = pop[pop % 16 == 0]
                grid, choices = search(x, y, tr, dv)
                setting = choices[int(
                    np.mean((grid-y[dv, None])**2, 0).argmin())]
                model = fit_readout(
                    x, y, tr, setting[0], setting[2], setting[1])
                pred, settings = readout(model, x), setting
            model.update(
                kind='summary', k=int(d['K']), boxes=int(d['NB']),
                screen=a.screen, context=a.context)
            if a.last:
                model['last_q'] = (
                    int(last['q']) if 'q' in last else a.q)

    save_model(a.output, model)
    np.save(
        a.output.with_name(a.output.stem+'_pred.npy'), pred)
    report = dict(
        population=score(pred, y, pop, costs),
        zoo=score(pred, y, zoo, costs),
        zoo_errors_pp=(100*(pred[zoo]-y[zoo])).tolist(),
        cross_fitted=a.cv)
    if a.command in ('summary', 'compiled'):
        report['selection'] = settings
    if costs is not None:
        np.save(
            a.output.with_name(a.output.stem+'_cost.npy'),
            np.broadcast_to(costs, y.shape))
    a.output.with_suffix('.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
```

The small development-only affine calibration used in the exact-last table can be reproduced with:

```python
import numpy as np
from q6_methods import score

reference = np.load("algzoo-432-interp/pop1k/reference.npz")["acc"]
last = np.load("last_q3.npz")  # Generated by the "last" command above.
dev = np.arange(0, 1024, 16)
held = np.arange(8, 1024, 16)
zoo = np.arange(1024, 1030)

x = np.column_stack([np.ones(1030), last["pred"]])
coef = np.linalg.lstsq(x[dev], reference[dev], rcond=None)[0]
calibrated = x @ coef
cost = last["cost"] + 2 / 2880

print("Coefficients:", coef)
print("Heldout:", score(calibrated, reference, held, cost))
print("Zoo:", score(calibrated, reference, zoo, cost))
print("Six zoo errors, points:",
      100 * (calibrated[zoo] - reference[zoo]))
```