# Round 7 independent audit

## Verdict

The population-supervised estimator is valid under the question's stated rules, and the low-budget advantage survives the audit. The nested train/test separation in the readout is correct. The advertised FE values describe a readily achievable **compiled deployment**, but two optimizations claimed in the accounting are missing from the general-purpose evaluation helpers. These should be implemented before describing the current helpers themselves as executing at those budgets.

## Validation and scoring

- `gfeat.fold_preds` fits the feature mean/standard deviation and the response mean on `tr` only; `gfeat.pick` chooses family/gamma/alpha from four inner training folds; `crossfit` never fits or selects on its outer held-out labels. Per-network RMS normalization and within-network weight summaries are legitimate target computations, not population leakage. The `tgt` fields in the zoo cell caches are not used by G.
- Training scripts generate a separate `(network,batch,time)` input block and initialization for each network. The stacked Adam/AdamW operation is elementwise, and the loss is a sum of per-model means. There is no model interaction in the training update.
- The exact fixed-cost score is `mean((prediction-p)**2) * C / mean(p*(1-p))`. If deployment cost varies between networks/folds, use `mean(p*(1-p)/C_i)` in the denominator. Do not substitute an average budget for the individual budgets.
- The shared label input bank creates correlated label noise across networks. This is not train/test network leakage, but validation is conditional on the finite bank and the learned readout can partly predict its particular integration error. Individual label variances must not simply be subtracted as though independent of training labels. A second independent bank is the clean final check. The mean individual label standard errors are 0.01417 points (unif) and 0.02567 points (zoo), far below present residuals; they cannot plausibly account for an order-one advantage.
- Exact label-bank statistics: unif mean accuracy 90.5552%, SD 3.6979 points, range 70.7757–96.2795%, mean `p(1-p)=0.08415976`; zoo 92.3930%, SD 3.4338 points, range 78.6045–97.1245%, mean `p(1-p)=0.0691045116264`.
- `gfeat.pick` constructs inner folds using `tr*7919 % len(y)`. Thus appending unused evaluation networks or changing a population's global row offset changes the CV folds and sometimes the selected hyperparameters. No held-out values leak through this, but it makes the result unnecessarily layout-dependent. Define inner folds using stable IDs restricted to the training population.
- Nested hyperparameter selection evaluates a prespecified feature/design family. Repeatedly picking new families after inspecting the same population's outer-CV losses adds research-selection optimism. The additional 64-network sets have also been viewed repeatedly and should not be called an untouched final test set. A fresh seed batch with a frozen chosen method is the appropriate confirmatory experiment.
- The mystery's batch size is unknown. Batch 64 is a useful recipe-shift diagnostic; it is not another exact batch-128 holdout. Its systematic bias is real and should remain separate in tables.
- Charging 1,024 supports while outer folds train on 896 is conservative. A selected primal ridge model would be even cheaper than the RBF allowance. FE is an arithmetic accounting convention; it is not a measured hardware latency claim.

## Compute accounting

Let `n_t` be the number of distinct input prefixes at depth t, `v` the number of distinct scalar input values and `n_10` the number of leaves. For the specified optimized trie, the charged network cost is

`[256*sum(n_1,...,n_9) + 16*v + 160*n_10 + 16*sum(n_2,...,n_10) + 16*sum(n_1,...,n_10) + 20*n_10]/2880`.

This is correct for one recurrent product per parent state, cached scalar input products, one linear output per distinct leaf, and the stated scalar-operation convention. Counting input prefixes independently reproduces every value in `z48`:

| K | Distinct prefixes by depth | Physical rows | Network FE | G total FE |
|---|---|---:|---:|---:|
| 1 | 6,26,83,196,354,543,814,1112,1440,1440 | 1,440 | 563.4 | 618.545486 |
| 2 | 7,36,131,336,634,987,1507,2080,2736,2880 | 2,880 | 1,057.4 | 1,129.663542 |
| 4 | 9,56,227,616,1194,1875,2893,4016,5328,5760 | 5,760 | 2,045.4 | 2,151.899653 |

**Missing input-product cache.** In `gfeat.summary`, `inp=v[None,:,None]*Wc[0][:,None,:]` recomputes 16 products at every node. The formula charges 16 products per distinct scalar, globally. The difference is `16*(sum(n_t)-v)/2880`: **33.37778 / 62.92778 / 122.02778 FE** for K=1/2/4. Precomputing the small `value × hidden-unit` table and gathering by token makes the stated budget real. Recurrence sharing is already implemented correctly.

**Missing compiled readout.** `fit_predict` calls `fold_preds` on target rows. That helper evaluates ridge and all six gamma values with all nine alpha values, and performs `K_target,train @ V` before contracting with coefficients. Its six `n × n` eigenvector multiplications alone cost `6*1024**2/2880 = 2,184.533 FE` per target. They are absent from the proposed readout charge. All needed eigenvector contractions can instead be done offline after model selection: `beta=V @ ((V.T @ (y-ybar))/(max(eigenvalues,0)+alpha))`. Deployment then computes just `ybar + kernel(target,train) @ beta`. Training standardization and support squared norms are fixed offline. For a ridge choice, compile one primal coefficient vector. The smaller charged complexity is correct for this compiled model and yields the same mathematical predictions.

**Small-operation allowances.** The RBF formula includes a fixed cost for exponentials through its `34*n_support` term; it has not omitted them. This is a convention, not a portable exact instruction count. The 5,000-operation statistics/standardization allowance and 40 operations per cell are plausible for an implementation reusing norms, absolute values, reductions and divisors. An explicit primitive-operation ledger, or hardware timing measured separately, would make the claim more precise. Neither requires charging population fitting to each new target.

**Duplicate features.** For K=1 the ten global means and ten slice means are identical. For every K, a global mean is a known linear combination of the slice means. Simply deleting redundant dimensions changes the RBF distance and regularization geometry. They can be compressed exactly only if the induced metric is preserved (for K=1, multiply the retained standardized copy by sqrt(2), retaining the original gamma denominator). The K=1 z49 ablation used 34 dimensions while gfeat uses 44; part of the reported 0.58–0.61 range is this metric difference, not just outer-fold variation.

**Ties/numerics.** The caches preserve correctness bits from argmax, including lowest-index ties. A zero signed margin alone does not determine correctness, so new code should not replace a cached bit with `margin>0` at ties. The published z48 audit shows no bit differences at K=1/2 and only five near-boundary differences in 5.93M K=4 rows. Recomputing unif summaries from saved float32 margins reproduces cached `F` to 5.4e-15. The scale/log features would need a floor for degenerate all-zero networks, but none occur in the defined populations.

## Independently reproduced baseline

I extracted the unchanged NumPy/SciPy readout definitions from `gfeat.py`, since torch is absent from the audit runtime. I used the first 1,024 networks of each recipe, outer folds `default_rng(0).permutation(1024)%8`, and the inner fold denominator **1,024**. This fixes the layout issue. Hyperparameters for the extra networks are selected by the existing eight-fold `fit_predict` procedure, then refit on all 1,024. Errors are points, and each row uses its declared total cost. Full numeric data and predictions are in `audit_baselines.json`, `audit_unif_k*.npz` and `audit_zoo_k*.npz`.

| Recipe | K | Total FE | Cross-fit RMS | Mean | MSE/sampling |
|---|---:|---:|---:|---:|---:|
| Unif | 1 | 618.5455 | 0.963552 | +0.000716 | 0.682366 |
| Unif | 2 | 1,129.6635 | 0.678018 | +0.017069 | 0.617059 |
| Unif | 4 | 2,151.8997 | 0.583058 | +0.009817 | 0.869242 |
| Zoo | 1 | 618.5455 | 0.596731 | +0.008966 | 0.318730 |
| Zoo | 2 | 1,129.6635 | 0.515027 | +0.008144 | 0.433614 |
| Zoo | 4 | 2,151.8997 | 0.434513 | +0.005657 | 0.587925 |

| K | Extra b128 RMS / mean / ratio | b64 RMS / mean / ratio | Mystery error |
|---|---|---|---:|
| 1 | 0.875220 / +0.143340 / 0.560251 | 1.268588 / +0.804080 / 1.195655 | -0.552960 |
| 2 | 0.562636 / +0.092919 / 0.422844 | 0.929005 / +0.490470 / 1.171058 | -0.367031 |
| 4 | 0.529026 / +0.079661 / 0.712118 | 0.635267 / +0.253933 / 1.043104 | -0.065831 |

Zoo seeds 0–4 errors:

- K=1: +0.509921, -0.037216, -0.179646, -0.155464, +0.120748.
- K=2: -0.487352, +0.477107, -0.004854, -0.408766, -0.095646.
- K=4: -0.506469, +0.806992, -0.187894, -0.638401, -0.450408.

The external mystery-recipe numbers reproduce z55/z58 essentially exactly. Zoo K4 errors differ by at most about 0.002 points because the cached cells use the full-forward floating-point path while gfeat's summary uses the trie. Fixed-population nested folds can move unif K2/K4 RMS by about 0.03 points relative to the historical logs. Thus 'about 0.01 between fold schemes' describes particular checked schemes, not a general uncertainty bound. Compare every new method to its own paired baseline on the same fixed folds.

The six cached baseline refits took 177 seconds with one BLAS thread. No target sampling, quasi-random design or weight-noise networks were used.

### Reconciliation of the historical K4 values

For unif K4 I additionally reconstructed z55's historical metadata layout (population at rows 1024–2047, total array length 2822; unused rows were inert placeholders and were never used for fitting or selection). This gives **0.5570593673** RMS and **+0.0045167300** mean, reproducing the reported 0.557. Using the full 1153-row z59 cache with the population at rows 0–1023 gives **0.5498833534** RMS and **+0.0019050539** mean, reproducing the reported 0.550. Using stable 1024-only inner-fold IDs gives 0.583058. This directly identifies the split-layout cause of the discrepancy and confirms that the historical numerical values were not fabricated or transcribed incorrectly. These two checks took another 50 seconds; neither evaluated or selected on any weight-noise network.
