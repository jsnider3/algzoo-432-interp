# Cached-cell summary experiments

The runnable NumPy/SciPy program is `aggregates.py`; run it from the extracted
project directory with `--recipe unif|zoo --K 1|2|4`. It reads only deterministic
cell caches, population weights, and accuracy labels; it never uses the
within-cell sampled `tgt` arrays or weight-noise replicas.

## Prespecified features

The original 44/54/74-dimensional `gfeat` feature layout is retained, including
the duplicated K=1 global and slice averages. Write a=m4/RMS(m4) and b=m3/RMS(m4).
All new features are per S slice.

* **Tails**, 15K numbers: each margin's CDF at −0.1, −0.03, 0.01, 0.03, 0.1,
  0.3, followed by mean |a−b|, mean (a−b)^2, and mean(I4≠I3). They describe
  the shape of the vulnerable-margin population and sensitivity to changing
  the lower-input mean field.
* **Temporal groups**, 24K numbers: averages of I4, I3, a/(0.1+|a|), and
  b/(0.1+|b|), separately for M last, S last, M immediately after S, M after
  S, R after S, and |position(M)−position(S)|≥5. These distinguish late
  handoff failures, direct second-max writes, and long retention intervals.
* **Both** concatenates the two blocks.

The NPZ output stores X (base), T (tails), and H (temporal groups). T is ordered
by slice then feature. H is ordered by group, then slice, then the four channels.

## Validation and model selection

Every result uses all 1,024 networks in eight outer folds,
`default_rng(0).permutation(1024) % 8`. Within each outer training fold,
four inner folds select the kernel width, penalty, and the added block's
weight, including a baseline-only option. The inner assignment uses
`tr*7919 % 1024`, independent of external test rows. Final external predictions
use an eight-fold search on the 1,024 training networks and then one refit.

After training-only standardization, the RBF metric is

    exp[-gamma * (D_base/d_base + weight*D_extra/d_extra)].

`weight` is 0.1 or 1 for each added block, or zero for the baseline.
The original six gammas and nine penalties are used. Baseline ridge is an
additional candidate. The `tails`, `groups`, and `both` algorithms each select
between their own weighted kernels and baseline. The `select` algorithm also
selects which of the three feature blocks to use, fully within inner folds.
Thus this feature-family choice does not use outer errors. Comparing the
separately reported algorithms after observing their outer errors would still
be exploratory project-level selection.

## Arithmetic

The added tail features require at most 32 operations per cell plus 15K
normalizations when the already computed a,b and baseline softsigns are reused.
The six temporal masks are fixed offline. Their sizes sum to 1,176, giving
4,704K accumulation operations and 24K normalizations. Together the blocks
add 27,783K scalar operations, or 9.647K FE. The added 39K readout dimensions
also increase the full-support RBF's target inference cost by roughly 27.7K
FE. We conservatively charge the entire candidate block even when an inner
fold chooses baseline, so each algorithm has a fixed declared budget.

The batch cache-analysis script need not itself be a low-arithmetic deployment:
training, hyperparameter fitting, and repeated cache transformations are offline.
The declared deployed extractor shares the normalized margins and baseline
softsigns and stores the chosen coefficients/supports. No fitting or eigensolve
is performed on the target network.

Results and run times are in the corresponding JSON files. All uncertainty
statements should distinguish model-label noise from the finite network sample.

## Results before pulse combinations

All RMS values below are percentage points. Costs use 1,024 deployed supports,
including the candidate block's arithmetic even if baseline is selected.

| Recipe/K | Algorithm | FE | Crossfit RMS | MSE/sampling |
|---|---|---:|---:|---:|
| Unif/1 | Baseline | 618.56 | 0.96355 | 0.68238 |
| Unif/1 | Temporal groups | 637.29 | 0.89991 | 0.61325 |
| Unif/1 | Nested block choice | 655.98 | 0.90155 | 0.63353 |
| Unif/2 | Baseline | 1129.68 | 0.67802 | 0.61707 |
| Unif/2 | Temporal groups | 1167.15 | 0.62582 | 0.54316 |
| Unif/2 | Both | 1204.52 | 0.61066 | 0.53371 |
| Unif/2 | Nested block choice | 1204.52 | 0.61617 | 0.54339 |
| Unif/4 | Baseline | 2151.93 | 0.58306 | 0.86925 |
| Unif/4 | Temporal groups | 2226.86 | 0.51717 | 0.70770 |
| Unif/4 | Both / nested block choice | 2301.61 | 0.51252 | 0.71836 |
| Zoo/1 | Baseline | 618.56 | 0.59673 | 0.31874 |
| Zoo/1 | Temporal groups | 637.29 | 0.71774 | 0.47508 |
| Zoo/1 | Nested block choice | 655.98 | 0.71338 | 0.48309 |

Tails alone do not improve the compute-adjusted score at Unif K=1 or 2, but
they help at K=4 (0.52851 points at 2226.68 FE; ratio 0.73903). They can therefore
be complementary to temporal groups once extreme S slices are available.
The base-versus-improved external-network results are all retained in JSON,
including cases where the mystery-model error increases.

### Zoo failure diagnosis

The Zoo K=1 aggregate deterioration is concentrated in network index 994,
whose reference accuracy is 78.605%. Baseline error is +4.210 points; the
nested block-choice model errs by +13.694 points. This one network explains
108.5% of the increase in total squared error. Descriptively removing it would
change RMS from 0.58233 to 0.57106, but **the reported population result keeps
it**. This is consistent with a distance-based predictor reverting toward the
population mean when a rare poorly performing network is far from training
networks in the appended coordinates.

A natural follow-up is an additive base-plus-mechanism kernel, or a calibrated
base predictor plus an OOF-trained residual model, retaining extrapolation from
the original features. That idea has not been tested in this experiment.

## Pulse combinations and MF4-only design

`aggregates_pulse.py` appends the mechanical agent's ten signed-pulse features
to the base, then selects among the base kernel and temporal-group kernels
at weights 0.1 or 1, within exactly the same nested folds. The pulse computation
has a six-FE allowance. Its cache is `mechanics_pulse_unif.npz`, key `pulse`.

For `--mf4`, the script keeps only MF4 context/summary channels and H's I4 and
soft4 channels, giving 12K temporal features. Its target computation needs
no MF3 input evaluations. The original population cache is used only for
offline fitting convenience. The MF4 features reconstructed this way match
the independently implemented `cost_experiments.features(..., kind='4')`
to 6.6×10^−14 on all 1,153 K=2 networks.

The all-MF4 prefix cost is exactly 264.5K+37.9 FE. The base summary costs
22 operations per cell plus the existing 5,000-op allowance. H4 costs
2,352K additions plus 12K normalizations (half the original H cost).

| Design | FE | RMS | Mean error | MSE/sampling | Extra b128 RMS | Extra b64 RMS | Mystery error |
|---|---:|---:|---:|---:|---:|---:|---:|
| G(1,1)+pulse+H | 650.42 | 0.78537 | +0.01422 | 0.47668 | 0.69558 | 1.00422 | −0.31583 |
| G(1,2)+pulse+H | 1180.27 | 0.58706 | +0.00250 | 0.48332 | 0.48835 | 0.79585 | −0.64245 |
| G(1,4)+pulse+H | 2239.98 | 0.51624 | +0.00801 | 0.70933 | 0.49583 | 0.58168 | −0.03660 |
| MF4 K=2 + pulse | 631.91 | 0.66252 | +0.01376 | 0.32957 | 0.62982 | 0.83883 | −0.35801 |
| MF4 K=2 + pulse + H4 | 650.64 | 0.60603 | +0.00452 | 0.28394 | 0.54325 | 0.83083 | −0.44528 |
| MF4 K=4 + pulse | 1179.03 | 0.59795 | +0.00313 | 0.50091 | 0.53904 | 0.68757 | −0.14177 |
| MF4 K=4 + pulse + H4 | 1216.50 | 0.51548 | +0.00256 | 0.38409 | 0.48331 | 0.58506 | +0.00017 |

Every number uses 1,024-support deployed cost. Refit tuning for the external
networks uses index-mod-8 folds on the 1,024 training networks, whereas root's
cost-study refit reuses the random outer-fold candidate predictions; this
can yield different external errors for otherwise identical feature sets.
Their outer-fold protocol is the same.

The nearly zero last mystery error is far below the accuracy-label noise and
should not be interpreted as estimating true accuracy to that precision.
These combination choices were motivated by earlier results on the same
population and extra groups. Hyperparameters remain nested, but the broader
research process is exploratory and the extra groups are no longer pristine
for an unrestricted new claim of model selection.

The pulse experiments took 78–86 seconds each on one BLAS thread; the fuller
seven-metric aggregate searches took 210–237 seconds each. All model fitting
is offline. This experiment used no GPU and trained no new RNNs.
