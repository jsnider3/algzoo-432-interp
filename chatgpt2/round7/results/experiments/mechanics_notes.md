# Round 7: mechanically motivated input-pulse features

## Candidate and interpretation

For each target network, propagate two exactly designed trajectories starting from
`h0 = ReLU(+Wih)` and `h0 = ReLU(-Wih)`, then repeatedly apply `ReLU(Whh h)` nine
times. At each age a = 0,...,9, decode with Wo. These are the responses to a single
positive or negative unit input amid zeros, with zero padding before the impulse
understood. There are 20 logical length-10 probe rows but only 18 nontrivial
recurrent matrix-vector updates, because the zero prefix is known and the two
post-impulse histories are reused.

For each sign, compute the margin of the pulse's age label 9-a against the best
other logit, normalize it by RMS centered logits, and retain its mean over all
ages, its mean over late ages 5,...,9, and the fraction of ages at which the
readout chooses that age label. Also retain mean and SD of successive log output
amplitude changes. These are 10 features total; their exact column order is in
`mechanics.py:pulse_features`.

The age-alignment fraction is **not task accuracy** on these rows: the zero
coordinates create ties, and the pulse generally need not be the second maximum.
It tests how a simple known transient is retained and decoded. Positive pulses
probe an isolated large observation; negative pulses followed by zeros also
probe how the dynamics recover when subsequent inputs supersede that observation.
The features capture finite-horizon gated dynamics and are more faithful to
reachable histories than unrestricted eigenvalues or arbitrary hidden-basis
injections. Their usefulness does not prove they isolate a causal clock or
handoff mechanism.

## Cost

Budget the pulse extractor at 6 FE = 17,280 MAC-equivalent operations. A direct
implementation uses:

- initialization: 16 sign changes and 32 ReLUs;
- recurrent dynamics: 9 times 2 times 256 = 4,608 MACs, plus 288 ReLUs;
- 20 output vectors: 20 times 160 = 3,200 MACs;
- logit centering, normalization, 10-way age alignment, margins, two log-amplitude
  trajectories, and the final 10 means/SDs.

The latter arithmetic can be bounded by fewer than 3,000 scalar operations even
assigning each log and square root 32 ordinary operations. The whole extractor
is under 11,200 operations (3.89 FE) under that convention; 6 FE is conservative,
including the feature normalizations. Keep the full-kernel query cost for the
10 new coordinates separate: 2*1024*10 + 2*10 = 20,500 operations = 7.1181 FE.
Thus adding the pulse to full G costs at most 13.1181 FE with the supplied cost
model. A reduced or linear readout can make the query increment much smaller.

The pulse-only routine and the pulse columns of the joint basis/pulse extractor
were identical on every unif network. A fixed positive diagonal hidden-unit
rescaling and a 3x logit rescaling changed no pulse feature by more than
2.96e-13 (float64). This is an exact-symmetry check, not weight-noise evaluation.

## Validation

All pulse and basis summaries were specified before viewing their scores.
Each recipe has a separate readout. Outer folds are
`default_rng(0).permutation(1024) % 8`. Inner folds fix the denominator at 1024,
as in z49, instead of letting unrelated appended arrays alter it. Readout
family/gamma/penalty use the complete supplied grid and four inner folds. The
extra batches and released networks never enter a fit or hyperparameter choice.
Full-refit parameters use the same eight-fold selection on the first 1024 as
gfeat.fit_predict. Labels and summaries are the supplied caches.

The stats-plus-pulse-only option was specified after seeing the first pulse
augmentation result. Treat this extra model family and choosing among all tested
families as exploratory; its own hyperparameter validation is still nested.
No random or quasi-random target probes and no weight-noise populations are used.

## Negative candidate

The additional `basis` extractor injects each of the 16 positive hidden basis
states, follows ReLU recurrence, and summarizes age decoding with zero or one
step pointer lag. It costs a conservatively budgeted 35 FE for a pruned deployed
basis implementation, plus its 16-coordinate query cost. On unif, adding it to G
produced RMS .97387 pp at K1 and .68178 pp at K2, versus paired G baselines .96355
and .67802. This did not improve accuracy or compute ratio. Although plausible
as a clock diagnostic, a basis injection need not be a reachable network state.
The joint exploratory feature-generation routine computes both basis and pulse;
use `pulse_features` for the actual cheap pulse-only deployment.

## Outputs

`mechanics_results.json` contains paired G and G+pulse K1/K2/K4 results, the two
basis results, fold choices, full choices, runtimes, and external errors.
`mechanics_lowbudget.json` contains raw weights and weights+pulse with zero G
anchor rows. Prediction and feature npz files preserve all rows, in the archive's
original order. Root is responsible for final persistence and packaging.

## Completed measured results

All RMS and mean errors are percentage points. Costs use 1,024 kernel supports.

### Full G and G + pulse

| Recipe | K | Features | FE | CF RMS | CF mean | MSE/sampling |
|---|---:|---|---:|---:|---:|---:|
| unif | 1 | baseline | 618.5455 | 0.963552 | +0.000716 | 0.682366 |
| unif | 1 | pulse | 631.6635 | 0.841084 | +0.006065 | 0.530958 |
| unif | 2 | baseline | 1129.6635 | 0.678018 | +0.017069 | 0.617059 |
| unif | 2 | pulse | 1142.7816 | 0.654475 | +0.012066 | 0.581628 |
| unif | 4 | baseline | 2151.8997 | 0.583058 | +0.009817 | 0.869242 |
| unif | 4 | pulse | 2165.0177 | 0.576470 | +0.006306 | 0.854890 |
| zoo | 1 | baseline | 618.5455 | 0.596731 | +0.008966 | 0.318730 |
| zoo | 1 | pulse | 631.6635 | 0.578691 | +0.007951 | 0.306106 |
| zoo | 2 | baseline | 1129.6635 | 0.515027 | +0.008144 | 0.433614 |
| zoo | 2 | pulse | 1142.7816 | 0.489101 | +0.008846 | 0.395598 |
| zoo | 4 | baseline | 2151.8997 | 0.434513 | +0.005657 | 0.587925 |
| zoo | 4 | pulse | 2165.0177 | 0.405185 | +0.000428 | 0.514354 |

### Extra unif networks

| K | Method | B128 RMS | B128 mean | B64 RMS | B64 mean | Mystery error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 0.875220 | +0.143340 | 1.268588 | +0.804080 | -0.552960 |
| 1 | pulse | 0.808048 | +0.214165 | 1.031602 | +0.451264 | -0.416222 |
| 2 | baseline | 0.562636 | +0.092919 | 0.929005 | +0.490470 | -0.367031 |
| 2 | pulse | 0.634906 | +0.156050 | 0.802055 | +0.353462 | -0.463815 |
| 4 | baseline | 0.529026 | +0.079661 | 0.635267 | +0.253933 | -0.065831 |
| 4 | pulse | 0.529607 | +0.101887 | 0.595392 | +0.219293 | -0.207388 |

### Zoo seed errors for pulse

| K | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.594886 | +0.470499 | -0.326824 | -0.256544 | +0.183834 |
| 2 | -0.450290 | +0.960811 | -0.213568 | -0.319024 | -0.036233 |
| 4 | -0.264154 | +0.609625 | -0.434893 | -0.331216 | -0.207542 |

### Zero-anchor models (exploratory follow-on)

| Recipe | Method | FE | CF RMS | CF mean | MSE/sampling |
|---|---|---:|---:|---:|---:|
| unif | weights | 28.0622 | 1.604419 | -0.010547 | 0.085833 |
| unif | statspulse | 41.1802 | 1.354574 | -0.008362 | 0.089782 |
| zoo | weights | 28.0622 | 1.177860 | +0.012749 | 0.056338 |
| zoo | statspulse | 41.1802 | 1.075250 | +0.015349 | 0.068897 |

### Interpretation of evidence

The unif K1 augmentation is the clearest improvement in this experiment. The pulse addition reduces its paired MSE by 23.8%, at about 2.1% additional target compute. The unif K2 effect is smaller and mixed on the extra batch-128 set; K4 is too small to treat as established. Zoo improves at all three K values, with the clearest effect at K4 (.4345 to .4052 pp). The five released zoo seeds do not improve uniformly, as expected for a very small auxiliary sample.

A descriptive paired-error diagnostic for unif K1 gives a mean squared-error decrease .2210 pp^2 with SE .0546 pp^2 across networks. This does not include training-set or method-selection uncertainty. Absolute error improves on 55.8% of networks and median absolute error decreases .5236 to .5021 pp. Five networks account for 38.7% of the MSE gain, so some of the benefit is correction of large errors. For K4 the paired mean decrease .00764 pp^2 has SE .00872 pp^2 and median absolute error worsens slightly; avoid making a strong K4-unif claim.

Offline cost: the twelve paired baseline/pulse nested evaluations and refits took 356.0 measured seconds total at one BLAS thread; low-budget fits took 121.8 seconds. The first exploratory impulse/basis passes and cache/invariance checks were additional; no network was trained.
