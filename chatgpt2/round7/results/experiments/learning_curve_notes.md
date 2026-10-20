# Fixed-estimator learning curve

Estimator: all MF4 cells at K = 2, the ten signed-pulse features, and a 128-support readout. Every point has the same 591.448264 FE query budget. The feature vector has 48 coordinates.

## Protocol

Choose one label-independent population ordering with default_rng(202607). For each original outer fold, exclude the entire test fold, take the first requested number of eligible training networks in that ordering, and sort their IDs before fitting. Training sets are nested as the count grows. Standardization and family/gamma/alpha choice use only those training networks, with four inner folds and a fixed denominator of 1024 in the inner-fold ID hash. All 1024 networks still receive out-of-fold predictions at every curve point.

For the extra networks, take the first 256/512/768/1024 networks of the same population ordering and choose readout parameters in eight training-only folds. The cap is 128 supports throughout; the existing readout implementation selects supports within each fitted training set. Thus support identities may change with the training count. This is the learning curve of the specified fitting algorithm, not an oracle curve holding a latent feature map fixed.

## Cross-fitted population

| Outer training networks | Final-refit training networks | CF RMS (pp) | CF mean (pp) | MSE / sampling |
|---:|---:|---:|---:|---:|
| 224 | 256 | 0.901905 | -0.056828 | 0.571655 |
| 448 | 512 | 0.772067 | -0.004931 | 0.418912 |
| 672 | 768 | 0.651225 | -0.005547 | 0.298040 |
| 896 | 1024 | 0.650640 | +0.001324 | 0.297505 |

## Extra networks from final refits

| Training networks | B128 RMS (pp) | B128 mean (pp) | B64 RMS (pp) | B64 mean (pp) | Mystery error (pp) |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.710574 | +0.118705 | 1.170597 | +0.122117 | -0.308094 |
| 512 | 0.645803 | +0.048239 | 0.931441 | +0.387946 | -0.577153 |
| 768 | 0.645276 | +0.062957 | 0.942554 | +0.498753 | -0.662912 |
| 1024 | 0.612921 | +0.060648 | 0.925080 | +0.498356 | -0.389669 |

## Conclusion and limits

Additional population data helps substantially up to 672 outer-training networks in this subset sequence. The last increase, from 672 to 896, is essentially flat (.651225 to .650640 pp). The extra B128 group continues to improve somewhat at the final refit. B64 bias grows, underscoring that more training data from batch 128 does not necessarily cure recipe mismatch.

Do not extrapolate these four points to claim a particular result at 2,048 or 4,096 networks. One deterministic subset sequence is too little evidence for a stable power-law fit, and the final plateau may reflect sample composition, tuning, 128-support capacity, or feature limitations. An additional population should be judged on the fixed method with an independently held-out cohort; test larger support caps offline to distinguish readout compression from feature or data limits.

The largest training setting reproduces every prediction and hyperparameter choice of the source experiment exactly (maximum prediction difference 0). Runtime for the complete four-point curve and external refits was 6.767 seconds with one BLAS thread. No target input sampling or network training was performed.
