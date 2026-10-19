# Circuit model, first pass, on the 2²⁸ snapshot of the 1,024-network population (2026-10-18)

This is working notes; the numbers will be redone on the converged (2³⁰) population. Accuracy here is 0.908 ± 3.6 points across networks. An estimate with RMSE r is worth p(1−p)/r² samples, and the target is ±0.3 points, about 9,200 samples.

## 1. Where networks differ (`z4.py`, `pop1k/z4_snap_2e28.txt`)

The variance share of a cell is cov(cell's error contribution, total error) / var(total error). The shares sum to 1.

| region | mass | error | share of across-network variance |
|---|---|---|---|
| S < 0 | 1.1% | 69% | 1% |
| M–S tie (a < .05) | 8.6% | 25% | 18% |
| S–R tie (b < .03) | 9.4% | 32% | 15% |
| middle | 25.9% | 8.0% | 35% |
| clean | 55.0% | 2.1% | 31% |

Inside the clean and middle regions, the variance comes from **handoffs**: cases where the 2nd max is the old max, displaced when a newer max arrives.
- In the clean region, handoffs err 3.5% of the time, against 0.7% for direct writes. Handoffs carry 26% of the variance in clean and 22% in middle.
- The worst case is a **handoff at the last step**, when the max arrives at t = 9. It errs 11% (clean) and 20% (middle), about 25% of all variance from 8% of the mass.
- Handoff error falls with the number of steps k between the new max and the end (`z6.py`). Error rates are over non-tie inputs; correlations are with the network's accuracy.

  | k | handoff error | spread across networks (SD) | corr. with accuracy | direct-write error |
  |---|---|---|---|---|
  | 0 | 14.1% | 13.8 | −0.81 | 2.6% |
  | 1 | 4.9% | — | −0.85 | 2.6% |
  | 2 | 3.4% | — | — | 2.5% |
  | 3 | 2.8% | — | — | 2.3% |

- **The handoff failure** is mostly that the answer stays at the old 2nd max R (78% of errors). Errors rise with R/S: 5% when R < 0, 21% when R/S > 0.8. The order in which S and R arrived makes no difference (14.6% against 13.6%).
- **The comparator count controls handoffs.** Last-step handoff error by number of comparators:

  | comparators | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|---|
  | handoff error | 40.5% | 36.8% | 23.8% | 11.7% | 6.4% | 4.3% | 4.3% |

  This is why comparator count predicts accuracy.
  - Late-pointer networks hand off better: 6.9%, against 16.3% for event-pointer networks and 14.9% with no pointer.
  - pop994 fails 98% of last-step handoffs.
- **It holds in the fully trained zoo models.** Last-step handoff error:

  | model | error |
  |---|---|
  | mystery | 0.7% |
  | seed 2 | 2.4% |
  | seed 3 | 3.3% |
  | seed 0 | 8.1% |
  | seed 4 | 13% |
  | seed 1 | 23% |

  This lines up with REPORT 5's memory vs. selection split.
- **The 8-number latency profile** (handoff and direct-write error at k = 0..3) predicts accuracy to 0.79–0.89 points (CV). Each rate needs sampling to measure.

## 2. Mean field (`z5.py`, `pop1k/z5_snap_2e28.txt`)

Mean field replaces the lower 7 inputs by E[x | x < R].
- **Fit to true accuracy:**
  - Mean-field accuracy correlates 0.988 with true accuracy.
  - The difference (the leakage) averages +0.42 points with an SD of 0.585 across networks.
  - After one affine calibration the RMSE is 0.55 points.
  - Network features predict the leakage only down to 0.43 points.
- **Leakage is spread over all the lower inputs.** Keeping one input exact barely helps:

  | input kept exact | SD (points) |
  |---|---|
  | x₉ | 0.55 |
  | 4th largest | 0.40 |
  | inputs after the last of the top 3 | 0.50 |

- **Mean field caps at about 0.4–0.55 points, so it cannot reach ±0.3 on its own.**

## 3. Evaluating a reduced function at many points does not beat sampling (`z5.py`; scratch `mf5`, `mf6`)

Per-network RMSE in points at K forward passes, raw and after calibration:

| K | plain sampling | ordered Sobol | mean field on a Sobol top-3 design |
|---|---|---|---|
| 256 | 1.68 / 1.55 | 1.57 / 1.37 | 1.54 / 1.46 |
| 1,024 | 0.78 / 0.69 | 0.76 / 0.74 | 0.98 / 0.95 |
| 4,096 | 0.42 / 0.38 | 0.35 / 0.33 | 0.86 / 0.67 |

- **Learned weights over a fixed probe set** (ridge across networks) do no better than the plain mean of the outcomes.
- **A soft version**, Φ(margin / (κ · blur from the lower-input gradient)) at mean-field points with κ fitted on the population, scores 0.87 at K = 1,024, still worse than sampling.
- **So a circuit model has to integrate analytically over the values** and evaluate the network far less often than sampling would.

## 4. Attempts at closed form or a cheap circuit description

- **Switched-linear model.**
  - The gates are not constant between events: 72% of idle-to-idle steps flip at least one neuron, mostly "other", pointer and marker neurons.
  - Given the event chain, the most common gate pattern covers only 63–72% of steps.
  - So the network is not a linear system switched only by events.
- **Last-step handoff.**
  - Attribution at the last step is exact once the gates are fixed. The margin is mostly the new max's own write (+4.9) plus always-on activity (+3.0); comparators contribute −1.2 through the recurrence.
  - No single part tracks the error rate. The total margin does (corr −0.81).
  - A single R/S threshold explains none of the errors within any (network, age) cell.
  - A GBM on the top-3 values and ages explains 30–44% of the deviance, and 43–54% with the whole prefix.
  - So these errors are blur from many small effects (REPORT 4).
- **Ways to estimate the last-step handoff rate.**
  - Network features: R² 0.85.
  - 128 sampled prefixes with x₉ integrated over 32 conditional quantiles: R² 0.98. Mean-field prefixes are no better (0.96).
- **Ties.**
  - The method: bisect the switch gap on 48 mean-field arrangements, then measure the lower-input blur at the switch point.
  - Under mean field the switch sits almost exactly at the tie: the median offset is 0.001–0.002·M and the mean 0.013–0.018·M.
  - Tie errors are blur from the lower inputs.
  - Twelve numbers from about 2,000 passes predict the M–S tie contribution with R² 0.91 and S–R with 0.83, about the same as the z3 role features (0.89 and 0.84).
- **Combined accuracy predictors** (CV RMSE, points):

  | features | RMSE |
  |---|---|
  | weights | 1.23 |
  | + roles | 1.18 |
  | + tie probes | 0.99 |
  | composed per mechanism component (weights + roles) | 1.085 |
  | ChatGPT's best | 1.05 |

## Verdict

The circuit picture is now clear.
- Networks differ mostly in **how fast and how reliably they hand the old max over to the 2nd-max channel**, which comparators make possible, and in **tie blur**.
- Errors inside each mechanism are blur from many small contributions, not sharp thresholds.

Nothing tried here gets below about 1 point cheaply. Below that, every route evaluates the network often enough that sampling with the same budget does as well or better. The best estimator in the repo remains the exact one: last-input integration plus ordered Sobol (REPORT 7.15), at about 2.5–4× sampling.
