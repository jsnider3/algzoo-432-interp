# Follow-up: the interpretability route, i.e. neuron roles that predict a network's errors

Thank you for the careful population analysis. It settles that population-informed *sampling* designs are worth about 2× here. But that is not the route we want, and we should have said so more clearly. **The goal is interpretability.** We want to understand trained networks well enough that, for a new network, we can **look at its neurons, put each into a role bucket, and read off its accuracy, or its errors in each failure region, from that mechanistic description**, with little or no sampling. The population is for learning the catalog of roles and how each role, and its parameters, maps to errors. Please do not spend effort on control variates, importance sampling or stratification in this round.

## What we have done since (2²⁸-sequence snapshot of the same 1,024 networks, attached)

**`z2.py`** computes, per neuron, index-free features on 4,096 probe inputs:
- weights: input weight, self-loop, recurrent and readout norms;
- firing rates on and one step after events (new max, new 2nd max, neither);
- a comparator-threshold fit, z_t ≈ c_x·x_t + c_M·M_{t−1} + c_S·S_{t−1};
- final-state coding: η² on the ages of the max and of the 2nd max, and correlations with M and S;
- the readout column's DFT over positions.

It also computes, per network, the error counts in REPORT 7.15's 44 task strata on a shared pool of 2¹⁸ inputs. Output: `pop1k/snap_2e28.roles.npz`.

**`z3.py`** buckets neurons with rules taken from REPORT, checked on the zoo models (mystery n4 = max detector, n5 = one-step-late pointer, n9 = late max marker; seed 2 n10 and seed 3 n9 = on-event pointers). It also runs k-means, then predicts accuracy and per-region error from each network's role mix. Full output: `pop1k/z3_snap_2e28.txt`. Summary:

- **Population at 2²⁸.**
  - 53% have a write pointer: 38% on-event, 17% one step late, matching REPORT's 37–50% switch rate.
  - A network has 5.7 always-on neurons on average and 0–6 comparators (mean 3.1); 78% have a new-max detector.
  - Accuracy is 0.909 ± 0.036 across networks.
- **Predicting accuracy from the role mix** (5-fold CV RMSE):

  | description | CV RMSE |
  |---|---|
  | role counts | 1.9 points |
  | role counts + per-role mean features | 1.3 points |
  | each feature sorted over neurons | 1.1 points |
  | weights only (sorted input weights, self-loops, \|eig W_hh\|, readout norms) | 1.2 points |

  An RMSE of r is worth p(1−p)/r² samples, so 1.3 points ≈ 500 samples. Being useful needs about 0.3 points (≈ 10k samples), from features that cost far less than that to compute.
- **Error by region, and which roles go with it** (contribution to the error rate in points; correlation of that contribution with role counts):

  | region | mass | error rate | SD across networks | strongest role correlations |
  |---|---|---|---|---|
  | clean (both gaps ≥ 0.1·M) | 55% | 2.0% | 1.24 | comparators −0.65 |
  | middle | 26% | 8.0% | 1.28 | comparators −0.51 |
  | M–S near tie | 8.5% | 25% | 0.74 | max detectors −0.45, always-on +0.39 |
  | S–R near tie | 9.4% | 33% | 0.60 | max detectors −0.39, comparators −0.37 |
  | 2nd max negative | 1.1% | 71% | 0.13 | pointers +0.48, always-on −0.44 |

  So networks differ mostly in the clean and middle regions, and those errors track the **number of comparators**. Accuracy runs 0.87–0.89 with 0–2 comparators and 0.93–0.94 with 4–6. Near ties are hard for every network.

## What we would like from you

1. **A mechanistic account of why comparator count controls the clean-region error.** Ideally it is quantitative: each comparator tests x_t against a threshold a·M + b·S at certain ages of the max (REPORT 2.2). Does a network err on clean inputs exactly when no comparator covers the relevant (age, threshold) situation, or when coverage is too coarse? Can you derive a per-network "coverage" description from the comparators' parameters that predicts clean- and middle-region error?
2. **A better role taxonomy where it matters.** "Other" (5.6 per network) and "always-on" (5.7) are coarse. Are there sub-roles, such as clock pairs, value registers and memory channels with different readout patterns, whose parameters explain the remaining variance? Use any features you like, but keep them interpretable.
3. **Roles from weights alone, or from a few dozen designed inputs.** The probe features cost 4,096 forward passes, which is more than they are worth. Can each neuron's role and parameters be read from W_ih, W_hh and W_o? For example: the sign and size of its input weight, its recurrent threshold row, its eigenmode participation, its readout column. Alternatively, use a small set of deterministic structured probes. Designed inputs are allowed and cheap, because the networks are positively homogeneous.
4. **A composed mechanistic predictor.** Predict error in each region from the roles relevant to it, sum with the known region masses, and report cross-validated RMSE over the 1,024 networks. Also report how it does on the zoo models (`weights.npz`, seeds 0–4; accuracies 0.9531, 0.910, 0.880, 0.949, 0.932, 0.890). State the per-network cost, both in forward passes and in other compute.
5. Tell us honestly whether you think ±0.3 points is reachable this way, and what the remaining error is made of.

Please paste all code inline, since last time the sandbox files expired before they could be saved. A converged (2³⁰-sequence) version of this population arrives in about 8 hours, and we will rerun your code on it on our GPU.

## Files in the zip

- **Briefs and write-up:** `QUESTION3_FOR_CHATGPT_PRO.md` (this file), `REPORT.md` and `README.md`.
- **Analysis scripts:** `z2.py` (features), `z3.py` (roles, clusters, prediction, regions) and `z1.py` (`task_strata`).
- **Model code:** `batch_train.py` (torch forward) and `common.py` / `seeds_clocks.py` (numpy forward).
- **Population data:**
  - `pop1k/snap_2e28.npz`: weights of the 1,024 networks at 2²⁸, keys `s262144_{Wih,Whh,Wo}`.
  - `pop1k/snap_2e28.roles.npz`: z2 output, with `feats` (1024,16,24), `names`, `strata_err` (1024,44), `strata_n` (44,) and `acc`.
  - `pop1k/snap_2e28.buckets.npz`: z3 roles and clusters.
  - `pop1k/z3_snap_2e28.txt`: z3's printed output.
- **Zoo models:** `weights.npz`, `weights_seed0..4.npz`.
