# ChatGPT Pro, round 2: interpretability (answer to QUESTION3_FOR_CHATGPT_PRO.md)

This is a summary of the answer in https://chatgpt.com/c/6ad44fa0-e8bc-83ea-91f2-b4c5ab5c2840, posted 2026-10-18 after 59 minutes of work. The text was read from the page; the verbatim answer could not be scraped. ChatGPT saved five files to the ChatGPT library, and they still need downloading:
- `roles_mechanism.py`: designed probes, neuron descriptions, comparator interventions, regional calibration, cross-validation and new-network prediction;
- `roles_results.json`;
- `calibrated_hybrid.npz`;
- `README_roles.txt`;
- `algzoo_roles_results.zip`.

The script is also pasted inline at the end of the answer.

## Headline

- **Best predictor.** The best accuracy predictor scores **1.05–1.06 points** CV RMSE: a GBM on weights + roles + gated features. The best ridge, on weights + regional mechanism features, scores **1.16–1.18**. Weights alone score 1.17–1.24.
  - That is a gain of about 10% over weights only.
  - The ±0.3 target needs R² = 0.993; the best model reaches 0.913.
  - Only 28% of predictions fall within ±0.3 points.
- **Cost.**
  - The hybrid features take 3.2 ms per network. They use 60 forwards plus reverse and tangent passes, about 360 forward-equivalents.
  - 4,096 plain forwards take 2.1 ms.
  - So the predictor is not a shortcut; its value is diagnostic.
- **Zoo transfer.** The richer features transfer worse:

  | predictor | zoo RMSE |
  |---|---|
  | weights-only ridge | 0.83 |
  | hybrid ridge | 1.54 |
  | hybrid GBM | 1.33 |

  All six hybrid-GBM predictions underestimate accuracy.

## Mechanistic findings

1. **Comparators are analog.**
   - A downstream gate sees Σ_j A_pj·|w_j|·(θ_j − x)_+. That is a piecewise-linear spline, and its root can sit between the comparator hinges.
   - Example: ReLU(M−S−ReLU(M−x)) opens at x > S using a comparator with threshold M.
   - So a comparator near S is not needed. Counting nearby thresholds throws away amplitudes and routing.
2. **The literal coverage rule fails.**
   - The test bank has 48 controlled prefixes: M=1, S ∈ {.45, .8}, 8 ages of M and 3 positions of S, each branched into the three event types.
   - Rule tested: a case is "covered" if min_j |θ_j(h)−S|/(M−S) ≤ .25.
   - 60% of new-2nd write mistakes and 45% of handoff mistakes happen *with* coverage.
3. **Comparator thresholds depend on the age of S as well as the age of M.**
   - The median threshold span over S positions is 0.29(M−S).
   - Fitting θ = a_age(M)·M + b_age(M)·S leaves an RMS residual of 0.22(M−S).
4. **State swaps give a causal double dissociation for timing.**
   - Setup: a clean new-2nd event (x=.725) is paired with a "neither" event (x=.2) after matched histories (M=1, S=.45). One role group's hidden state is swapped, and the table gives the share of correct outputs.

     | state swapped | late-ptr networks (n=169) | event-ptr networks (n=385) |
     |---|---|---|
     | none | 99.98% | 99.95% |
     | comparators at the event | 8.3% | 88.9% |
     | pointers at the event | 96.0% | 5.6% |
     | comparators one step later | 91.6% | 95.2% |
     | pointers one step later | 2.0% | 98.5% |

   - Late-pointer networks hold the event in comparators at t and hand it to the pointer at t+1. Event-pointer networks hold it in the pointer at t. This confirms REPORT's two strategies causally.
   - Swapping comparator or max-marker states in the new-2nd vs new-max contrast drops accuracy to about 50%. They choose between writing the current event and promoting the old max.
5. **Invariance critiques of z2/z3.**
   - **Rescaling.** Under w′=Dw, A′=DAD⁻¹, W_o′=W_o D⁻¹ (positive diagonal D) the function is unchanged, but D=.1I sends the comparator count from 3.1 to 0. Proposed fix: normalize by centered readout norms, c_j = ‖C W_o e_j‖.
   - **Readout phase.** Cyclically rotating W_o's rows leaves all 24 z2 features unchanged but sends zoo accuracy to 0.4–0.9%. z2 keeps only DFT magnitudes, so it loses the phase.
6. **Taxonomy.** Roles overlap and are continuous, not exclusive bins.

   | measure | always-on | other |
   |---|---|---|
   | mean corr² with M | .25 | .14 |
   | η² on max-age | .30 | .38 |
   | value R² > .25 | 38% | 10% |
   | age η² > .25 | 65% | 86% |

   - Readout-aligned pulse paths W_o A^a e_j identify the write channel in mystery (n5, delay 1, 9/9 ages) and in seed 2 (n10, delay 0, 10/10). The other models are imperfectly aligned.
7. **Data correction.** z1.task_strata yields **60** strata (4 for S<0, 7×8 for S≥0, one of them empty), not 44. The z3 region masks were already right; the "44" in the docstrings and QUESTION3 was wrong.

## The 60-probe design and the composed predictor

- **Probes.** Six value regimes (clean sparse, clean crowded, middle, M–S tie, S–R tie, S<0), each with 10 rotations.
- **Features.** 216 role summaries (8 counts + 26 continuous parameters per role) and 222 gated decision summaries:
  - margins for 2nd vs max, 3rd and the best lower logit;
  - exact local gradients, using the Euler identity Δ = gᵀx;
  - nuisance shares;
  - distance to the first gate boundary.
- **Gate cells.** The whole rank gap stays in one activation cell on only 12% of M–S paths and 33% of S–R paths, so local derivatives rarely describe the whole path to a tie.
- **Regional residuals** (GBM hybrid, contribution RMSE in points):

  | region | RMSE |
  |---|---|
  | S<0 | .05 |
  | M–S tie | .22 |
  | S–R tie | .22 |
  | middle | .42 |
  | clean | .44 |

  - Cross-region covariances are 0.63 of the 1.11 pp² total MSE.
  - Clean + middle account for about 70%.
  - Even a perfect clean + middle prediction leaves 0.40 RMSE.
- **Coverage-feature diagnostics** (clean / middle contribution RMSE):

  | features | clean | middle |
  |---|---|---|
  | sign-based coverage | 1.01 | 1.07 |
  | + routed gates | 0.94 | 0.96 |
  | + designed endpoints | 0.73 | 0.68 |
  | + cached role counts | 0.60 | 0.63 |

## Verdict

The answer does not see a path to ±0.3 points by refining bags of neuron roles. It suggests a circuit description that keeps:
- both ages;
- analog gate routing;
- handoff behaviour;
- phase-aligned memory;
- rejection of lower-input interference.

It gives no impossibility result. For the 2³⁰ checkpoint it recommends freezing the features and calibrator first, to test temporal transfer, and then repeating the within-checkpoint CV.

## Downloaded files and reproduction (2026-10-18)

The user downloaded three of the five files. Copies are in `chatgpt2/roles_round2/`:
- `roles_mechanism.py`, 844 lines. Its SHA-256 (58a31bf5…) matches the hash recorded in the JSON.
- `roles_results.json`, the consolidated results.
- `raw.txt`, the README.

`calibrated_hybrid.npz` and `algzoo_roles_results.zip` were not downloaded. The rerun regenerates the calibrators.

I reviewed the code before running it. It uses only NumPy and scikit-learn, makes no network or subprocess calls, and writes only to `--out`.

Rerun command:
`python -P chatgpt2/roles_round2/roles_mechanism.py --root . --population pop1k/snap_2e28.npz --labels pop1k/snap_2e28.roles.npz --buckets pop1k/snap_2e28.buckets.npz --out chatgpt2/roles_run_2e28 --gbm --mechanisms --threads 2`

It took 3 min 42 s on CPU with a peak of 360 MB, and it reproduces every headline number:

| model | CV RMSE, splits 0 / 1729 | zoo RMSE |
|---|---|---|
| weights additive | 1.231 / 1.242 | 0.83 |
| roles60 ridge | 1.704 / 1.735 | 2.40 |
| invariant ridge | 1.364 / 1.349 | 1.74 |
| hybrid ridge | 1.180 / 1.156 | 1.54 |
| weights GBM | 1.170 / 1.203 | 0.92 |
| hybrid GBM | 1.052 / 1.064 | 1.31 |

The hybrid GBM result differs from ChatGPT's 1.053 / 1.061 only by GBM noise. The state swaps reproduce exactly (8.3 / 88.9 / 96.0 / 5.6 / 91.6 / 95.2 / 2.0 / 98.5%). The phase counterexample gives 0.39–0.85% accuracy, and the threshold-history residual is 0.22·(M−S).

All six hybrid-GBM zoo predictions come out low. One hypothesis: the zoo models are trained longer than the 2²⁸ snapshot, so the richer features carry a training-stage shift. The converged 2³⁰ population will test this.

The JSON also answers the brief's question 3 (roles from weights alone), which the chat text skipped. A tree classifier on weight features recovers the z3 role labels with 73% accuracy (F1 per role):
- comparator 0.93
- max detector 0.94
- event pointer 0.82
- late pointer 0.61
- always-on 0.67 and "other" 0.65 (mostly confused with each other)
- late marker 0.

## Our check (2026-10-18)

The rescaling critique is right in principle but not in practice here.
- Only 0.07 negative-input comparators per network fail the |w|>2 test alone.
- Requiring only w_in < 0, which is invariant to positive D, gives the same count-to-accuracy curve: corr +0.56 against +0.54.
- A scale-free version using thr_cx < −.3 is weaker (corr +0.34).
