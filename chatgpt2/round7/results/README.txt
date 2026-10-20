ALGZOO ROUND 7: EXPERIMENTS AND RUNNABLE ESTIMATOR

Start with ROUND7_REPORT.pdf. The main program is round7_estimator.py.
Dependencies: Python 3.10+, numpy, scipy, threadpoolctl.
All required input weights and K=1,2,4 cell caches are in the user's original
algzoo-432-interp-r7.zip. Point --root at its extracted algzoo-432-interp folder.

Portable commands (replace PROJECT with that folder):
  python round7_estimator.py --root PROJECT --out mf4_k2
  python round7_estimator.py --root PROJECT --K 4 --out mf4_k4
  python round7_estimator.py --root PROJECT --K 1 --kind both --subset ends --groups none --supports 128 --refit outer --out compact
  python round7_estimator.py --root PROJECT --K 2 --groups fixed --supports 128 --refit outer --out mf4_k2_small
  python round7_estimator.py --root PROJECT --recipe zoo --K 2 --groups none --supports 128 --refit outer --out zoo_k2
  python round7_estimator.py --root PROJECT --K 16 --groups fixed --supports 128 --refit outer --generate --out mf4_k16
  python round7_estimator.py --predict models/mf4_k2.model.npz --weights PROJECT/weights.npz

The supplied models/mf4_k2.model.npz is fitted on the 1,024 primary batch-128
networks. It supports prediction directly from a target weight file, without
training caches, labels, or any online fitting. Query cost is 650.617708 FE.
The main program is self-contained: it does not import the research scripts.

experiments/ retains original research scripts, JSON metrics, per-network
NPZ predictions/features, printed outputs, audit notes and validation checks.
Some harnesses expect to run from PROJECT; consult their docstrings/CLI.
For easy inspection, ROUND7_ALL_CODE.txt concatenates all authored Python
source files. The full main estimator is also pasted inline in the chat reply.
The bundle omits regenerated K=8/16 raw cell caches and rendered PDF QA pages;
--generate recreates larger fixed designs deterministically.

Validation protocol:
  Eight outer folds, fixed permutation seed 0, on the first 1,024 networks.
  Four inner folds for family, regularization, kernel width and (where used)
  temporal block weight. All standardization and centering are training-only.
  Stable inner fold IDs use modulus 1,024, fixing metadata-sensitive splits.
  --refit index reproduces original final-refit CV; --refit outer reproduces
  the cost/compact experiment harness. Both use only training-population data.
  Feature-family and structural-design comparisons are exploratory research,
  not a new untouched holdout. External sets have been examined repeatedly.

Cost convention:
  One FE is 2,880 multiply-adds under the task's operation ledger.
  Costs include prefix evaluation, features, weight statistics, exponentials
  and one compiled selected readout. Input products are cached by scalar.
  Pulse counts refer to two trajectories with 20 age-output evaluations;
  their complete extraction is conservatively charged at six FE.
  A fixed 1,024-support budget is used for full cross-fit comparisons, although
  outer training sets have 896 networks and a selected linear model is cheaper.
  Historical tables retain conservative exploration costs. The final extractor
  shares normalization: full MF4+P+H4 K2 saves 0.025 FE; fixed-H4 K2/128 saves
  3 FE versus the published exploratory ledger. Prediction results are unchanged.

No new RNNs, no weight-noise copies, and no random/quasi-random target inputs
were used. Population-level random CV/landmark selection is offline and allowed.
No GPU was needed. JSON files record individual offline runtimes.

Primary evidence: supplied question, REPORT.md, gfeat.py, source scripts/logs,
and original weights/caches. Every new numerical claim is traceable through
headline_summary.json and the source-file paths it records.
