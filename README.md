# algzoo-432-interp

Mechanistic interpretability of ARC's AlgZoo 432-parameter 2nd-argmax RNN (`alg_zoo.example_2nd_argmax()`): a 16-neuron ReLU RNN with no biases that outputs the position of the second-largest of 10 N(0,1) inputs. The write-up is **[REPORT.md](REPORT.md)**. It covers the mechanism (age code, clocks, comparators, write pointer), its failure modes, the other zoo seeds and sizes, the training-time fork between "memory" and "online selection", and attempts at ARC's mechanistic accuracy-estimation challenge. [QUESTION_FOR_CHATGPT_PRO.md](QUESTION_FOR_CHATGPT_PRO.md) is the brief that started REPORT 7.15. It is kept as sent, so a few of its numbers predate later corrections; REPORT.md is authoritative. `QUESTION2_FOR_CHATGPT_PRO.md` and `QUESTION3_FOR_CHATGPT_PRO.md` are the briefs for the population rounds (REPORT 7.16), with the answers summarized in `chatgpt2/`. `QUESTION4_FOR_CHATGPT_PRO.md` asks for a deterministic integral of the leakage model over the top-4 configurations. The answer, a negative result, is summarized in `chatgpt2/chatgpt_answer_3.md`, with the follow-up in REPORT 7.16 (`z13`–`z39`). `QUESTION5_FOR_CHATGPT_PRO.md` maps what fails and asks for a decomposition with a cheap correction, or an argument that none exists. The answer is summarized in `chatgpt2/chatgpt_answer_4.md`, with its full text, code, report and results in `chatgpt2/round5/`. Its §9 wrongly excludes fitting on the population; `z45`–`z47` do that. `QUESTION6_FOR_CHATGPT_PRO.md` is the round-6 brief. The answer's text and code are in `chatgpt2/round6/`, checked in REPORT 7.17 (`z48`–`z53`). `QUESTION7_FOR_CHATGPT_PRO.md` is the round-7 brief. The answer's text, its standalone estimator (`round7_estimator.py`), PDF report and result archive (without its per-network `.npz` outputs) are in `chatgpt2/round7/`, checked in REPORT 7.18 (`z59`–`z61`). `refs/` holds copies of the sources behind 7.17:
- ARC's AlgZoo post, with Zihao Chen's write-up and Colab notebook (the mystery model's training recipe) and the Hilton and Robinson notes it links;
- ARC's "Competing with sampling" post;
- the LessWrong version of the AlgZoo post and its comments.

## Setup

- Python 3 with **numpy** and **scipy**. That is enough for most analysis scripts.
- Additional packages for some scripts:
  - **torch**: training scripts and a few analyses (`a8`, `c1`, `j1`, `j2`, `seeds_probe`).
  - **scikit-learn**: `b4`, `n3`, `x4`.
  - **numba**: `w5`, `v1`, `lines`, `z11` (exact mode).
  - **CUDA torch** (one 8 GB GPU is enough): the population scripts `batch_train`, `batch_train_unif`, `z1`–`z8`, `z10`–`z43`, `z46`–`z48`, `z50`–`z59`, `z61`–`z80`, `v1`, `lw2` (`z41` also runs `lines.trace_line` on the CPU; `z44`, `z45`, `z49`, `z60` are CPU-only, and `z48`, `z50`–`z59`, `z61`–`z80` fall back to the CPU). `z9` is CPU-only (multiprocessing).
- **[AlgZoo](https://github.com/alignment-research-center/alg-zoo)** (`alg_zoo`) is needed only for the `load*.py` exporters and for scripts that build or train a `DistRNN` (all `k`, `l`, `m`, `p` and `q` training scripts, plus `n3`, `u1` and `w3`). Its weights are already exported here, so analyses do not need it.

Run every script **from the repository root**, for example `python a7.py`. Scripts import each other by module name (`common`, `seeds_clocks`, `k1`, `j5`, …) and read data files by relative path. Each script prints its own results; nothing is cached between scripts except the files listed below.

## Data files

| file | contents | written by |
|---|---|---|
| `weights.npz` | the 432-parameter "mystery" model | `load.py` |
| `weights_seed{0..4}.npz` | zoo 16-neuron, length-10 models, seeds 0–4 | `load_seeds.py` |
| `weights_h{4,6,8,32}.npz` | zoo models with other hidden sizes | `load_sizes.py` |
| `ckpt/s{seed}_e{k}.npz` | zoo training checkpoints after 2^k sequences (k = 0 is the initialization; k = 20–30) | `load_ckpts.py` |
| `pop/p{0..15}.npz` | 16 fresh 16-neuron seeds trained to 2^27 sequences | `w3.py` |
| `fork_s0.npz` | a switched fork of seed 0 from 2^22 (REPORT 7.14 barrier) | `k8.py` |
| `ckpt_metrics.json` | per-checkpoint metrics for REPORT 6 | `d1.py` |
| `p1_cols.npy` | minimal readout-column transplant set | `p1.py` |
| `thr.npy` | per-age decision thresholds | `a22.py` |
| `pop1k/seed0.npz` | 1,024 networks trained to 2^30 sequences, with snapshots: keys `s<step>_{Wih,Whh,Wo}` (step 1,048,576 = 2^30 sequences), `init_*`, and `acc_2e20` | `batch_train.py` |
| `pop1k/snap_2e28.npz` | the same networks at 2^28 sequences | `batch_train.py` |
| `pop1k/*.roles.npz`, `*.buckets.npz` | per-neuron features and role buckets | `z2.py`, `z3.py` |
| `pop1k/z*_seed0.txt`, `pop1k/v1_*.txt` | printed outputs of the population analyses | `z2`–`z11`, `v1` |
| `pop1k/notes_circuit_2e28.md`, `pop1k/notes_circuit_2e30.md` | working notes behind REPORT 7.16 | |
| `pop1k/seed0_z12.npz` | per-network outputs of the integrand check | `z12.py` |
| `chatgpt2/` | ChatGPT Pro's population answers (`chatgpt_answer_1`–`3.md`), its scripts, and their reruns | |
| `pop1k/seed0_z52_b<n>_k<K>.npz` | per-network summary features of the global readout G(n³, K) | `z52.py` |
| `pop_unif/b<batch>_seed<s>.npz`, `pop_unif/b128_n1024_seed4.npz`, `pop_unif/b128_n512_seed6.npz`, `pop_unif/b128_n1024_seed7.npz`, `pop_unif/b128_n2048_seed{8,9}.npz` | networks trained with the mystery model's recipe: 64 per batch size (64, 128, 256, 512, 1,024), 1,024 at batch 128, 512 more at batch 128 (`b128_n512_seed6`, trained after round 7 for the test in REPORT 7.18), 1,024 more (`b128_n1024_seed7`, the blind test of REPORT 7.21, then extra training networks), and 2 × 2,048 more (`b128_n2048_seed8`, training; `b128_n2048_seed9`, the second blind test); keys `init_*`, `s<epoch>_{Wih,Whh,Wo}` (epochs 10, 30, 100, 300, 1,000), `acc_gauss`, `acc_unif` (2²⁰ inputs) and `batch` | `batch_train_unif.py` |
| `pop_unif/*.acc22.npy` | accuracy on 2²² N(0,1) inputs per network, including the mystery model's weight-noise copies | `z55.py`, `z58.py`, `z61.py` |
| `pop_unif/*.acc22s61.npy`, `*.acc22s72.npy`, `*.acc22s79.npy`, `*.acc22s80.npy` | accuracy on independent banks of 2²² N(0,1) inputs (seeds 61, 72, 79, 80) | `z61.py`, `z72.py`, `z79.py`, `z80.py` |
| `pop_unif/cells_k{1,2,4}.npz` | per-cell MF4/MF3 bits and margins, G's summaries and weight statistics of 1,153 mystery-recipe networks, for fitting without a GPU (not in the repository: `z59` rebuilds them in seconds on a GPU) | `z59.py` |
| `chatgpt2/round7/runs/` | round 7's estimator rerun here: compiled models (`*.model.npz`, read by `z61`) and metrics (`*.json`) | `chatgpt2/round7/round7_estimator.py` |
| `pop_unif/z5*.txt`, `pop_unif/z6*.txt` | printed outputs of `z54`–`z58`, `z60`–`z80` | |
| `pop_unif/roles_b128_n1024_n512.npz`, `pop_unif/roles_b128_n1024_seed7.npz`, `pop_unif/roles_b128_n2048_seed8.npz` | neuron roles (`z3.roles`) of the mystery-recipe networks (the first also has pointer selectivity) | `z63.py`, `z75.py`, `z79.py` |
| `pop_unif/frozen_720.pkl`, `pop_unif/frozen_721.pkl` | estimators frozen before their blind tests: 7.19–7.20's on the 1,024 training networks (before seed 7), and the learned rows on 4,096 (before seed 9) | `z72.py --fit`, `z80.py --fit` |
| `refs/` | sources for REPORT 7.17 (see the top of this file) | |

Weight keys: `rnn.weight_ih_l0` (H,1), `rnn.weight_hh_l0` (H,H), `linear.weight` (10,H). `common.py` (mystery model only, with Jacobians) and `seeds_clocks.py` (`load`, `run` for any weights file) are NumPy re-implementations of the forward pass.

**Training logs are not in the repository.** Training scripts print one line per run to stdout. The report's run counts in 7.14 were tallied from those outputs, saved as `*.log` files, which `.gitignore` excludes. The scripts that re-read them (`k11`, `k13`, `l8`, `m6`, `m7`, `n3`, `q4`) need the training scripts re-run first, with output redirected to the file names they open. Several of the quadrature estimators (`n1`, `n4`, `r5`, `s*`, `t2`) take CPU-hours. Large regenerable outputs are not committed either: `pop1k/seed0_z13_q*.npz` (16 MB each; `python z13.py 8192 256 <q>`, about 15 s), `pop1k/seed0_z15_{mean,median}.npz` (about 430 MB each, read by `z16` and `z19`; `python z15.py 8192 64 mean`, about 1 min), `pop1k/seed0_z16_mean.npz` and `pop1k/seed0_z18_*.npz`, the `designed_v3_population.npz` and `mechanisms.npz` arrays of `chatgpt2/roles_run_*` (rerun `chatgpt2/roles_round2/roles_mechanism.py`), the training resume checkpoints `pop1k/seed0.npz.resume.pt` and `pop_unif/*.resume.pt`, and the zip bundles sent to ChatGPT.

## Script index

Scripts are named by round of experiments (one letter per round, numbered within it). This table maps them to REPORT.md sections.

| report section | scripts |
|---|---|
| 2.1 state stores ages | `a6`, `a7` (shifted readout), `a8` (held-out age probes), `a9`, `a16` (state surrogates), `a3`, `a5`, `a18`, `feats` (feature regressions) |
| 2.2 comparators | `a10`–`a12`, `a22` |
| 2.3 clocks, phase handoff | `a19`–`a21`, `seeds_clocks` |
| 2.4 causal tests | `e1` (ageing/rotating clocks), `a21` (mode removal), `y2` (zero- vs mean-ablation) |
| 2.5 Fourier readout | `a23` |
| 2.6 length generalization | `e5`, `e6` |
| 3 failure modes | `a13`, `a14`, `a17`, `e8`–`e10`, `b5`, `a2`, `e7`, `y3` (region counts) |
| 4 short programs, pruning | `b1`–`b7` |
| 5 other seeds | `c1`–`c6`, `seeds_clocks`, `seeds_probe`, `e2`–`e4` |
| 6 training dynamics | `d1`, `d2` |
| 7.1 patching | `f1`, `f2` |
| 7.2 2nd-max storage | `f3`, `f4` |
| 7.3 max suppression | `f5`–`f7`, `a15` |
| 7.4 seed 2's switch | `g2` |
| 7.5 stratified estimate | `g1` |
| 7.6 seed 2 decoded | `h1`–`h4` |
| 7.7 comparator-resolution estimate | `h5`, `h6` |
| 7.8 other sizes | `h7`–`h9` |
| 7.9 write pointer | `i1`–`i4`, `i9`–`i14`, `n2`, `y1` (64-candidate pulse scan) |
| 7.10 32-neuron time cells | `i5` |
| 7.11 mechanism-informed estimators | `i6`–`i8` |
| 7.12 readout optimum, n10's switch | `j1`–`j4` |
| 7.13 sampling-free estimation | `j5`–`j7`, `k9`, `k10`, `l1`, `l3`, `n1`, `n4`, `n5`, `o1`–`o4`, `r1`–`r5`, `s1`–`s8`, `t1`, `t2`, `cache`, `y6` (sampled 3- vs 4-input mean field) |
| 7.14 training replays | `k1`–`k8`, `k11`–`k14`, `l2`, `l4`–`l10`, `m1`–`m7`, `n3`, `p1`–`p4`, `q1`–`q6`, `u1`, `y4` (straight-line barrier control), `y5` (switch-criterion control), `y7` (initial-gradient statistics) |
| 7.15 second estimation round | `w1`–`w9`, `x1`–`x4` |
| 7.16 population and leakage model | `batch_train` (population trainer), `z1` (population strata), `z2`, `z3` (neuron roles), `z4`–`z7` (error anatomy, mean field, handoffs, informative inputs), `z8` (lower-input closures, first-order model), `v1` (exact 2nd-max line), `z9` (exact mean-field integral), `z10`, `lw2` (leakage anatomy), `z11`, `lines` (leakage model), `z12` (structure of the top-4 integrand), `z13`, `z14` (local fidelity of lower-input models; margin-additive model), `z15`–`z19` (the margin model's bias: draw-level check, gate double counting, ANOVA, nearby-pair projection, roles of the double-counted neurons), `z20` (the indicator model's total, resolved), `z21`, `z23` (deterministic outer quadrature of the MF4 anchor integral, CDF and homogeneous coordinates), `z22` (is the lower-input correction local to the boundary?), `z24` (can the arrangement enumeration be cut?), `z25`, `z26` (a closed form from linear regions; how many regions an arrangement holds, and the oracle bound), `z27` (homogeneous coordinates with the geometric kinks removed), `z28` (sparse grids), `z29` (the clock-state simplification: clock and memory channels made linear), `z30` (the linear-region closed form with oracle-chosen patterns, measured), `z31` (an adaptive outer rule), `z32` (third-order lower-input terms), `z33` (deterministic pattern discovery along S-lines), `z34` (the ideal additive lower-input model), `z35` (closed form with a least-violation fallback), `z36` (joint product rule over the lower inputs), `z37` (top-2 and top-3 decompositions), `z38` (an end-to-end deterministic estimate with the correction), `z39` (decomposition by events), `z40` (the paired scoring table `pop1k/reference.npz`), `z41`–`z44` (checks of ChatGPT round 5: its two-level estimator, margin gating and screening, arrangement dependence, gate histories), `z45` (calibration against the population), `z46`, `z47` (a learned cell correction) |
| 7.17 global readout, mystery recipe, ARC's techniques | `gfeat` (shared G(B, K) pieces), `z48` (prefix-sharing cost), `z49` (the readout and its ablations), `z50` (population vs the zoo's training logs; needs the zoo's jsonl logs from ARC's public bucket), `z51` (logit scale by strategy), `z52` (larger designs G(B, K)), `z53` (exact last-input integral; needs the `last_q*.npz` written by `chatgpt2/round6/q6_methods.py last`), `batch_train_unif` (the mystery model's recipe), `z54` (mystery vs each recipe group), `z55` (readouts by recipe, weight-noise protocol), `z56` (gauge-fixed weights, subcircuit template, covariance features), `z57` (Hermite-projection oracle), `z58` (batch-64 check) |
| 7.18 cheaper probes, pulses, test on new networks | `z59` (CPU cell caches), `z60` (independent rebuild of round 7's designs; needs `z59`'s caches), `z61` (frozen estimators on 512 new networks with an independent label bank; reads the models in `chatgpt2/round7/runs/`), `chatgpt2/round7/round7_estimator.py` (round 7's standalone estimator) |
| 7.19 a readable estimator | `z62` (44 named quantities; ridge, additive and forward-selected models), `z63` (invariant quantities only, the minimal two-quantity estimator, neuron roles, sampling shrunk toward the population mean), `z64` (double-pulse handoff probe), `z65` (where the residual lives: per-network loss by input region) |
| 7.20 memory probes, roles from weights, learned probe rows | `z66` (lower-input and distractor probes), `z67` (per-unit role classifier on weight descriptors), `z68` (learned MF4 rows), `z69` (learned rows with weight statistics and predicted roles), `z70` (learning curve), `z71` (learned rows from a mixed pool), `z72` (frozen estimators and their blind test), `z73` (nonlinear readouts), `z74` (learned rows from 8 value boxes), `z75` (2,048 training networks), `z76` (what the learned rows test; readable formulas with predicted roles), `z77` (a signal-to-noise ratio in place of the logit scale) |
| 7.21 blind tests, more training networks | `z72` (`--fit` before the test networks existed, `--score` after), `z75` (1,024 vs 2,048 training networks), `z78` (up to 512 learned rows from the combined pool, 2,048 training networks), `z79` (1,024 / 2,048 / 4,096 training networks), `z80` (blind test of the 4,096-network estimators) |
| exporters | `load`, `load_ckpts`, `load_seeds`, `load_sizes` |
| misc. diagnostics | `a1` (activity statistics), `a4` (homogeneity, activation patterns) |

`y1`–`y7` were added in an audit of the report. They back the corrections listed in REPORT.md section 10.
