ALGZOO POPULATION ESTIMATION — RECOVERED DELIVERABLES
Recovered 2026-10-18 from the preceding conversation.

The temporary workspace used for the original study was no longer available
when the user authorized persistent saving. The original upload had failed,
and the original 51-file experimental archive was not recovered.

This recovered package contains:

  population_estimator.py
    Complete estimator and benchmark implementation restored from the inline
    code in the previous answer. Script integrity details are in manifest.json.

  population_summary.json
    Reported results and protocol information reconstructed from the conversation.
    This is a recovered summary, not the original JSON file. Precision and scope
    are recorded in the file. No raw replicate observations have been invented.

  README_recovery.txt
    This explanation and commands for generating new benchmark results.

  manifest.json
    File sizes and SHA-256 checksums for these recovered deliverables.

NOT RECOVERED

The original per-replicate NPZ arrays, full per-target result tables, diagnostic
scripts other than the main estimator, timing trials, logs, and intermediate
caches are absent. The recovered archive must not be represented as the original
full experimental archive. The original source weights are also not included.

RUNNING THE RESTORED SCRIPT

Install NumPy and SciPy; threadpoolctl is optional. Torch is optional for CUDA.
The earlier study tested CPU execution; the CUDA path was not executed there.
Run from the directory containing the original supplied AlgZoo data files.

Single target, exactly 4096 target calls:

  python population_estimator.py \
    --population chatgpt2/pop1k_2e24_2e26.npz --snapshot s65536 \
    --estimate weights.npz --budget 4096 --pilot 256 \
    --bank-size 512 --reference 16384 --candidate-block 16384 \
    --out mystery_estimate.json

Recreate a new instance of the all-target margin benchmark:

  python population_estimator.py \
    --population chatgpt2/pop1k_2e24_2e26.npz --snapshot s65536 \
    --method margin --bank-size 512 --budget 4096 --pilot 256 \
    --reference 16384 --candidate-block 16384 --reps 32 --targets 0 \
    --zoo-root . --stress --seed 451203 --perturb-seed 8803 \
    --bootstrap 1000 --threads 2 --out continuous_all_4096.json

Population-mean benchmark:

  python population_estimator.py \
    --population chatgpt2/pop1k_2e24_2e26.npz --snapshot s65536 \
    --method mean --bank-size 1024 --budget 4096 \
    --reference 16384 --candidate-block 16384 --reps 32 --targets 0 \
    --zoo-root . --stress --seed 451203 --perturb-seed 8803 \
    --bootstrap 1000 --threads 2 --out continuous_mean_4096.json

Larger-budget margin benchmark:

  python population_estimator.py \
    --population chatgpt2/pop1k_2e24_2e26.npz --snapshot s65536 \
    --method margin --bank-size 512 --budget 16384 --pilot 1024 \
    --reference 65536 --candidate-block 65536 --reps 32 --targets 128 \
    --zoo-root . --stress --seed 451203 --perturb-seed 8803 \
    --bootstrap 1000 --threads 2 --out continuous_subset_16384.json

The script writes JSON summaries and compressed NPZ replicate arrays. Running
these commands is a new reproduction run; it does not recover the expired bytes.
The original study used Python 3.12, NumPy 2.3.5, and SciPy 1.17.0. Floating-point
and backend differences can affect predictions close to decision boundaries.

STATISTICAL INTERPRETATION

The estimator is unbiased for each fixed target over pilot, independent
reference, and correction randomness, subject to the implementation's numerical
convention. All target pilot calls count toward the budget. The reference
uncertainty is included in the variance estimate. The narrow normal interval
is approximate; the guaranteed bounded-variable interval is conservative.

Benchmark mode evaluates extra outcomes for paired MC comparison and efficient
simulation, but only counted pilot and accepted correction outcomes enter the
estimator. Single-target mode enforces the actual target-call budget. The reported
query savings did not establish a wall-clock speedup for the adaptive prototype.
