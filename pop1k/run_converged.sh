#!/bin/bash
# Waits for the 2^30 pilot (pop1k/seed0.npz) and runs the role / anatomy / mean-field / latency analyses on it.
cd "$(dirname "$0")/.."
while kill -0 33987 2>/dev/null; do sleep 60; done
[ -f pop1k/seed0.npz ] || { echo "training ended but pop1k/seed0.npz is missing"; tail -5 pop1k/train_log.txt; exit 1; }
set -x
python z2.py pop1k/seed0.npz                                   > pop1k/z2_seed0.txt 2>&1
python z3.py pop1k/seed0.roles.npz pop1k/seed0.npz             > pop1k/z3_seed0.txt 2>&1
python z4.py pop1k/seed0.npz                                   > pop1k/z4_seed0.txt 2>&1
python z6.py pop1k/seed0.npz                                   > pop1k/z6_seed0.txt 2>&1
python z5.py pop1k/seed0.npz                                   > pop1k/z5_seed0.txt 2>&1
mkdir -p chatgpt2/roles_run_2e30
python -P chatgpt2/roles_round2/roles_mechanism.py --root . --population pop1k/seed0.npz --labels pop1k/seed0.roles.npz \
  --buckets pop1k/seed0.buckets.npz --out chatgpt2/roles_run_2e30 --gbm --mechanisms --threads 4 > chatgpt2/roles_run_2e30/log.txt 2>&1
echo done
