#!/bin/bash
# reruns after the init_* key fix: z3 and z6 (crashed at the weights step), ChatGPT's script with an explicit snapshot, z7
set -x
cd /mnt/c/Users/josh/Documents/Git/algzoo-432-interp
python z3.py pop1k/seed0.roles.npz pop1k/seed0.npz > pop1k/z3_seed0.txt 2>&1
python z6.py pop1k/seed0.npz > pop1k/z6_seed0.txt 2>&1
python z7.py pop1k/seed0.npz > pop1k/z7_seed0.txt 2>&1
python -P chatgpt2/roles_round2/roles_mechanism.py --root . --population pop1k/seed0.npz --snapshot s1048576 --labels pop1k/seed0.roles.npz --buckets pop1k/seed0.buckets.npz --out chatgpt2/roles_run_2e30 --gbm --mechanisms --threads 4 > chatgpt2/roles_run_2e30/log.txt 2>&1
echo done
