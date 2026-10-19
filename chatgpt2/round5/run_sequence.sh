export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMBA_NUM_THREADS=2

python - <<'PY'
import json
from pathlib import Path
p = Path("r6_results")
p.mkdir(exist_ok=True)
settings = []
for n, K, q in [(1, 8, 4), (2, 16, 8)]:
    for keep in [1, 2, 4, 8]:
        settings.append(dict(k=3, n=n, K=K, q=q, keep=keep))
for keep in [4, 8, 56]:
    settings.append(dict(k=4, n=1, K=8, q=8, keep=keep))
settings.append(dict(k=4, n=2, K=8, q=8, keep=8))
f = p / "correction_frozen.json"
if not f.exists():
    f.write_text(json.dumps({
        "settings": settings,
        "note": "Replays the fixed round-6 settings; no new tuning."
    }, indent=2))
PY

python r6_verify.py
python r6_low.py

for split in dev held; do
  python r6_core.py --split "$split" \
    --out "r6_results/anchors_${split}.json"
  python r6_event.py --split "$split" \
    --out "r6_results/event_${split}.json"
  python r6_exact_anchor.py --split "$split" --orders 3,4,6 \
    --out "r6_results/exact_anchor_${split}.json"

  python r6_correction.py --split "$split" --k 3 --n 1 --K 8 --q 4 --keep 8
  python r6_correction.py --split "$split" --k 3 --n 2 --K 16 --q 8 --keep 8
  python r6_correction.py --split "$split" --k 4 --n 1 --K 8 --q 8 --keep 56
  python r6_correction.py --split "$split" --k 4 --n 2 --K 8 --q 8 --keep 8

  # Fixed matched z38 comparator:
  python r6_correction.py --split "$split" --k 3 --n 3 --K 8 --q 8 --keep 8
  python r6_correction.py --split "$split" --report
done

python r6_summary.py
python r6_bounds_minimal.py
python r6_bounds_check.py

# Measurement only; these outputs never enter an estimator:
python r6_correction.py --measure 256
python r6_geometry.py --dev-count 8 --log2-inputs 16

# Optional dense verification of an exact S node:
python r6_exact_anchor.py --split dev --orders 3,4,6 \
  --resume --verify --out r6_results/exact_anchor_dev.json
