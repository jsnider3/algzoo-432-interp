import json, numpy as np
from scipy.stats import mannwhitneyu
R=[json.loads(l) for f in ["k12_a.log","k12_b.log"] for l in open(f) if l.startswith('{')]
sw=[r for r in R if r['switch']]; ns=[r for r in R if not r['switch']]
print(f"{len(R)} fresh inits: {len(sw)} switched ({len(sw)/len(R):.2f})")
print(f"acc at 2^25: switched {np.mean([r['acc'] for r in sw]):.3f} vs not {np.mean([r['acc'] for r in ns]):.3f}")
for k in ['n_pos_in','max_in','min_in','mean_abs_in','spec','diag_mean','pos_in_self','max_in_x_self']:
    a=[r[k] for r in sw]; b=[r[k] for r in ns]
    p=mannwhitneyu(a,b).pvalue if len(a) and len(b) else np.nan
    print(f"  {k:14s} switched {np.mean(a):+.3f}  not {np.mean(b):+.3f}   Mann-Whitney p={p:.3f}")
