import json, numpy as np, torch as th
from scipy.stats import mannwhitneyu
from alg_zoo import DistRNN
from m4 import grads
R=[json.loads(l) for f in ["k12_a.log","k12_b.log","k12_24.log","k12_32.log","k12_40.log"] for l in open(f) if l.startswith('{')]
gap=[];sw=[]
for r in R:
    th.manual_seed(1000+r['init']); m=DistRNN(hidden_size=16,seq_len=10)
    g=grads(m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float))
    nr=np.sort(np.linalg.norm(g[1],axis=1)); gap.append(nr[-1]/nr[-2]); sw.append(r['switch'])
gap=np.array(gap); sw=np.array(sw)
new=np.array([r['init']>=24 for r in R])
print(f"n={len(R)}, switch rate {sw.mean():.2f} (first 24: {sw[~new].mean():.2f}, new 24: {sw[new].mean():.2f})")
for nm,sel in [("new 24 only (held-out replication)",new),("all 48",np.ones(len(R),bool))]:
    a=gap[sel&sw]; b=gap[sel&~sw]; print(f"  {nm}: winner/runner-up gradient ratio switched {a.mean():.3f} (n={len(a)}) vs not {b.mean():.3f} (n={len(b)}), Mann-Whitney p={mannwhitneyu(a,b).pvalue:.3f}")
