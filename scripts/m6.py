import json, numpy as np, torch as th
from scipy.stats import mannwhitneyu, spearmanr
from alg_zoo import DistRNN
from seeds_clocks import load
from m4 import grads
def feats(Wih,Whh,Wo):
    g=grads(Wih,Whh,Wo); nr=np.linalg.norm(g[1],axis=1); top=np.argmax(nr)
    return dict(dom=float(nr.max()/np.median(nr)), gap=float(np.sort(nr)[-1]/np.sort(nr)[-2]), top_in_grad=float(-g[0][top]), top_Wih=float(Wih[top]))
R=[json.loads(l) for f in ["k12_a.log","k12_b.log"] for l in open(f) if l.startswith('{')]
rows=[]
for r in R:
    th.manual_seed(1000+r['init']); m=DistRNN(hidden_size=16,seq_len=10)
    W=[m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float), m.rnn.weight_hh_l0.detach().numpy().astype(float), m.linear.weight.detach().numpy().astype(float)]
    f=feats(*W); f['switch']=r['switch']; rows.append(f)
print(f"fresh inits (n={len(rows)}):")
for k in ['dom','gap','top_in_grad','top_Wih']:
    a=[x[k] for x in rows if x['switch']]; b=[x[k] for x in rows if not x['switch']]
    print(f"  {k:12s} switched {np.mean(a):+.3f} vs not {np.mean(b):+.3f}  Mann-Whitney p={mannwhitneyu(a,b).pvalue:.3f}")
rates={0:5/15,1:2/15,2:4/14,3:14/14,4:2/15}
zs=[feats(*load(f"ckpt/s{s}_e0.npz")) for s in range(5)]
for k in ['dom','gap','top_in_grad']:
    print(f"zoo seeds {k}: values {[round(z[k],2) for z in zs]} vs rates {[round(rates[s],2) for s in range(5)]}  Spearman {spearmanr([z[k] for z in zs],[rates[s] for s in range(5)])[0]:+.2f}")
