import re, json, numpy as np
from scipy.stats import wilcoxon, binomtest
D={}
for f in ["q3_0.log","q3_3.log","q3_6.log","q3b_8.log","q3b_14.log","q3b_19.log"]:
    for l in open(f):
        m=re.search(r"init (\d+), .*acc ([\d.]+) clean-err ([\d.]+) pointer ([\d.]+) switch=(\w+)",l)
        if m: D[int(m[1])]=(float(m[2]),float(m[3]),m[5]=="True")
N={}
for f in ["k12_a.log","k12_b.log"]:
    for l in open(f):
        if l.startswith('{'): r=json.loads(l); N[r['init']]=(r['acc'],r['clean'],r['switch'])
ks=sorted(set(D)&set(N)); a=np.array([D[k][0] for k in ks]); b=np.array([N[k][0] for k in ks])
sd=sum(D[k][2] for k in ks); sn=sum(N[k][2] for k in ks)
print(f"pairs {len(ks)}: switch delayed {sd}/{len(ks)} vs normal {sn}/{len(ks)}")
print(f"mean acc delayed {a.mean():.3f} vs normal {b.mean():.3f}; improved in {(a>b).sum()}/{len(ks)}; Wilcoxon p={wilcoxon(a,b).pvalue:.4f}; sign test p={binomtest(int((a>b).sum()),len(ks)).pvalue:.4f}")
print(f"mean clean-err delayed {np.mean([D[k][1] for k in ks]):.4f} vs normal {np.mean([N[k][1] for k in ks]):.4f}")
