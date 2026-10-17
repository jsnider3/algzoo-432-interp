import re, numpy as np
rows=[]
for f in ["k7_a.log","k7_b.log"]:
    for l in open(f):
        m=re.search(r"seed (\d) data \d+: acc ([\d.]+) clean-err ([\d.]+) pointer ([\d.]+) switch=(\w+)",l)
        if m: rows.append((int(m[1]),float(m[2]),float(m[3]),float(m[4]),m[5]=="True"))
# add earlier original-schedule from-init runs (k4 data 41, k5 data 51/52)
prev={0:[True,False,False],1:[False,False,False],2:[False,True],3:[True,True],4:[False,False,False]}
from scipy.stats import beta
print("seed | switch (k7) | + earlier | total rate [95% CI]")
tot=[0,0]
for s in range(5):
    k=[r[4] for r in rows if r[0]==s]; allv=k+prev[s]; a=sum(allv); n=len(allv); tot[0]+=a; tot[1]+=n
    lo,hi=beta.ppf(0.025,a,n-a+1) if a>0 else 0, beta.ppf(0.975,a+1,n-a) if a<n else 1
    print(f"  {s}  | {sum(k)}/{len(k)} | {sum(prev[s])}/{len(prev[s])} | {a}/{n} = {a/n:.2f} [{lo:.2f}, {hi:.2f}]")
print(f"all seeds: {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.2f}")
sw=np.array([r for r in rows if r[4]]); ns=np.array([r for r in rows if not r[4]])
print(f"at 2^25 seqs: switched runs mean acc {sw[:,1].mean():.3f}, clean-err {sw[:,2].mean():.4f} | non-switched mean acc {ns[:,1].mean():.3f}, clean-err {ns[:,2].mean():.4f}")
