import re
from scipy.stats import beta
cnt={s:[0,0] for s in range(5)}
for f in ["k7_a.log","k7_b.log","l4.log"]:
    for l in open(f):
        m=re.search(r"seed (\d) data \d+: .*switch=(\w+)",l)
        if m: cnt[int(m[1])][0]+=m[2]=="True"; cnt[int(m[1])][1]+=1
prev={0:(1,3),1:(0,3),2:(1,2),3:(2,2),4:(0,3)}
T=[0,0]
for s in range(5):
    a=cnt[s][0]+prev[s][0]; n=cnt[s][1]+prev[s][1]; T[0]+=a; T[1]+=n
    lo=beta.ppf(0.025,a,n-a+1) if a>0 else 0.0; hi=beta.ppf(0.975,a+1,n-a) if a<n else 1.0
    print(f"seed {s}: {a}/{n} = {a/n:.2f}  [{lo:.2f}, {hi:.2f}]")
print(f"all: {T[0]}/{T[1]} = {T[0]/T[1]:.2f}")
