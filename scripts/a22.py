from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(600000); lg, hs = run(x); N=len(x); pm=lg.argmax(1)
# model belief of age of 2nd max at each t (shifted readout, restricted)
belS=np.full((N,T),-1)
for t in range(1,T): belS[:,t]= 9-(hs[:,t]@Wo.T)[:,9-t:].argmax(1)-(9-t)  # age
# Estimate decision threshold: at step t, given (true) M,S of prefix and x_t in (S,M), P(model says age 0) vs u=(x-S)/(M-S)
# also x_t in (S - (M-S), S) to see false positives. Fit per ageM, pooled over t>=3.
rows={}
for t in range(2,T):
    P=x[:,:t]; o=np.argsort(-P,1); srt=-np.sort(-P,1); M,S=srt[:,0],srt[:,1]; aM=t-1-o[:,0]
    xt=x[:,t]; u=(xt-S)/(M-S); good=(belS[:,t-1]==(t-1-o[:,1]))  # belief right before
    said_new = belS[:,t]==0
    for a in range(t):
        s=good&(aM==a)&(xt<M)&(u>-1)
        rows.setdefault(a,[]).append((u[s],said_new[s]))
thr={}
for a in sorted(rows):
    u=np.concatenate([r[0] for r in rows[a]]); y=np.concatenate([r[1] for r in rows[a]])
    bins=np.linspace(-1,1,41); ib=np.digitize(u,bins); p=np.array([y[ib==b].mean() if (ib==b).sum()>20 else np.nan for b in range(1,len(bins))])
    c=(bins[:-1]+bins[1:])/2; i=np.nanargmin(abs(p-0.5)); thr[a]=c[i]
    print(f"ageM={a}: n={len(u)} u at P(new 2nd)=0.5: {c[i]:+.3f}   P curve:", p[16:28].round(2))
np.save("thr.npy", np.array([thr.get(a,0) for a in range(10)]))
