"""Can a mechanistic signal-to-noise ratio replace the logit scale in the readable formula?
For each of z63's 180 compact lower-slice MF4 rows, the margin m (lower inputs at mu(Q) = -0.577) and m_hi (the six
lower inputs at 0.20, z66's "high" rows): SNR = m / |m - m_hi|, how many such shifts of the lower inputs the decision
absorbs.  Both m and m - m_hi scale with W_o, so SNR does not change under the exact symmetries.  Per network: the
median SNR, the mean of log(1 + |SNR|) with the sign of m, and the share of rows with |SNR| < 1.
Readable formulas, cross-fitted on the 1,024 and applied to the 512 new networks:
  S-vs-R separation alone; + log s (z63); + each SNR summary instead of log s; + both.
Cost: the 180 base rows, the 180 shifted rows (their extra prefix cost) and a few operations per row.
Usage: python z77.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat, z60, z66
from z62 import ridge
if __name__=='__main__':
    Wt,_=load('pop_unif/b128_n1024_seed4.npz'); Wn,_=load('pop_unif/b128_n512_seed6.npz'); mz=zload('weights.npz')
    W=[th.cat([a.float(),b.float(),th.tensor(c,dtype=th.float32)[None]]) for a,b,c in zip(Wt,Wn,mz)]; Wnp=[w.numpy().astype(float) for w in W]
    y=np.r_[np.load('pop_unif/b128_n1024_seed4.acc22.npy'),np.load('pop_unif/b128_n512_seed6.acc22s61.npy'),0.95304775]
    tr,new,mys=np.arange(1024),np.arange(1024,1536),1536; te=np.r_[new,mys]; pq=(y[tr]*(1-y[tr])).mean(); pqn=(y[new]*(1-y[new])).mean()
    fold=np.random.default_rng(0).permutation(1024)%8
    _,m0,_=z66.run(Wnp,z66.BASE,z66.BT); x_hi,tg,_=z66.FAMS['lower inputs high']; _,m1,_=z66.run(Wnp,x_hi,tg)
    s=np.sqrt((m0**2).mean(1)); a=m0/s[:,None]; sep=(a/(.1+abs(a))).mean(1)
    snr=m0/np.maximum(np.abs(m0-m1),1e-9*s[:,None])
    Q={'S-vs-R separation':sep,'log s':np.log(s),'median SNR':np.median(snr,1),'mean signed log(1+|SNR|)':(np.sign(snr)*np.log1p(np.abs(snr))).mean(1),'share |SNR| < 1':(np.abs(snr)<1).mean(1)}
    base_fe=z66.trie_fe(z66.BASE)+(22*180+5000)/2880; hi_fe=z66.trie_fe(np.r_[z66.BASE,x_hi])-z66.trie_fe(z66.BASE)+10*180/2880
    print("correlations with accuracy on the 1,024: "+", ".join(f"{k} {np.corrcoef(v[tr],y[tr])[0,1]:+.2f}" for k,v in Q.items()))
    print(f"correlation of log s with the SNR summaries: "+", ".join(f"{k} {np.corrcoef(Q['log s'][tr],Q[k][tr])[0,1]:+.2f}" for k in list(Q)[2:]))
    for keys in (['S-vs-R separation'],['S-vs-R separation','log s'],['S-vs-R separation','median SNR'],['S-vs-R separation','mean signed log(1+|SNR|)'],
                 ['S-vs-R separation','share |SNR| < 1'],['S-vs-R separation','mean signed log(1+|SNR|)','share |SNR| < 1'],['S-vs-R separation','log s','mean signed log(1+|SNR|)','share |SNR| < 1']):
        X=np.stack([Q[k] for k in keys],1); cf=np.zeros(1024); fe=base_fe+(hi_fe if any('SNR' in k for k in keys) else 0)
        for f in range(8): A,B=tr[fold!=f],tr[fold==f]; cf[B]=ridge(X[A],y[A],X[B])[0]
        p,r=ridge(X[tr],y[tr],X[te]); r1=np.sqrt(((cf-y[tr])**2).mean()); rn=np.sqrt(((p[:-1]-y[new])**2).mean())
        print(f"  {' + '.join(keys):70s} {fe:6.1f} FE | cross-fit {100*r1:.3f} ({r1**2/(pq/fe):.2f}x) | new {100*rn:.3f} ({rn**2/(pqn/fe):.2f}x) | mystery {100*(p[-1]-y[mys]):+.2f}")
        print("    accuracy = "+f"{100*r.intercept_:.2f}% "+" ".join(f"{100*b:+.2f} x z({k})" for k,b in zip(keys,r.coef_)))
