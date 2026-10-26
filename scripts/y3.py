from common import *
# REPORT 3: share of S<0 inputs, and error count in the "clean" region S/M in [0.2,0.8), (S-R3)/M >= 0.2 (e8.py's sample)
x=sample(1500000,seed=13); pm=run(x)[0].argmax(1)
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S,R3=srt[:,0],srt[:,1],srt[:,2]; err=pm!=o[:,1]
q=S/M; g3=(S-R3)/M
print(f"P(S<0) sampled {np.mean(S<0):.4f}, exact 11/1024 = {11/1024:.4f}")
for hi in [1.0,np.inf]:
    s=(q>=.2)&(q<.8)&(g3>=.2)&(g3<hi)
    print(f"S/M in [0.2,0.8), (S-R3)/M in [0.2,{hi}): n={s.sum()}, errors={err[s].sum()} ({err[s].mean():.3%})")
