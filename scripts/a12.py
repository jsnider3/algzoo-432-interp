from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(300000); lg, hs = run(x)
t=8  # threshold used when reading x_8, from state h_7 summarizing x_0..x_7
P=x[:,:t]; o=np.argsort(P,1); srt=np.sort(P,1); M,S=srt[:,-1],srt[:,-2]; aM=(t-1)-o[:,-1]; aS=(t-1)-o[:,-2]
for n in [1,6,7,8,4]:
    thr = -(hs[:,t-1]@Whh[n])/Wih[n]
    print(f"neuron {n}: thr ≈ a*M + b*S + c  by age of max at time of comparison (age measured at t-1)")
    for a in range(t):
        s=aM==a; A=np.c_[M[s],S[s],np.ones(s.sum())]; c,*_=np.linalg.lstsq(A,thr[s],rcond=None)
        r2=1-((A@c-thr[s])**2).mean()/thr[s].var()
        print(f"   ageM={a}: a={c[0]:.2f} b={c[1]:.2f} c={c[2]:.2f}  R2={r2:.3f}")
