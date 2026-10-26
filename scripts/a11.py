from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(200000); lg, hs = run(x)
for n in [1,6,7,8,4]:
    print(f"neuron {n}: threshold thr = -(Whh h_(t-1))/Wih ; fires when x_t {'<' if Wih[n]<0 else '>'} thr")
    for t in [3,6,9]:
        P=x[:,:t]; srt=np.sort(np.c_[np.zeros(len(x)),P],1); M,S,R3=srt[:,-1],srt[:,-2],srt[:,-3]
        thr = -(hs[:,t-1]@Whh[n])/Wih[n]
        A=np.c_[M,S,R3,np.ones(len(x))]; c,*_=np.linalg.lstsq(A,thr,rcond=None)
        r2=1-((A@c-thr)**2).mean()/thr.var()
        print(f"  t={t} thr ≈ {c[0]:.2f}*M + {c[1]:.2f}*S + {c[2]:.2f}*3rd + {c[3]:.2f}   R2={r2:.3f}  (max(0,..) incl.)")
