from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
act=[0,1,2,3,6,7,8,10,11,12,13,14,15]
A=Whh[np.ix_(act,act)]; ev,R=np.linalg.eig(A); Linv=np.linalg.inv(R)  # rows of Linv = left eigvecs
order=np.argsort(-abs(ev)); print("eigs:", ev[order]); 
x = sample(300000); lg, hs = run(x); h=hs[:,9][:,act]
o=np.argsort(-x,1); srt=-np.sort(-x,1); M,S=srt[:,0],srt[:,1]; aM=9-o[:,0]; aS=9-o[:,1]
Z=h@Linv.T   # modal coordinates
for k in order:
    if ev[k].imag<0: continue
    z=Z[:,k]
    print(f"\nmode {ev[k]:.3f} (|{abs(ev[k]):.3f}|, angle {np.degrees(np.angle(ev[k])):.1f}°)")
    # circular mean of phase and mean |z|/M by ageM
    for nm,ag in [('ageM',aM),('ageS',aS)]:
        ph=[np.degrees(np.angle(np.mean(z[ag==a]/np.abs(z[ag==a])))) for a in range(10)]
        conc=[np.abs(np.mean(z[ag==a]/np.abs(z[ag==a]))) for a in range(10)]
        print(f"  {nm}: phase", np.array(ph).round(0), "\n        concentration", np.array(conc).round(2))
