from common import *
act=[0,1,2,3,6,7,8,10,11,12,13,14,15]
A=Whh[np.ix_(act,act)]; ev,R=np.linalg.eig(A)
x = sample(300000); lg, hs = run(x); Linv=np.linalg.inv(R); Z=hs[:,9][:,act]@Linv.T
o=np.argsort(-x,1); aS=9-o[:,1]
for target_ev in [0.538+0.501j, 0.698+0.645j, 0.11+0.931j]:
    k=np.argmin(abs(ev-target_ev)); g=Wo[:,act]@R[:,k]  # logit_i contribution = 2 Re(g_i z)
    # contribution peaks when phase(z) = -phase(g_i)
    pref=np.degrees(-np.angle(g))%360
    ph=np.array([np.degrees(np.angle(np.mean((Z[:,k]/abs(Z[:,k]))[aS==a])))%360 for a in range(10)])
    print(f"mode {ev[k]:.2f}: logit i preferred phase  ", pref.round(0), "\n             S-clock phase at ageS=9-i", ph[::-1].round(0), " |g|", abs(g).round(1))
