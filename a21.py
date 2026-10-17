from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
act=[0,1,2,3,6,7,8,10,11,12,13,14,15]; inact=[4,5,9]
A=Whh[np.ix_(act,act)]; ev,R=np.linalg.eig(A); Linv=np.linalg.inv(R)
x = sample(400000); lg, hs = run(x); h=hs[:,9]
o=np.argsort(-x,1); aM=9-o[:,0]; aS=9-o[:,1]; p1,p2=o[:,0],o[:,1]
Z=h[:,act]@Linv.T
kS=np.argmin(abs(ev-(0.538+0.501j))); kM=np.argmin(abs(ev-(0.698+0.645j)))
for nm,sel in [("S before M (inherited from old max)",p2<p1),("S after M (set directly)",p2>p1)]:
    z=Z[:,kS]; ph=[np.degrees(np.angle(np.mean((z/abs(z))[sel&(aS==a)]))) for a in range(10)]
    print(nm, "S-clock phase by ageS:", np.array(ph).round(0))
# per-mode contribution to logits; ablation
contrib = {}  # logits = sum_k Re(Wo_act R_k z_k) + Wo_inact h_inact
Wa=Wo[:,act]; base=0
groups={}
for k in range(len(ev)):
    key=round(abs(ev[k]),3), round(abs(np.degrees(np.angle(ev[k]))),1)
    groups.setdefault(key,[]).append(k)
def logits_without(drop_keys=(), drop_inact=False):
    L=np.zeros((len(x),10))
    for key,ks in groups.items():
        if key in drop_keys: continue
        for k in ks: L+=np.real(np.outer(Z[:,k],Wa@R[:,k]))
    if not drop_inact: L+=h[:,inact]@Wo[:,inact].T
    return L
print("recon check", np.abs(logits_without()-lg).max())
print("acc full", acc(lg,x))
for key in groups: print("drop mode", key, "acc", round(acc(logits_without([key]),x),3))
print("drop inactive-subsystem neurons 4,5,9 from readout: acc", round(acc(logits_without(drop_inact=True),x),3))
# keep only modes
for keep in [[(0.951,42.7),(0.735,43.0)]]:
    L=logits_without([k for k in groups if k not in keep]); print("keep only", keep, "acc", round(acc(L,x),3))
