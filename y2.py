from common import *
# REPORT 2.4 mode-removal table: zero-ablation (as in a21.py) vs mean-ablation of each eigenmode of the active block
act=[0,1,2,3,6,7,8,10,11,12,13,14,15]; inact=[4,5,9]
ev,R=np.linalg.eig(Whh[np.ix_(act,act)]); Li=np.linalg.inv(R)
x=sample(300000,seed=77); lg,hs=run(x); h=hs[:,9]; Z=h[:,act]@Li.T; Wa=Wo[:,act]; tgt=target(x)
groups={}
for k in range(len(ev)): groups.setdefault((round(abs(ev[k]),3),round(abs(np.degrees(np.angle(ev[k]))),1)),[]).append(k)
def acc_without(drop,mode):
    L=h[:,inact]@Wo[:,inact].T
    for key,ks in groups.items():
        for k in ks:
            z=Z[:,k]
            if key==drop: z=0*z if mode=='zero' else np.full_like(z,z.mean())
            L=L+np.real(np.outer(z,Wa@R[:,k]))
    return (L.argmax(1)==tgt).mean()
print(f"full reconstruction {acc_without(None,'zero'):.3f}")
for key in sorted(groups,key=lambda k:-k[0]):
    print(f"mode |{key[0]:.3f}| {key[1]:5.1f}°: zero-ablated {acc_without(key,'zero'):.3f}  mean-ablated {acc_without(key,'mean'):.3f}")
