from common import *
act=[0,1,2,3,6,7,8,10,11,12,13,14,15]
A=Whh[np.ix_(act,act)]; ev,R=np.linalg.eig(A); Li=np.linalg.inv(R)
x=sample(200000,seed=11); lg,hs=run(x); h=hs[:,-1]; pm=lg.argmax(1)
Z=h[:,act]@Li.T
def find(e): return [k for k in range(len(ev)) if abs(ev[k]-e)<1e-3 or abs(ev[k]-np.conj(e))<1e-3]
modes={'Sclock':find(0.538+0.501j),'Mclock':find(0.698+0.645j),'h83':find(0.110+0.931j),'h125':find(-0.478+0.688j)}
modes={k:v for k,v in modes.items()}
print({k:[np.round(ev[i],3) for i in v] for k,v in modes.items()})
def rotate(names, steps):
    Z2=Z.copy()
    for nm in names:
        for k in modes[nm]:
            Z2[:,k]=Z[:,k]*np.exp(1j*np.angle(ev[k])*steps)
    h2=h.copy(); h2[:,act]=np.real(Z2@R.T); return (h2@Wo.T).argmax(1)
o=np.argsort(-x,1); p2=o[:,1]
for names in [['Sclock'],['Sclock','h83','h125'],['Mclock'],['Mclock','h83','h125']]:
    for steps in [-1,1]:
        p=rotate(names,steps); d=p-pm
        sel=(pm>=1)&(pm<=8)
        print(f"rotate {'+'.join(names):22s} by {steps:+d} step: pred changed {(p!=pm).mean():.2f}; shift dist (−2..+2) {[round(((d==k)&sel).sum()/sel.sum(),2) for k in range(-2,3)]}")
print("--- with amplitude decay too (z *= lambda^steps)")
def rotate_full(names, steps):
    Z2=Z.copy()
    for nm in names:
        for k in modes[nm]: Z2[:,k]=Z[:,k]*ev[k]**steps
    h2=h.copy(); h2[:,act]=np.real(Z2@R.T); return (h2@Wo.T).argmax(1)
for names in [['Sclock','h83','h125']]:
    for steps in [-2,-1,1,2]:
        p=rotate_full(names,steps); d=p-pm; sel=(pm>=2)&(pm<=7)
        print(f"age {'+'.join(names)} by {steps:+d}: shift dist (−3..+3) {[round(((d==k)&sel).sum()/sel.sum(),2) for k in range(-3,4)]}")
