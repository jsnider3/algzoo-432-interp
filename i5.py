import numpy as np
from seeds_clocks import load, run, T
np.set_printoptions(precision=1, suppress=True, linewidth=220)
Wih,Whh,Wo=load("weights_h32.npz"); Hn=len(Wih)
print("32-neuron: weights-only impulse readout per neuron (count of ages a=0..9 decoded as slot 9-a, or with +1 delay)")
res=[]
for n in range(Hn):
    for sgn in [1,-1]:
        v=np.zeros(Hn); v[n]=sgn; p=[]
        for a in range(10): p.append(int(np.argmax(Wo@v))); v=Whh@v
        c0=sum(p[a]==9-a for a in range(10)); c1=sum(p[a]==8-a for a in range(9))
        res.append((max(c0,c1),n,sgn,c0,c1,p))
for r in sorted(res,reverse=True)[:5]: print(f"  {'+' if r[2]>0 else '-'}n{r[1]}: exact {r[3]}/10, one-step-delayed {r[4]}/9, picks {r[5]}")
x=np.random.default_rng(9).standard_normal((400000,T)); hs=run(x,Wih,Whh); h=hs[:,-1]
o=np.argsort(-x,1); aM=9-o[:,0]; M=-np.sort(-x,1)[:,0]; k=M>0.3
print("\nTime cells and their strongest inhibitors: mean h/M of inhibitor by ageM (0..9), time-cell firing rate by ageM")
for cell,inh in [(14,[12]),(22,[28,25]),(2,[30]),(15,[16,11])]:
    print(f"  cell n{cell} firing:", np.array([(h[k&(aM==a),cell]>0).mean() for a in range(10)]).round(2))
    for j in inh: print(f"     inhibitor n{j} (W={Whh[cell,j]:+.2f}) h/M:", np.array([(h[k&(aM==a),j]/M[k&(aM==a)]).mean() for a in range(10)]).round(1))
    # preactivation decomposition: inhibitor contribution vs rest, at the cell's preferred age vs others (using h at t=8 -> pre at t=9)
