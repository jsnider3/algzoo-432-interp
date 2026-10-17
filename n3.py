import json, numpy as np, torch as th
from alg_zoo import DistRNN
from m4 import grads
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
R=[json.loads(l) for f in ["k12_a.log","k12_b.log","k12_24.log","k12_32.log","k12_40.log"] for l in open(f) if l.startswith('{')]
F=[];y=[]
for r in R:
    th.manual_seed(1000+r['init']); m=DistRNN(hidden_size=16,seq_len=10)
    Wih=m.rnn.weight_ih_l0.detach().numpy()[:,0].astype(float); Whh=m.rnn.weight_hh_l0.detach().numpy().astype(float); Wo=m.linear.weight.detach().numpy().astype(float)
    g=grads(Wih,Whh,Wo); rn=np.linalg.norm(g[1],axis=1); o=np.argsort(-rn)
    f=np.r_[np.sort(rn)[::-1][:6]/np.median(rn),            # top gradient-norm profile
            -g[0][o[:4]]*1e3,                                # input-weight gradient of top-4 pushed neurons
            Wih[o[:4]],                                     # their input weights
            np.diag(Whh)[o[:4]],                            # their self-connections
            np.sort(Wih)[::-1][:4], np.sort(Wih)[:4],       # input-weight extremes
            np.abs(np.linalg.eigvals(Whh)).max(), np.linalg.norm(Wo)]
    F.append(f); y.append(r['switch'])
F=np.array(F); y=np.array(y)
clf=lambda C: make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000))
for C in [0.03,0.1,1.0]:
    acc=cross_val_score(clf(C),F,y,cv=LeaveOneOut()).mean()
    rng=np.random.default_rng(0); null=[cross_val_score(clf(C),F,rng.permutation(y),cv=LeaveOneOut()).mean() for _ in range(200)]
    print(f"C={C}: LOO-CV accuracy {acc:.3f} (chance {max(y.mean(),1-y.mean()):.2f}); permutation p={np.mean(np.array(null)>=acc):.3f}")
