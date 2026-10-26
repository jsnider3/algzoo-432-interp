from common import *
x = sample(300000, seed=5); pm=run(x)[0].argmax(1)
allw=np.r_[np.abs(Wih),np.abs(Whh).ravel(),np.abs(Wo).ravel()]
for tau in [0.01,0.03,0.05,0.1,0.2,0.3,0.5]:
    f=lambda W: np.where(np.abs(W)<tau,0,W)
    p=run(x,f(Wih),f(Whh),f(Wo))[0].argmax(1)
    print(f"prune |w|<{tau}: nonzero {(allw>=tau).sum()}/432  agreement {(p==pm).mean():.4f}  acc {(p==target(x)).mean():.4f}")
for dec in [2,1]:
    r=lambda W: np.round(W,dec); p=run(x,r(Wih),r(Whh),r(Wo))[0].argmax(1)
    print(f"round to {dec} decimals: agreement {(p==pm).mean():.4f}")
