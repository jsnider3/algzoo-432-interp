from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
ev=np.linalg.eigvals(Whh); print("eig Whh", ev[np.argsort(-abs(ev))], "\n|.|", np.sort(abs(ev))[::-1])
# eigs of subsystem of neurons that are almost always active late
act=[0,1,2,3,6,7,8,10,11,12,13,14,15]
ev=np.linalg.eigvals(Whh[np.ix_(act,act)]); print("eig active sub", ev[np.argsort(-abs(ev))])
x = sample(200000); lg, hs = run(x)
# do intermediate readouts track anything?
for t in range(T):
    l = hs[:,t]@Wo.T; print(t, "argmax Wo h_t == 2nd argmax of prefix?", (l.argmax(1)==np.argsort(x[:,:t+1],1)[:,-2] if t>0 else 0).mean() if t>0 else '-', " dist of argmax:", np.bincount(l.argmax(1),minlength=10)/len(x))
