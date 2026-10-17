from common import *
x = sample(200000); lg, hs = run(x)
for t in range(1,T):
    l = hs[:,t]@Wo.T; p2=np.argsort(x[:,:t+1],1)[:,-2]
    print(t, "acc of shifted readout (pos+9-t):", ((l.argmax(1))==p2+(9-t)).mean().round(3),
          " restricted to logits 9-t..9:", ((l[:,9-t:].argmax(1)+9-t)==p2+(9-t)).mean().round(3))
