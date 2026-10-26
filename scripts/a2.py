from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(400000)
lg, hs = run(x)
o = np.argsort(x,-1); p1=o[:,-1]; p2=o[:,-2]
pred = lg.argmax(-1); ok = pred==p2
A = np.zeros((T,T)); 
for i in range(T):
    for j in range(T):
        s=(p1==i)&(p2==j)
        if s.any(): A[i,j]=ok[s].mean()
print("acc by (row=pos max, col=pos 2nd max)\n", A)
# error types
wrong = ~ok
print("frac of errors that predicted the argmax:", (pred[wrong]==p1[wrong]).mean())
print("frac of errors predicting 3rd max:", (pred[wrong]==o[wrong,-3]).mean())
# gap between max and second when wrong vs right
srt = np.sort(x,-1)
print("mean m-s right/wrong", (srt[:,-1]-srt[:,-2])[ok].mean(), (srt[:,-1]-srt[:,-2])[wrong].mean())
print("mean s-3rd right/wrong", (srt[:,-2]-srt[:,-3])[ok].mean(), (srt[:,-2]-srt[:,-3])[wrong].mean())
