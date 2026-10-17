from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(400000); lg, hs = run(x); N=len(x)
# belief about position of 2nd max at each t via shifted readout restricted to valid ages
bel = np.full((N,T),-1); truth=np.full((N,T),-1)
for t in range(1,T):
    l=hs[:,t]@Wo.T; bel[:,t]=l[:,9-t:].argmax(1); truth[:,t]=np.argsort(x[:,:t+1],1)[:,-2]
wrong_final = bel[:,9]!=truth[:,9]
print("final err", wrong_final.mean())
# for final errors: was belief right at t=8 ?  first wrong step & last right step
right = bel==truth
# last time the belief transitioned from right to wrong and stayed wrong
first_bad = np.array([ (np.where(~right[i,1:])[0]+1) for i in np.where(wrong_final)[0][:0]])
lastgood = np.full(N,-1)
for t in range(1,T):
    lastgood[right[:,t]] = t
s=wrong_final
fb = lastgood[s]+1  # step where it went wrong (and stayed wrong)
print("step at which belief broke for good (1..9), counts:", np.bincount(fb,minlength=10))
# classify the input at breaking step relative to prefix M,S
idx=np.where(s)[0]; cases={'x>M (new max)':0,'S<x<M (new 2nd)':0,'x<S (no change)':0}
gapS=[]; ageMl=[]
for i,t in zip(idx,fb):
    if t==0: continue
    P=x[i,:t]; srt=np.sort(P); M=srt[-1]; S=srt[-2] if t>1 else -np.inf; xt=x[i,t]
    if xt>M: cases['x>M (new max)']+=1
    elif xt>S: cases['S<x<M (new 2nd)']+=1
    else: cases['x<S (no change)']+=1
    gapS.append(xt-S); ageMl.append(t-1-np.argmax(P))
print(cases); gapS=np.array(gapS); print("median |x-S| at break:", np.median(np.abs(gapS)), " vs typical |x-S|:", np.median(np.abs(x[:,5]-np.sort(x[:,:5],1)[:,-2])))
print("ageM (at t-1) at break:", np.bincount(ageMl,minlength=9))
