from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
x = sample(200000); lg, hs = run(x)
N=len(x)
def r2fit(Hm, y, tr=slice(0,150000), te=slice(150000,None)):
    A=np.c_[Hm,np.ones(len(Hm))]; c,*_=np.linalg.lstsq(A[tr],y[tr],rcond=None)
    p=A[te]@c; return 1-((p-y[te])**2).mean()/y[te].var(), c
for t in [3,5,7,9]:
    X=x[:,:t+1]; o=np.argsort(X,1); 
    M=X.max(1); S=np.sort(X,1)[:,-2]; ageM=t-o[:,-1]; ageS=t-o[:,-2]
    Mp=np.maximum(M,0); Sp=np.maximum(S,0)
    out=[]
    for name,y in [('M',M),('S',S),('relu M',Mp),('relu S',Sp),('ageM',ageM),('ageS',ageS),('M*ageS',M*ageS),('S*ageS',S*ageS),('x_t',X[:,-1])]:
        out.append(f"{name}:{r2fit(hs[:,t],y)[0]:.3f}")
    print("t",t," ".join(out))
