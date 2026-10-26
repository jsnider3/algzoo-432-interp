from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
x = sample(2000000, seed=1); lg, hs = run(x); pm=lg.argmax(1); N=len(x)
o=np.argsort(-x,1); Xr=np.take_along_axis(x,o,1); ages=9-o
rank_of_pred=np.argsort(o,1)[np.arange(N),pm]
print("agreement of true answer:", (pm==o[:,1]).mean())
print("rank of model's pick (0=max):", (np.bincount(rank_of_pred,minlength=10)/N).round(4))
Lr=np.take_along_axis(lg,o,1)
# is the model's choice always the argmax among top-k ranks? i.e. upper bound if we restrict to top k
for k in [2,3,4,5]:
    print(f"pick lies in top-{k}:", (rank_of_pred<k).mean())
