"""Cell caches for the mystery-recipe networks, in z46's layout, so G(1,K) and new cell features can be fitted
without a GPU.  Networks: the 1,024 of pop_unif/b128_n1024_seed4.npz, the 64 further batch-128 networks
(b128_seed0), the 64 batch-64 networks (b64_seed5) and the mystery model (weights.npz), in that order.
Writes pop_unif/cells_k{K}.npz for K = 1, 2, 4 with, per network: I4, I3 (bool) and m4, m3 (float32), each
(N, 720 K), for cell c = a K + k (a indexes arr = the 720 (M, S, R) position triples, Q at qrep[a]; k the S slice);
F (gfeat.summary's per-network summary), stats (gfeat.stats's 20 weight statistics), acc (accuracy on 2^22 N(0,1)
inputs, the *.acc22.npy labels), group, and net_fe (the trie's network-row cost in FE).
Usage: python scripts/z59.py
"""
import numpy as np, torch as th
from z2 import load
from seeds_clocks import load as zload
import gfeat
G=[('unif',*load('pop_unif/b128_n1024_seed4.npz')[:1],'pop_unif/b128_n1024_seed4.acc22.npy'),
   ('b128_seed0',*load('pop_unif/b128_seed0.npz')[:1],'pop_unif/b128_seed0.acc22.npy'),
   ('b64_seed5',*load('pop_unif/b64_seed5.npz')[:1],'pop_unif/b64_seed5.acc22.npy')]
z=zload('weights.npz'); W=[th.cat([g[1][i].float() for g in G]+[th.tensor(z[i],dtype=th.float32)[None]]) for i in range(3)]
acc=np.r_[np.concatenate([np.load(g[2]) for g in G]),0.95304775]
group=np.array(sum([[g[0]]*len(g[1][0]) for g in G],[])+['mystery'])
st=gfeat.stats(*[w.numpy().astype(float) for w in W])
for K in (1,2,4):
    F,net,C,(I4,I3,m4,m3)=gfeat.summary(W,1,K,cells=True)
    np.savez_compressed(f'pop_unif/cells_k{K}.npz',I4=I4,I3=I3,m4=m4,m3=m3,F=F,stats=st,acc=acc,group=group,net_fe=net,
                        arr=gfeat.ARR,qrep=gfeat.QREP,K=K)
    print(f"K = {K}: {len(acc)} networks x {C} cells, {F.shape[1]} summary features, network rows {net:.1f} FE; "
          f"mean I4 {I4.mean():.4f}, I3 {I3.mean():.4f}, accuracy {acc.mean():.4f}",flush=True)
