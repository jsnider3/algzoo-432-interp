"""Independent check of ChatGPT round 6's exact recurrent-prefix sharing on the z46 one-box designs (K = 1, 2, 4).
The design is fixed, so the trie of distinct input prefixes is built offline; on the target the recurrent product
W_hh h is computed once per distinct prefix state, W_ih x once per distinct scalar, and the readout once per row.
  1. rows rebuilt here from z46's own rule (not from ChatGPT's design()); the full forward reproduces the cached bits
     and margins of pop1k/seed0_z46_cells_b1_k{K}.npz;
  2. distinct prefixes counted with Python sets of byte strings; MAC and scalar-op cost by ChatGPT's formula
     (MACs: 256 per non-leaf node, 16 per distinct scalar, 160 per row; scalar ops: 16 adds per non-root node,
     16 ReLUs per node, 20 per row for the argmax and margin);
  3. the trie evaluation (torch, float32, all 1,030 networks) against the full forward: bits and margins.
Usage: python scripts/z48.py
"""
import itertools, numpy as np, torch as th
from batch_train import forward
from z2 import load
from seeds_clocks import load as zload
from scipy.special import ndtr, ndtri, betaincinv
dev='cuda' if th.cuda.is_available() else 'cpu'
P,_=load('pop1k/seed0.npz'); Z=[zload(f) for f in ["weights.npz"]+[f"weights_seed{i}.npz" for i in range(5)]]
Wz=[th.tensor(np.stack([z[i] for z in Z]),dtype=th.float32) for i in range(3)]
Wall=[th.cat([a,b]) for a,b in zip(P,Wz)]; NT=Wall[0].shape[0]
def mu(v): return -np.exp(-v*v/2)/np.sqrt(2*np.pi)/ndtr(v)
ARR=np.array(list(itertools.permutations(range(10),3))); A=len(ARR)
QOF={0:7,7:0,1:3,3:1,2:5,5:2,4:6,6:4}
free=[[x for x in range(10) if x not in r[:2]] for r in ARR]; QREP=np.array([f[QOF[f.index(r[2])]] for f,r in zip(free,ARR)])
def rows(K):
    """z46's feature rows for one box (NB = 1): (A, K, 2, 10) float32, cell index a*K + k."""
    cen=np.array([betaincinv(*ab,0.5) for ab in [(7,4),(1,3),(2,1)]]); fq=cen[0]; fr=fq+(1-fq)*cen[1]; fm=fr+(1-fr)*cen[2]
    Q,R,M=ndtri(fq),ndtri(fr),ndtri(fm); s=ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K)
    X=np.empty((A,K,2,10)); ai=np.arange(A)[:,None]; ki=np.arange(K)[None,:]
    X[:,:,0]=mu(Q); X[ai,ki,0,ARR[:,0,None]]=M; X[ai,ki,0,ARR[:,1,None]]=s; X[ai,ki,0,ARR[:,2,None]]=R; X[ai,ki,0,QREP[:,None]]=Q
    X[:,:,1]=mu(R); X[ai,ki,1,ARR[:,0,None]]=M; X[ai,ki,1,ARR[:,1,None]]=s; X[ai,ki,1,ARR[:,2,None]]=R
    return X.astype(np.float32)
def margin(L,y):
    yt=y[...,None]; return L.gather(-1,yt)[...,0]-L.scatter(-1,yt,-1e30).max(-1).values
print("K  rows    distinct prefixes by depth (1..10)                         MAC FE   scalar FE  total FE  494K+69.4  saving")
for K in (1,2,4):
    X=rows(K); x=X.reshape(-1,10); n=len(x); y=np.repeat(ARR[:,1],K*2)
    # 2. distinct prefixes, by sets of byte strings
    cnt=[len({x[i,:t+1].tobytes() for i in range(n)}) for t in range(10)]; vals=len(set(x.ravel().tolist()))
    mac=256*sum(cnt[:-1])+16*vals+160*n; other=16*sum(cnt[1:])+16*sum(cnt)+20*n
    print(f"{K}  {n:6d}  {cnt}  {mac/2880:8.2f}  {other/2880:8.2f}  {(mac+other)/2880:8.2f}  {494*K+69.4:8.1f}  {n*2880/(mac+other):.2f}x")
    # trie: node ids per depth, parent of each node
    lev=[]; par=np.zeros(n,np.int64)
    for t in range(10):
        key={}; idx=np.empty(n,np.int64)
        for i in range(n):
            k=(int(par[i]),x[i,t].tobytes()); idx[i]=key.setdefault(k,len(key))
        p_of=np.zeros(len(key),np.int64); v_of=np.zeros(len(key),np.float32)
        for (pp,vb),j in key.items(): p_of[j]=pp; v_of[j]=np.frombuffer(vb,np.float32)[0]
        lev.append((th.tensor(p_of,device=dev),th.tensor(v_of,device=dev))); par=idx
    leaf=th.tensor(par,device=dev); assert [len(l[0]) for l in lev]==cnt
    # 1 and 3: full forward vs cache, trie vs full forward
    cache=np.load(f'pop1k/seed0_z46_cells_b1_k{K}.npz'); bad=dict(full_vs_cache_bits=0,trie_vs_full_bits=0); mdiff=[0.,0.]
    for c in range(0,NT,206):
        sl=slice(c,min(NT,c+206)); W=[w[sl].to(dev) for w in Wall]; N=W[0].shape[0]; yt=th.tensor(y,device=dev)[None].expand(N,-1)
        with th.no_grad():
            L=forward(*W,th.tensor(x,device=dev)[None].expand(N,-1,-1))
            h=None
            for t,(p,v) in enumerate(lev):
                inp=v[None,:,None]*W[0][:,None,:]
                h=th.relu(inp) if t==0 else th.relu(th.bmm(h,W[1].transpose(1,2))[:,p]+inp)
            Lt=th.bmm(h,W[2].transpose(1,2))[:,leaf]
        ok=(L.argmax(-1)==yt).view(N,A*K,2).cpu().numpy(); mg=margin(L,yt).view(N,A*K,2).cpu().numpy()
        okt=(Lt.argmax(-1)==yt).view(N,A*K,2).cpu().numpy(); mgt=margin(Lt,yt).view(N,A*K,2).cpu().numpy()
        bad['full_vs_cache_bits']+=int((ok[...,0]!=cache['I4'][sl]).sum()+(ok[...,1]!=cache['I3'][sl]).sum())
        bad['trie_vs_full_bits']+=int((ok!=okt).sum())
        mdiff[0]=max(mdiff[0],float(np.abs(mg[...,0]-cache['m4'][sl]).max()),float(np.abs(mg[...,1]-cache['m3'][sl]).max()))
        mdiff[1]=max(mdiff[1],float(np.abs(mg-mgt).max()))
    print(f"   bit mismatches over {NT} networks x {n} rows: {bad};  max |margin diff| full vs cache {mdiff[0]:.2e}, trie vs full {mdiff[1]:.2e}")
