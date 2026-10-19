"""MF3 anchor: tensor Beta quadrature and exact Gaussian S-line integration.

Uses the supplied lines.trace_line; reports its measured matrix/readout MAC
count (2,880 MACs/forward), a lower bound excluding roots, sorting and CDF work.
No assumed prefix-tree acceleration. Failed overflow attempts are also charged.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import json
import time
from pathlib import Path
import numpy as np
import numba
from scipy.special import ndtri
from lines import trace_line
from r6_core import ARR, load, rule, forward, score


def nodes(n):
    ur, wr = rule(n,8,3)
    ug, wg = rule(n,2,1)
    out = []
    for a,wa in zip(ur,wr):
        for g,wb in zip(ug,wg):
            b = a+(1-a)*g
            R,M = ndtri(a),ndtri(b)
            mu = -np.exp(-R*R/2)/np.sqrt(2*np.pi)/a
            x = np.full((len(ARR),10),mu)
            x[np.arange(len(ARR)),ARR[:,0]] = M
            x[np.arange(len(ARR)),ARR[:,1]] = R
            x[np.arange(len(ARR)),ARR[:,2]] = R
            out.append((x,R,M,float(wa*wb),float(a),float(b)))
    return out


def exact_node(W,x,R,M,cap=512):
    remaining = np.arange(len(x))
    prob = np.empty(len(x))
    total_cost = 0.0
    discarded = 0
    p = np.ascontiguousarray(ARR[:,1],dtype=np.int64)
    while len(remaining):
        pp,cc,overflow = trace_line(*W,x[remaining],p[remaining],
            np.full(len(remaining),R),np.full(len(remaining),M),p[remaining],cap)
        total_cost += float(cc.sum())
        good = overflow == 0
        if np.any((pp[good] < -1e-10) | (pp[good] > 1+1e-10)):
            raise ArithmeticError("Invalid exact line probability")
        prob[remaining[good]] = pp[good]
        remaining = remaining[~good]
        discarded += len(remaining)
        if len(remaining):
            cap *= 2
    return prob,total_cost,discarded,cap


def mf3_exact(W,n,cap=512):
    est,cost,discarded,maxcap = 0.0,0.0,0,cap
    for x,R,M,w,a,b in nodes(n):
        p,c,d,used = exact_node(W,x,R,M,cap)
        est += w*p.mean()
        cost += c
        discarded += d
        maxcap = max(maxcap,used)
    return est,cost,discarded,maxcap


def verify(W,n=3,K=4096,cap=512):
    """One central outer node, all720 lines, checked against a dense CDF grid."""
    x,R,M,w,a,b = nodes(n)[n*n//2]
    exact,c,d,used = exact_node(W,x,R,M,cap)
    s = ndtri(a+(b-a)*(np.arange(K)+.5)/K)
    dense = np.empty(len(ARR))
    for start in range(0,len(ARR),8):
        part = np.arange(start,min(start+8,len(ARR)))
        xx = np.repeat(x[part,None,:],K,axis=1)
        xx[np.arange(len(part)),:,ARR[part,1]] = s
        logits = forward([v[None] for v in W],xx.reshape(-1,10))[0]
        dense[part] = (logits.argmax(1).reshape(len(part),K)==ARR[part,1,None]).mean(1)
    return dict(n=n,K=K,lines=len(ARR),mean_error_pp=float(100*(dense-exact).mean()),
                max_abs_error_pp=float(100*np.max(np.abs(dense-exact))),
                rms_error_pp=float(100*np.sqrt(np.mean((dense-exact)**2))),
                exact_node_mac_forwards=c,discarded_lines=d,maxcap=used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",choices=("dev","held"),default="dev")
    ap.add_argument("--orders",default="3,4")
    ap.add_argument("--dev-count",type=int,default=64)
    ap.add_argument("--maxp",type=int,default=512)
    ap.add_argument("--verify",action="store_true")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--out",default="r6_results/exact_anchor_dev.json")
    a = ap.parse_args()
    pop = np.arange(0 if a.split=="dev" else 8,1024,16)[:a.dev_count]
    ids = np.r_[pop,np.arange(1024,1030)]
    W = load(ids)
    result = []
    output = Path(a.out)
    output.parent.mkdir(exist_ok=True)
    if a.resume and output.exists():
        result = json.loads(output.read_text())
        if any(row["ids"] != ids.tolist() for row in result):
            raise ValueError("Cannot resume a different network cohort")
    if a.verify:
        v = verify([w[0] for w in W],cap=a.maxp)
        output.with_suffix(".verification.json").write_text(json.dumps(v,indent=2))
        print("verification",v,flush=True)
    for n in map(int,a.orders.split(",")):
        if any(row["n"]==n for row in result):
            continue
        start = time.time()
        est,cost,discarded,maxcaps = [],[],[],[]
        for i in range(len(ids)):
            e,c,d,m = mf3_exact([w[i] for w in W],n,a.maxp)
            est.append(e); cost.append(c); discarded.append(d); maxcaps.append(m)
            if i%8==0:
                print("order",n,"network",int(ids[i]),"MAC forwards",round(c,1),
                      "seconds",round(time.time()-start,1),flush=True)
        est,cost = np.array(est),np.array(cost)
        row = score(est,ids,float(cost[:len(pop)].mean()))
        row.update(n=n,cost_is_lower_bound=True,cost_definition=__doc__,
                   costs_per_network=cost.tolist(),discarded_lines=discarded,maxcaps=maxcaps,
                   seconds=time.time()-start)
        ref = np.load("pop1k/reference.npz")
        pq = ref["acc"][ids]*(1-ref["acc"][ids])
        sample = float(100*np.sqrt(np.mean(pq[:len(pop)]/cost[:len(pop)])))
        metric = row[a.split]
        metric.update(sampling_at_network_costs=sample,
                      ratio_at_network_costs=metric["rms"]**2/sample**2,
                      cost_min=float(cost[:len(pop)].min()),cost_max=float(cost[:len(pop)].max()),
                      anchor_rms_pp=float(100*np.sqrt(np.mean((est[:len(pop)]-ref["mf3"][pop])**2))))
        result.append(row)
        output.write_text(json.dumps(result,indent=2))
        print("completed",n,a.split,metric,flush=True)


if __name__ == "__main__":
    numba.set_num_threads(2)
    main()
