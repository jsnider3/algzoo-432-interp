"""Small regression checks only; randomized fixtures never enter an estimator."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import json
from pathlib import Path
import numpy as np
import numba
from scipy.special import ndtr
from r6_core import load, forward, prefix_forward
from r6_event import last_input
from r6_exact_anchor import nodes, exact_node
from lines import trace_line


def verify():
    W = load(np.array([0,1024]))
    tokens = np.array([[0]*10,[0]*9+[1],[0]*8+[1,2],
        [0,1,2,0,1,2,0,1,2,0],[0,1,2,0,1,2,0,1,2,0],
        [1,0,2,2,0,1,1,2,0,0]])
    values = np.array([-1.25,.125,1.5])
    direct = forward(W,values[tokens])
    shared,cost = prefix_forward(W,tokens,values)
    assert np.allclose(direct,shared,rtol=1e-12,atol=1e-10)
    result = dict(prefix_max_logit_difference=float(np.max(np.abs(direct-shared))),
                  prefix_counted_cost=float(cost),direct_rows=len(tokens))

    # Independent Gaussian prefixes are a measurement fixture, not an estimator.
    x = np.zeros((6,10))
    x[:,:9] = np.random.default_rng(2026101907).standard_normal((6,9))
    top = np.argsort(x[:,:9],axis=1)[:,-2:]
    jp,ip = top[:,0],top[:,1]
    S,M = x[np.arange(6),jp],x[np.arange(6),ip]
    h = np.zeros((len(W[0]),6,16))
    for t in range(9):
        h = np.maximum(h@W[1].transpose(0,2,1)+W[0][:,None,:]*x[None,:,t,None],0)
    swept,_ = last_input(W,h,M,S,ip,jp)
    xx = np.repeat(x,3,axis=0)
    lo = np.column_stack([np.full(6,-12.),S,M]).ravel()
    hi = np.column_stack([S,M,np.full(6,12.)]).ravel()
    target = np.column_stack([jp,np.full(6,9),ip]).ravel().astype(np.int64)
    p = np.full(len(xx),9,np.int64)
    checked = []
    for i in range(len(W[0])):
        probability,_,overflow = trace_line(*[w[i] for w in W],xx,p,lo,hi,target)
        assert not overflow.any()
        checked.append((probability*(ndtr(hi)-ndtr(lo))).reshape(6,3).sum(1))
    checked = np.array(checked)
    # The omitted Gaussian tails beyond +/-12 have total mass below 4e-33.
    assert np.allclose(swept,checked,rtol=1e-10,atol=1e-10)
    result["last_input_max_probability_difference"] = float(np.max(np.abs(swept-checked)))

    wi = np.tile([1.,-1.],8)
    wh = np.zeros((16,16)); wh[:,0] = -1
    wo = np.random.default_rng(0).standard_normal((10,16)); wo[0] = 0
    hh = np.zeros((1,1,16)); hh[0,0,0] = 1
    probability,_ = last_input([wi[None],wh[None],wo[None]],hh,
        np.array([3.]),np.array([2.]),np.array([1]),np.array([0]))
    expected = float(ndtr(1)-ndtr(-1))
    assert abs(float(probability[0,0])-expected)<1e-14
    result["all_off_probability"] = float(probability[0,0])
    result["all_off_expected"] = expected

    x,R,M,_,_,_ = nodes(3)[4]
    ww = [w[0] for w in W]
    a,c,_,_ = exact_node(ww,x,R,M,512)
    b,retry_cost,discarded,capacity = exact_node(ww,x,R,M,4)
    assert np.array_equal(a,b) and discarded>0 and retry_cost>c
    result["overflow_retry"] = dict(max_probability_difference=float(np.max(np.abs(a-b))),
        baseline_mac_forwards=c,with_retries_mac_forwards=retry_cost,
        discarded_lines=discarded,final_capacity=capacity)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",default="r6_results/verify.json")
    args = parser.parse_args()
    numba.set_num_threads(2)
    result = verify()
    path = Path(args.out); path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
