"""Measurement-only gate-history diagnostic; not an accuracy estimator.

The first half of independent iid Gaussian inputs discovers patterns; the second
half measures mass coverage. Error fragments also fix top-two and output indices,
so they form a particular convex-cone partition, not maximal convex error pieces.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import json
from pathlib import Path
import numpy as np


def keys(gates, group=None):
    """Opaque exact gate bitstrings, optionally prefixed by a grouping label."""
    raw = np.ascontiguousarray(gates).view(np.dtype((np.void, gates.shape[1]*2))).ravel()
    if group is None:
        return raw
    out = np.empty(len(raw), dtype=[("group", "u4"), ("pattern", raw.dtype)])
    out["group"], out["pattern"] = group, raw
    return out


def species(key, split, keep=None):
    """Discovery/test counts and held-out coverage of discovered species."""
    if keep is None:
        keep = np.ones(len(key), bool)
    which = np.arange(len(key))[keep] < split
    unique, inverse = np.unique(key[keep], return_inverse=True)
    a = np.bincount(inverse[which], minlength=len(unique))
    b = np.bincount(inverse[~which], minlength=len(unique))
    na, nb = int(a.sum()), int(b.sum())
    result = dict(observed_species=len(unique), discovery_species=int((a>0).sum()),
                  discovery_rows=na, test_rows=nb,
                  unseen_test_mass=float(b[a==0].sum()/max(1,nb)))
    order = np.argsort(-a, kind="stable")
    result["top_k_test_mass"] = {
        str(k): float(b[order[:min(k,int((a>0).sum()))]].sum()/max(1,nb))
        for k in (1,16,64,256,1024,4096)
    }
    if unique.dtype.names is not None:
        group = unique["group"]
        order = np.lexsort((-a, group))
        gs = group[order]
        starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1])+1]
        rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts,len(order)]))
        result["observed_groups"] = len(starts)
        result["top_k_per_group_test_mass"] = {
            str(k): float(b[order[(rank<k)&(a[order]>0)]].sum()/max(1,nb))
            for k in (1,4,16)
        }
    return result


def input_groups(x, steps):
    M = np.full(len(x), -np.inf)
    S = M.copy()
    history = np.zeros(len(x), np.uint32)
    result = {}
    for t in range(10):
        nm, ns = x[:,t] > M, x[:,t] > S
        event = np.where(nm,2,np.where(ns,1,0))
        history = 3*history + event.astype(np.uint32)
        S, M = np.where(nm,M,np.where(ns,x[:,t],S)), np.maximum(M,x[:,t])
        if t+1 in steps:
            top = np.argsort(x[:,:t+1],axis=1)[:,-2:]
            result[t+1] = ((t-top[:,1])*(t+1)+(t-top[:,0]), history.copy())
    return result


def load_models(root, count):
    p = np.load(root/"pop1k/seed0.npz")
    w = [p["s1048576_"+k].astype(np.float64) for k in ("Wih","Whh","Wo")]
    indices = np.arange(0,64,max(1,64//count))[:count]*16
    for i in indices:
        yield "pop"+str(i), [v[i] for v in w]
    for name, file in [("mystery","weights.npz")]+[(f"seed{i}",f"weights_seed{i}.npz") for i in range(5)]:
        p = np.load(root/file)
        yield name, [p["rnn.weight_ih_l0"][:,0].astype(float),
                     p["rnn.weight_hh_l0"].astype(float), p["linear.weight"].astype(float)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--dev-count",type=int,default=8,choices=(8,16,32,64))
    ap.add_argument("--log2-inputs",type=int,default=16)
    ap.add_argument("--seed",type=int,default=2026101906)
    ap.add_argument("--out",type=Path,default=Path("r6_geometry.json"))
    args = ap.parse_args()
    n = 2**args.log2_inputs
    x = np.random.default_rng(args.seed).standard_normal((n,10))
    top = np.argsort(x,axis=1)[:,-2:]
    target, maxpos = top[:,0], top[:,1]
    steps = (3,5,7,9,10)
    groups = input_groups(x,steps)
    result = dict(description=__doc__, seed=args.seed, inputs=n, discovery_inputs=n//2,
                  dev_count=args.dev_count, measurement_forwards_per_network=n, networks=[])
    for name,(wi,wh,wo) in load_models(args.root,args.dev_count):
        h = np.zeros((n,16))
        gates = np.empty((n,10),np.uint16)
        bit = (1 << np.arange(16)).astype(np.uint16)
        for t in range(10):
            h = h@wh.T+x[:,t,None]*wi
            gates[:,t] = ((h>0)*bit).sum(1,dtype=np.uint16)
            np.maximum(h,0,out=h)
        pred = (h@wo.T).argmax(1)
        error = pred != target
        row = dict(name=name, accuracy=float(1-error.mean()), prefixes={})
        for t in steps:
            age, event = groups[t]
            row["prefixes"][str(t)] = dict(
                all=species(keys(gates[:,:t]),n//2),
                by_ages=species(keys(gates[:,:t],age),n//2),
                by_events=species(keys(gates[:,:t],event),n//2))
        row["error_patterns"] = species(keys(gates),n//2,error)
        fragment = ((maxpos*10+target)*10+pred).astype(np.uint32)
        row["error_cone_fragments"] = species(keys(gates,fragment),n//2,error)
        result["networks"].append(row)
        args.out.write_text(json.dumps(result,indent=2))
        end = row["prefixes"]["10"]["all"]
        print(name, "prefix_species", end["observed_species"], "unseen", round(end["unseen_test_mass"],4),
              "error_fragments", row["error_cone_fragments"]["observed_species"], flush=True)


if __name__ == "__main__":
    main()
