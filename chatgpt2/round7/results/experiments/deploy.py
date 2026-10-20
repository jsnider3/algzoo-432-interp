"""Compiled inference for the Round 7 estimators (NumPy/SciPy only).

Offline: select candidate hyperparameters on training networks, then call
compile_readout. Online: fixed_prefix_logits and predict use target weights
or features only. This implements the arithmetic assumed by the FE ledger.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from cost_experiments import ALP, GAM, trie_plan

def fixed_prefix_logits(wi, wh, wo, plan):
    """One network; plan=trie_plan(fixed_rows) is prepared offline.

    Input products are computed once per distinct input value. Recurrent
    products are computed once per unique parent state. No sampled inputs.
    """
    values, levels, leaves = plan
    inp = values[:,None]*np.asarray(wi).reshape(1,-1)
    h = None
    for parent, token in levels:
        h = np.maximum(inp[token],0) if h is None else np.maximum((h@wh.T)[parent]+inp[token],0)
    return (h@wo.T)[leaves]

def cell_outputs(wi, wh, wo, plan, target):
    lg = fixed_prefix_logits(wi,wh,wo,plan)
    correct = lg.argmax(1) == target  # NumPy chooses the lowest tied index.
    chosen = lg[np.arange(len(lg)),target].copy()
    lg[np.arange(len(lg)),target] = -np.inf
    return correct, chosen-lg.max(1)

def compile_readout(x, y, choice, supports=0):
    """Offline fit; choice is column number in cost_experiments.predictions.

    Exact RBF uses every training support; supports>0 uses a fixed,
    training-only inducing subset and a Nystrom kernel approximation.
    No validation or target rows enter this function.
    """
    mu, sd = x.mean(0), np.maximum(x.std(0),1e-6)
    a = (x-mu)/sd
    ym = y.mean(); yc=y-ym
    family, penalty = divmod(int(choice),len(ALP)); alpha=ALP[penalty]
    if family == 0:
        e,v=eigh(a.T@a)
        beta=v@((v.T@(a.T@yc))/(np.maximum(e,0)+len(y)*alpha))
        coef=beta/sd
        return dict(family='linear',coef=coef,intercept=ym-mu@coef)
    gamma=GAM[family-1]/x.shape[1]
    centers=a
    if supports:
        idx=np.random.default_rng(1701).permutation(len(y))[:min(supports,len(y))]
        centers=a[idx]
        km=np.exp(-gamma*cdist(centers,centers,'sqeuclidean'))
        e,v=eigh(km); keep=e>max(1e-12,e[-1]*1e-12)
        whitening=v[:,keep]/np.sqrt(e[keep])
        phi=np.exp(-gamma*cdist(a,centers,'sqeuclidean'))@whitening
        e,v=eigh(phi.T@phi)
        coef=whitening@v@((v.T@(phi.T@yc))/(np.maximum(e,0)+alpha))
    else:
        e,v=eigh(np.exp(-gamma*cdist(a,a,'sqeuclidean')))
        coef=v@((v.T@yc)/(np.maximum(e,0)+alpha))
    return dict(family='rbf',mu=mu,sd=sd,centers=centers,
                coef=coef,intercept=ym,gamma=gamma)

def predict(model, x):
    """Online inference; one selected compiled readout, no fitting/eigensystem."""
    if model['family']=='linear':
        return np.asarray(x)@model['coef']+model['intercept']
    a=np.atleast_2d((x-model['mu'])/model['sd'])
    distance=cdist(a,model['centers'],'sqeuclidean')
    return model['intercept']+np.exp(-model['gamma']*distance)@model['coef']
