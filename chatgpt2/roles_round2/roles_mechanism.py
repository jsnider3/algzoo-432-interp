#!/usr/bin/env python3
"""Mechanistic role diagnostics and population-calibrated regional predictions.

Train: python roles_mechanism.py --root ALGZOO --population SNAP.npz --out RUN
Predict: python roles_mechanism.py --predict weights.npz --calibrator RUN/calibrated_hybrid.npz --out PRED
Requires NumPy; fitting also requires SciPy and scikit-learn. All source is here.
No random target labels enter prediction. Calibration is empirical, not a universal guarantee.
"""
import argparse,hashlib,json,time
from pathlib import Path
from functools import lru_cache
import numpy as np

REGION_NAMES=['S_negative','MS_tie','SR_tie','middle','clean']
REGIMES=('clean_sparse','clean_crowded','middle','MS_tie','SR_tie','negative_S')

def load_population(path, snapshot=None):
    with np.load(path) as z:
        if snapshot is None:
            steps = [int(k[1:-4]) for k in z.files
                     if k.startswith('s') and k.endswith('_Wih')]
            snapshot = 's' + str(max(steps))
        elif not str(snapshot).startswith('s'):
            snapshot = 's' + str(snapshot)
        return tuple(np.ascontiguousarray(z[str(snapshot)+'_'+k], dtype=np.float32)
                     for k in ('Wih', 'Whh', 'Wo'))

def load_one(path):
    with np.load(path) as z:
        return tuple(np.ascontiguousarray(z[k].reshape(shape), dtype=np.float32)
                     for k, shape in [('rnn.weight_ih_l0', (16,)),
                                      ('rnn.weight_hh_l0', (16, 16)),
                                      ('linear.weight', (10, 16))])

def load_zoo(root):
    root = Path(root)
    names = ['mystery'] + ['seed'+str(i) for i in range(5)]
    paths = [root/'weights.npz'] + [root/f'weights_seed{i}.npz' for i in range(5)]
    models = [load_one(p) for p in paths]
    return names, tuple(np.stack([p[j] for p in models]) for j in range(3))

def forward(P, x, trace=False):
    """Return logits (N,B,10), or logits, h, z with h/z (N,B,T,H).

    P can contain a single model (H,), (H,H), (10,H), in which case
    the returned tensors still have a leading model axis of length one.
    This function is intentionally unchunked; the caller controls batch size.
    """
    wi, wh, wo = (np.asarray(w) for w in P)
    if wi.ndim == 1:
        wi, wh, wo = wi[None], wh[None], wo[None]
    x = np.asarray(x, dtype=wi.dtype)
    h = np.zeros((len(wi), wi.shape[1], len(x)), dtype=wi.dtype)
    hs, zs = [], []
    for t in range(x.shape[1]):
        z = wh @ h + wi[:, :, None]*x[None, None, :, t]
        h = np.maximum(z, 0)
        if trace:
            hs.append(h.transpose(0, 2, 1))
            zs.append(z.transpose(0, 2, 1))
    logits = (wo @ h).transpose(0, 2, 1)
    if trace:
        return logits, np.stack(hs, 2), np.stack(zs, 2)
    return logits

def target(x):
    return np.argsort(x, axis=1)[:, -2]

def task_strata(x):
    """Exact z1.task_strata bins: four negative-S plus seven by eight gaps."""
    s = -np.sort(-np.asarray(x), axis=1)
    M, S, R = s[:, 0], s[:, 1], s[:, 2]
    den = np.maximum(np.abs(M), np.finfo(s.dtype).tiny)
    a, b = (M-S)/den, (S-R)/den
    ab = np.digitize(a, [.01,.02,.05,.1,.2,.4])*8
    ab += np.digitize(b, [.005,.01,.03,.06,.1,.2,.4])
    return np.where(S < 0, np.digitize(S, [-1,-.5,-.2]), 4+ab)

def regions_from_strata(labels):
    lab = np.asarray(labels)
    a, b = (lab-4)//8, (lab-4)%8
    region = np.full(lab.shape, 3, dtype=np.int64)
    region[lab < 4] = 0
    region[(lab >= 4) & (a <= 2)] = 1
    region[(lab >= 4) & (a > 2) & (b <= 2)] = 2
    region[(lab >= 4) & (a >= 4) & (b >= 5)] = 4
    return region

def region_contributions(E, n):
    E, n = np.asarray(E), np.asarray(n)
    lab = regions_from_strata(np.arange(len(n)))
    total = n.sum()
    contribution = np.stack([E[..., lab == j].sum(-1)/total for j in range(5)], -1)
    mass = np.array([n[lab == j].sum()/total for j in range(5)])
    return contribution, mass

def baseline_features(wi, wh, wo):
    return np.concatenate([np.sort(wi, 1), np.sort(np.diagonal(wh, axis1=1, axis2=2), 1),
                           np.sort(abs(np.linalg.eigvals(wh)), 1),
                           np.sort(np.linalg.norm(wo, axis=1), 1)], 1)

def probe_bank(banks=1):
    """60 probes/bank: six value regimes, ten age rotations per regime.

    The max/second displacement alternates across value regimes, avoiding an
    exact alias between their ages. The optional second bank reverses this
    displacement and changes nuisance background. No population labels enter.
    """
    xs, group = [], []
    for bank in range(banks):
        for r in range(6):
            for im in range(10):
                isec, ir = (im+(3 if (bank+r) % 2 == 0 else 6)) % 10, (im+5) % 10
                M, S, R = [(1,.6,.2),(1,.6,.45),(1,.925,.85),
                           (1,.975,.65),(1,.7,.685),(.3,-.2,-.45)][r]
                background = np.array([-.87,-.54,-.28,.04,-.69,-.13,.11])
                if r == 1:
                    background = np.array([.14,.29,.36,.18,.39,.22,.32])
                elif r == 5:
                    background = np.array([-1.31,-.81,-.64,-.57,-1.09,-.73,-.92])
                if bank % 2:
                    background = np.roll(background[::-1], im % 7)*.91
                x = np.empty(10)
                rest = [k for k in range(10) if k not in (im,isec,ir)]
                x[rest] = background
                x[im],x[isec],x[ir] = M,S,R
                xs.append(x); group.append(r)
    x = np.asarray(xs,np.float32)
    order = np.argsort(x,axis=1)
    return x,np.asarray(group),order[:,-1],order[:,-2],order[:,-3]

@lru_cache(maxsize=8)
def _setup(banks):
    """Input-only setup. No target data or fitted population quantities."""
    x,groups,im,ise,ir=probe_bank(banks)
    b=len(x);rows=np.arange(b)
    lower=np.ones((b,10),bool)
    lower[rows,im]=lower[rows,ise]=lower[rows,ir]=False
    pos=np.stack([ise,ir],1)
    gaps=np.stack([x[rows,im]-x[rows,ise],x[rows,ise]-x[rows,ir]],1)
    regidx=np.stack([np.flatnonzero(groups==r) for r in range(6)])
    age_masks=[im<ise,im>ise,ise==9,im==9,ir==9]
    prevM=np.zeros((b,8));prevS=np.zeros((b,8))
    for k,t in enumerate(range(2,10)):
        p=np.sort(x[:,:t],axis=1);prevM[:,k],prevS[:,k]=p[:,-1],p[:,-2]
    now=x[:,2:]
    event=np.where(now>prevM,0,np.where(now>prevS,1,2))
    events=np.stack([event==k for k in range(3)],2)
    late=events[:,:-1,:]
    design=np.stack([now,prevM,prevS],2).reshape(-1,3)
    pinv=np.linalg.pinv(design)
    ageidx=[np.stack([np.flatnonzero(labels==age) for age in range(10)])
            for labels in [9-im,9-ise]]
    kinds=['margin','wrong_pair','sensitivity_M','sensitivity_S','sensitivity_R',
           'lower_gradient_share','lower_margin_contribution']
    names=[]
    for label in REGIMES:
        for kind in kinds:
            for cn in ('versus_M','versus_R','versus_lower'):
                names.append(f'{label}/{cn}/{kind}_mean')
                if kind=='margin':names.append(f'{label}/{cn}/{kind}_p10')
        for kind in ('gate_valid_gap_fraction','tie_extrapolated_margin',
                     'first_gate_margin','within_gate_pair_failure'):
            for cn in ('MS','SR'):names.append(f'{label}/{cn}/{kind}')
    for label in ('direct_second','max_handoff','S_last','M_last','R_last'):
        for cn in ('versus_M','versus_R','versus_lower'):
            names.extend([f'{label}/{cn}/margin',f'{label}/{cn}/lower_gradient_share'])
    events=(events.reshape(-1,3).astype(np.float32),events.sum((0,1)))
    late=(late.reshape(-1,3).astype(np.float32),late.sum((0,1)))
    return (x,groups,im,ise,ir,rows,lower,pos,gaps,regidx,age_masks,
            events,late,design,pinv,ageidx,names)

def _eta_indexed(v,index):
    mean=v.mean(1)
    groupmean=v[:,index,:].mean(2)
    between=((groupmean-mean[:,None,:])**2).mean(1)
    return between/np.maximum(v.var(1),1e-12)

def _features_chunk(P, banks):
    wi,wh,wo=(np.asarray(a,np.float32) for a in P)
    (x,groups,im,ise,ir,rows,lower,pos,gaps,regidx,age_masks,
     events,late,design,pinv,ageidx,names)=_setup(banks)
    n,b,h=len(wi),len(x),wi.shape[1]
    logits,hs,zs=forward((wi,wh,wo),x,trace=True)
    active=hs>0
    lm=np.where(lower[None],logits,-np.inf).argmax(2)
    competitor=np.stack([np.broadcast_to(im,(n,b)),np.broadcast_to(ir,(n,b)),lm],2)
    m=logits[:,rows,ise][:,:,None]-np.take_along_axis(logits,competitor,axis=2)
    q=wo[:,ise,None,:]-wo[np.arange(n)[:,None,None],competitor]
    grad=np.empty((n,b,3,10),np.float32)
    # Exact local reverse differentiation through actual gates.
    for t in range(9,-1,-1):
        qz=q*active[:,:,t,None,:]
        grad[:,:,:,t]=(qz*wi[:,None,None,:]).sum(-1)
        q=(qz.reshape(n,b*3,h)@wh).reshape(n,b,3,h)
    normg=np.maximum(np.linalg.norm(grad,axis=3),1e-10)
    normx=np.linalg.norm(x,axis=1)
    scale=normg*normx[None,:,None]
    margin=m/scale
    gm=np.take_along_axis(grad,np.broadcast_to(im[None,:,None,None],(n,b,3,1)),3)[...,0]/normg
    gs=np.take_along_axis(grad,np.broadcast_to(ise[None,:,None,None],(n,b,3,1)),3)[...,0]/normg
    gr=np.take_along_axis(grad,np.broadcast_to(ir[None,:,None,None],(n,b,3,1)),3)[...,0]/normg
    leak=np.linalg.norm(grad*lower[None,:,None,:],axis=3)/normg
    nuisance=(grad*x[None,:,None,:]*lower[None,:,None,:]).sum(3)/scale
    # The Euler identity is exact for this bias-free piecewise-linear network.
    euler=np.max(np.abs(m-(grad*x[None,:,None,:]).sum(3)))
    # Valid one-sided step before ANY recurrent ReLU gate changes. Directions
    # raise S towards M and raise R towards S. These do not change the task
    # label until the respective observed rank gap is exhausted.
    dh=np.zeros((n,b,h,2),np.float32)
    radius=np.full((n,b,2),np.inf,np.float32)
    step=np.empty((n,b,h,2),np.float32)
    for t in range(10):
        dz=(wh@dh.transpose(0,2,1,3).reshape(n,h,b*2)).reshape(n,h,b,2).transpose(0,2,1,3)
        dz += wi[:,None,:,None]*(pos==t)[None,:,None,:]
        z=zs[:,:,t,:,None]
        gate=active[:,:,t,:,None]
        cross=(gate&(dz<0))|((~gate)&(dz>0))
        step.fill(np.inf)
        np.divide(-z,dz,out=step,where=cross)
        radius=np.minimum(radius,np.min(step,axis=2))
        dh=dz*gate
    valid=np.minimum(radius/gaps[None],1.)
    pair_grad=np.stack([gs[:,:,0]*normg[:,:,0],gr[:,:,1]*normg[:,:,1]],2)
    pair_margin=m[:,:,:2]
    tie_margin=(pair_margin+pair_grad*gaps[None])/scale[:,:,:2]
    safe_margin=(pair_margin+pair_grad*np.minimum(radius,gaps[None]))/scale[:,:,:2]
    boundary=np.divide(-pair_margin,pair_grad,out=np.full_like(pair_margin,np.inf),
                       where=abs(pair_grad)>1e-10)
    certified_pair_cross=(pair_margin>0)&(pair_grad<0)&(boundary>=0)&(boundary<=radius)&(boundary<gaps[None])
    arrays=[margin,(m<0).astype(float),gm,gs,gr,leak,nuisance]
    kinds=['margin','wrong_pair','sensitivity_M','sensitivity_S','sensitivity_R','lower_gradient_share','lower_margin_contribution']
    descriptions=np.empty((n,6,32),dtype=float)
    selected=margin[:,regidx,:]
    descriptions[:,:,0:6:2]=selected.mean(2)
    # Exact default linear-quantile definition, with the constant rank cached
    # by the probe-bank size; avoids general quantile dispatch for tiny groups.
    rank=.1*(selected.shape[2]-1);lo=int(rank);fraction=rank-lo
    selected.partition((lo,lo+1),axis=2)
    low,high=selected[:,:,lo,:],selected[:,:,lo+1,:]
    diff=(high-low).astype(float)
    p10=(high.astype(float)-diff*(1-fraction) if fraction>=.5 else low.astype(float)+diff*fraction)
    p10=np.where(np.isnan(selected[:,:,-1,:]),np.nan,p10)
    descriptions[:,:,1:6:2]=p10
    col=6
    for value in arrays[1:]:
        descriptions[:,:,col:col+3]=value[:,regidx,:].mean(2);col+=3
    for value in (valid,tie_margin,safe_margin,certified_pair_cross):
        descriptions[:,:,col:col+2]=value[:,regidx,:].mean(2);col+=2
    age_description=np.empty((n,5,6),dtype=float)
    for k,mask in enumerate(age_masks):
        age_description[:,k,0::2]=margin[:,mask,:].mean(1)
        age_description[:,k,1::2]=leak[:,mask,:].mean(1)
    X=np.c_[descriptions.reshape(n,-1),age_description.reshape(n,-1)]
    # A cheap neuron description retains signed event behavior and the phase
    # of the actual readout. The buckets are reporting conveniences; F also
    # retains continuous, overlapping parameters.
    on=(active[:,:,2:,:].transpose(0,3,1,2).reshape(n,h,-1).astype(np.float32)@
        events[0]).astype(float)/events[1]
    after=(active[:,:,3:,:].transpose(0,3,1,2).reshape(n,h,-1).astype(np.float32)@
           late[0]).astype(float)/late[1]
    oc=wo-wo.mean(1,keepdims=True); c=np.linalg.norm(oc,axis=1).clip(1e-10)
    canon_in=wi*c; canon_in/=np.sqrt(np.mean(canon_in**2,1,keepdims=True)).clip(1e-10)
    A=wh*c[:,:,None]/c[:,None,:]
    f=[canon_in,np.diagonal(wh,axis1=1,axis2=2),np.log1p(np.linalg.norm(A,axis=2)),
       np.log1p(np.linalg.norm(A,axis=1)),active.mean((1,2))]
    fn=['canonical_input','self_loop','log_recurrent_in','log_recurrent_out','active_rate']
    for j,lab in enumerate(('newmax','newsecond','neither')):
        f.extend([on[:,:,j],after[:,:,j]]); fn.extend(['on_'+lab,'after_'+lab])
    yy=zs[:,:,2:,:].reshape(n,-1,h)
    coef=pinv@yy
    fitted=design@coef
    r2=1-((yy-fitted)**2).sum(1)/np.maximum(((yy-yy.mean(1,keepdims=True))**2).sum(1),1e-10)
    coef/=np.maximum(np.linalg.norm(coef,axis=1,keepdims=True),1e-10)
    for j,lab in enumerate(('x','M','S')):
        f.append(coef[:,j,:]); fn.append('threshold_coefficient_'+lab)
    f.append(r2);fn.append('threshold_fit_R2')
    final=hs[:,:,-1,:]/normx[None,:,None]
    f.extend([_eta_indexed(final,ageidx[0]),_eta_indexed(final,ageidx[1])]);fn.extend(['age_M_eta2','age_S_eta2'])
    # Fourier real/imaginary components retain phase, unlike magnitude alone.
    spec=np.fft.rfft(oc/c[:,None,:],axis=1)
    for k in range(1,6):
        f.append(spec[:,k,:].real);fn.append(f'readout_cos{k}')
        if k<5:f.append(spec[:,k,:].imag);fn.append(f'readout_sin{k}')
    F=np.stack(f,2)
    role=np.full((n,h),7,dtype=int)
    role[(wi<0)&(on[:,:,2]>=.7)&(on[:,:,0]<=.4)]=6
    role[(after[:,:,0]>=.7)&(after[:,:,1]<.3)&(after[:,:,2]<=.1)]=5
    role[(on[:,:,0]>=.7)&(on[:,:,1]<.5)&(on[:,:,2]<=.1)]=4
    role[(after[:,:,0]>=.7)&(after[:,:,1]>=.5)&(after[:,:,2]<=.3)]=3
    role[(on[:,:,0]>=.7)&(on[:,:,1]>=.5)&(on[:,:,2]<=.3)]=2
    role[(after[:,:,2]>=.95)&(on[:,:,0]>=.9)]=1
    role[active.mean((1,2))<.02]=0
    masks=role[:,:,None]==np.arange(8)
    counts=masks.sum(1)
    means=(masks.transpose(0,2,1).astype(float)@F)/np.maximum(counts[:,:,None],1)
    bag=np.concatenate([counts[:,:,None],means],2).reshape(n,-1)
    bn=[]
    for k in range(8):
        bn.append(f'role{k}/count')
        bn.extend([f'role{k}/{name}' for name in fn])
    detailed=np.concatenate([v.reshape(n,-1) for v in [margin,leak,nuisance,tie_margin,valid]],1)
    return X,list(names),bag,bn,F,fn,dict(role=role,margin=margin,leak=leak,nuisance=nuisance,
        gate_valid=valid,certified_pair_cross=certified_pair_cross,euler_abs_error=float(euler),
        detailed=detailed)

def extract_designed_features(P,banks=1,chunk=64):
    P=tuple(np.asarray(p) for p in P)
    if P[0].ndim==1:P=tuple(p[None] for p in P)
    ans=[]
    for start in range(0,len(P[0]),chunk):
        ans.append(_features_chunk(tuple(p[start:start+chunk] for p in P),banks))
    X=np.concatenate([v[0] for v in ans]);B=np.concatenate([v[2] for v in ans]);F=np.concatenate([v[4] for v in ans])
    diag={k:(max(v[6][k] for v in ans) if k=='euler_abs_error' else np.concatenate([v[6][k] for v in ans])) for k in ans[0][6]}
    return X,ans[0][1],B,ans[0][3],F,ans[0][5],diag

def prefix_bank():
    """48 prefixes: max ages0..7, second at cyclic offsets-1,+1,+4, S=.45/.8."""
    xs, ms, ss, ages = [], [], [], []
    for age in range(8):
        im = 7-age
        for shift in (-1,1,4):
            for s in (.45,.8):
                x = np.array([-.7,-.2,-.5,-.1,-.6,-.3,-.8,-.4],np.float32)
                x[im] = 1.; x[(im+shift)%8] = s
                xs.append(x); ms.append(im); ss.append(s); ages.append(age)
    return np.array(xs), np.array(ms), np.array(ss), np.array(ages)

def hidden(wi,wh,x):
    h = np.zeros((len(wi),wi.shape[1],len(x)),np.float32)
    for t in range(x.shape[1]):
        h = np.maximum(wh@h+wi[:,:,None]*x[None,None,:,t],0)
    return h

def coverage_features(weights, role_labels=None, negative_cutoff=-2., canonical_route=False):
    wi,wh,wo = (tuple(np.asarray(a,dtype=np.float64) for a in weights)
                if canonical_route else weights)
    xp,im,s,ages = prefix_bank()
    h = hidden(wi,wh,xp)
    base = wh@h
    # N, B, H. For negative wi: ReLU(|wi|*(theta-x)).
    theta = np.divide(-base,wi[:,:,None],out=np.zeros_like(base),
                      where=wi[:,:,None]!=0).transpose(0,2,1)
    negative = wi < negative_cutoff
    chosen = negative if role_labels is None else role_labels == 6
    gap = 1-s
    rel = (theta-s[None,:,None])/gap[None,:,None]
    distance = np.where(chosen[:,None,:],np.minimum(np.abs(rel),10),10)
    nearest = distance.min(2)
    cols = [chosen.sum(1),negative.sum(1)]
    names = ['comparator_count','negative_input_count']
    for a in range(8):
        cols += [nearest[:,ages==a].mean(1),nearest[:,ages==a].max(1)]
        names += [f'S_threshold_distance_age{a}_mean',f'S_threshold_distance_age{a}_max']
    for tol in (.1,.25,.5,1.):
        cols.append((nearest <= tol).mean(1)); names.append(f'S_threshold_coverage_{tol}')
    cols += [nearest.mean(1), np.quantile(nearest,.9,axis=1)]
    names += ['S_threshold_distance_mean','S_threshold_distance_p90']
    # Current input: definitely neither, a new second, or a new maximum.
    test_x = np.stack([s-.25*gap,s+.25*gap,1+.25*gap],0)
    h_event = np.maximum(base[:,None,:,:]+wi[:,None,:,None]*test_x[None,:,None,:],0)
    pre_next = np.einsum('nij,nejb->neib',wh,h_event)+wi[:,None,:,None]*(-.35)
    active_on = h_event>0; active_late = pre_next>0
    def sel(active):
        rates=active.mean(-1)
        return np.minimum(rates[:,1],rates[:,2])-rates[:,0]
    on_sel,late_sel=sel(active_on),sel(active_late)
    # Output-scaled response amplitude resolves selectivity ties; average exact ties.
    out_scale=np.linalg.norm(wo,axis=1)
    amplitude=out_scale*np.std(pre_next,axis=(1,3))
    eligible=late_sel==late_sel.max(1,keepdims=True)
    chosen_amplitude=np.where(eligible,amplitude,-np.inf)
    tied=eligible&(chosen_amplitude==chosen_amplitude.max(1,keepdims=True))
    mixture=tied/tied.sum(1,keepdims=True)
    gate_mix=mixture*out_scale if canonical_route else mixture
    gate_row=np.einsum('np,npj->nj',gate_mix,wh)
    gate_input=np.sum(gate_mix*wi,axis=1)
    best_late=np.argmax(mixture,axis=1)
    cols += [on_sel.max(1),late_sel.max(1)]
    names += ['designed_event_selectivity','designed_late_selectivity']
    # Which negative-input comparator signals actually inhibit that late gate?
    route=np.maximum(-gate_row,0)*np.maximum(-wi,0)*negative
    strength=route.sum(1)
    normalized=route/np.where(strength[:,None]>0,strength[:,None],1)
    wc=np.einsum('nh,nbh->nb',normalized,np.exp(-rel*rel))
    cols += [strength,wc.mean(1),wc.min(1)]
    names += ['late_pointer_inhibitory_route_strength','route_weighted_S_coverage_mean','route_weighted_S_coverage_min']
    # Exact roots of the two-step gate spline. All its hinges are theta_j.
    roots=[]
    for start in range(0,len(wi),64):
        end=min(start+64,len(wi)); nn=end-start
        low=s-gap; high=1+gap
        knots=np.sort(np.concatenate([
            np.broadcast_to(low[None,:,None],(nn,len(s),1)),
            np.clip(theta[start:end],low[None,:,None],high[None,:,None]),
            np.broadcast_to(high[None,:,None],(nn,len(s),1))],2),2)
        z=base[start:end].transpose(0,2,1)[:,:,None,:]+wi[start:end,None,None,:]*knots[:,:,:,None]
        hh=np.maximum(z,0)
        pp=np.einsum('nbkh,nh->nbk',hh,gate_row[start:end])+gate_input[start:end,None,None]*(-.35)
        opening=(pp[:,:,:-1]<=0)&(pp[:,:,1:]>0)
        denom=pp[:,:,1:]-pp[:,:,:-1]
        root=knots[:,:,:-1]-pp[:,:,:-1]*np.diff(knots,axis=2)/np.where(denom!=0,denom,1)
        dist=np.abs(root-s[None,:,None])/gap[None,:,None]
        roots.append(np.min(np.where(opening,np.minimum(dist,10),10),axis=2))
    gd=np.concatenate(roots)
    cols += [gd.mean(1),(gd<=.25).mean(1),(gd<=.5).mean(1)]
    names += ['late_gate_S_distance_mean','late_gate_S_coverage_025','late_gate_S_coverage_05']
    # Final readout after either event then one negative settle step. Labels are
    # event position8 for new second and old maximum position for new maximum.
    zz=np.einsum('noh,nehb->nebo',wo,np.maximum(pre_next,0))
    target=np.stack([np.argsort(xp,axis=1)[:,-2],np.full(len(s),8),im],0)
    e=zz.argmax(-1)!=target[None]
    cols += [e[:,1].mean(1),e[:,2].mean(1),e[:,0].mean(1)]
    names += ['designed_direct_second_error','designed_max_handoff_error','designed_no_event_error']
    return np.stack(cols,1).astype(float),names,dict(theta=theta,nearest=nearest,
           gate_distance=gd,best_late=best_late,late_selectivity=late_sel.max(1),
           chosen=chosen,designed_errors=e)

def handoff_bank():
    """24 clean matched new-second/new-max pairs, event at3,5,7.

    M=1,S=.45. x=.725 writes event; x=1.275 promotes old max. No later updates.
    """
    pairs=[]
    for t in (3,5,7):
        for old_age in (0,1):
            im=t-1-old_age
            for so in (1,2):
                si=(im-so)%t
                for bg in (0,1):
                    x=np.resize(np.array([-.7,-.2,-.5,-.1,-.6,-.3,-.8,-.4],np.float32),10)
                    if bg: x=-.2-np.arange(10,dtype=np.float32)*.03
                    x[im]=1; x[si]=.45
                    pairs.append((x,t,im))
    return pairs

def paired_swaps(weights,labels,contrast='handoff',intervention_delay=0):
    """Swap one role group's counterfactual state; continue the actual recurrence."""
    wi,wh,wo=weights; n=len(wi)
    groups={'comparator':labels==6,'max_markers':(labels==4)|(labels==5),
      'pointers':(labels==2)|(labels==3),'remaining':~np.isin(labels,[2,3,4,5,6])}
    res={k:{'second_to_max':[],'max_to_second':[]} for k in ['baseline']+list(groups)}
    for xp,t,im in handoff_bank():
        other_index=im if contrast=='handoff' else np.argsort(xp[:t])[-2]
        other_value=1.275 if contrast=='handoff' else .2
        h0=hidden(wi,wh,xp[None,:t])[:,:,0]
        hs=np.maximum(np.einsum('nij,nj->ni',wh,h0)+wi*.725,0)
        hm=np.maximum(np.einsum('nij,nj->ni',wh,h0)+wi*other_value,0)
        for u in range(t+1,t+intervention_delay+1):
            hs=np.maximum(np.einsum('nij,nj->ni',wh,hs)+wi*xp[u],0)
            hm=np.maximum(np.einsum('nij,nj->ni',wh,hm)+wi*xp[u],0)
        def finish(h):
            for u in range(t+intervention_delay+1,10):
                h=np.maximum(np.einsum('nij,nj->ni',wh,h)+wi*xp[u],0)
            z=np.einsum('noh,nh->no',wo,h)
            # margin favors event over old-max, useful for either branch.
            return z.argmax(1),z[:,t]-z[:,other_index]
        bs,ms=finish(hs.copy()); bm,mm=finish(hm.copy())
        res['baseline']['second_to_max'].append((bs==other_index,bs==t,ms))
        res['baseline']['max_to_second'].append((bm==t,bm==other_index,mm))
        for gn,mask in groups.items():
            cs,cm=finish(np.where(mask,hm,hs)); ds,dm=finish(np.where(mask,hs,hm))
            res[gn]['second_to_max'].append((cs==other_index,cs==t,cm))
            res[gn]['max_to_second'].append((ds==t,ds==other_index,dm))
    out={}
    for gn,data in res.items():
        out[gn]={}
        for branch,rows in data.items():
            wrong=np.stack([a for a,b,c in rows],1)
            correct=np.stack([b for a,b,c in rows],1)
            margin=np.stack([c for a,b,c in rows],1)
            basem=np.stack([c for a,b,c in res['baseline'][branch]],1)
            basec=np.stack([b for a,b,c in res['baseline'][branch]],1)
            out[gn][branch]=dict(correct_fraction=float(correct.mean()),
              opposite_event_answer_fraction=float(wrong.mean()),
              median_network_correct_fraction=float(np.median(correct.mean(1))),
              mean_event_minus_old_max_margin=float(margin.mean()),
              mean_margin_change=float((margin-basem).mean()),
              correct_to_opposite_fraction=float((wrong&basec).sum()/max(basec.sum(),1)))
            out[gn][branch]['subgroups']={}
            for name,ix in [('late_pointer',np.any(labels==3,axis=1)),
                ('event_pointer',np.any(labels==2,axis=1)),
                ('no_pointer',~np.any((labels==2)|(labels==3),axis=1))]:
                out[gn][branch]['subgroups'][name]={'networks':int(ix.sum()),
                  'correct_fraction':float(correct[ix].mean()),
                  'correct_to_opposite_fraction':float((wrong[ix]&basec[ix]).sum()/max(basec[ix].sum(),1))}
    if contrast!='handoff':
        out={g:{('second_to_neither' if k=='second_to_max' else 'neither_to_second'):v for k,v in d.items()} for g,d in out.items()}
    return out

def threshold_history_fits(weights, roles):
    """Move S at fixed M,S,max age; measure error of theta=a_age*M+b_age*S."""
    wi,wh,wo=weights; selected=roles==6; result={}
    for constant in (False,True):
        x,_,_,_=prefix_bank()
        if constant: x[x<0]=-.4
        h=hidden(wi,wh,x)
        theta=np.divide(-(wh@h),wi[:,:,None],out=np.zeros_like(h),where=wi[:,:,None]!=0)
        v=theta.transpose(0,2,1).reshape(len(wi),8,3,2,wi.shape[1])
        mean=v.mean(2); scale=1-np.array([.45,.8])
        span=np.ptp(v,axis=2)/scale[None,None,:,None]
        spans=span.transpose(0,3,1,2)[selected].ravel()
        residual=(v-mean[:,:,None,:,:])/scale[None,None,None,:,None]
        rms=np.sqrt(np.mean(residual*residual,axis=(1,2,3)))[selected]
        b=(mean[:,:,1]-mean[:,:,0])/.35; a=mean[:,:,0]-.45*b
        ab=(a+b-1).transpose(0,2,1)[selected]
        result['constant_background' if constant else 'varied_background']={
          'RMS_threshold_residual_over_gap_p5_p50_p95':np.percentile(rms,[5,50,95]).tolist(),
          'threshold_span_over_gap_p5_p50_p95':np.percentile(spans,[5,50,95]).tolist(),
          'fraction_span_over025gap':float((spans>.25).mean()),
          'median_abs_a_plus_b_minus1':float(np.median(np.abs(ab)))}
    return result

def coverage_confusion(diag):
    """Frequency checks, not probabilistic bounds; each prefix/network counted once."""
    _,_,s,age=prefix_bank(); result={}
    subsets={'all':np.ones(len(s),bool),'age_le6':age<=6,'age7':age==7,
      'clean_S045':s==.45,'harder_S080':s==.8,
      'clean_S045_age_le6':(s==.45)&(age<=6),'clean_S045_age7':(s==.45)&(age==7)}
    for subset,mask in subsets.items():
        covered=diag['nearest'][:,mask]<=.25; result[subset]={}
        for branch,j in [('direct_second',1),('handoff',2)]:
            error=diag['designed_errors'][:,j,mask]
            result[subset][branch]={'n':int(error.size),'coverage':float(covered.mean()),
              'error_rate':float(error.mean()),
              'error_when_covered':float(error[covered].mean()) if covered.any() else None,
              'correct_when_uncovered':float((~error)[~covered].mean()) if (~covered).any() else None,
              'errors_with_coverage_fraction':float(covered[error].mean()) if error.any() else None,
              'correct_without_coverage_fraction':float((~covered)[~error].mean()) if (~error).any() else None}
    return result

def analyze(weights,roles):
    """Report and arrays. Off-manifold swaps show route dependence, not count causality."""
    x,names,diag=coverage_features(weights,negative_cutoff=0.,canonical_route=True)
    xr,_,dr=coverage_features(weights,roles)
    xh,_,dh=coverage_features(weights)
    report={'cheap_feature_recurrent_steps':672,'cheap_feature_forward_equivalent':67.2,
      'late_pointer_networks':int(np.any(roles==3,axis=1).sum()),
      'event_pointer_networks':int(np.any(roles==2,axis=1).sum()),
      'both_pointer_types_networks':int((np.any(roles==2,axis=1)&np.any(roles==3,axis=1)).sum()),
      'no_pointer_networks':int((~np.any((roles==2)|(roles==3),axis=1)).sum()),
      'probe_regions':'S=.45 gives clean cases; S=.8 gives middle direct-second and max-near-tie handoff.',
      'coverage_confusion':{k:coverage_confusion(d) for k,d in
        [('sign_only',diag),('cached_comparator',dr),('weight_below_minus2',dh)]},
      'threshold_history':threshold_history_fits(weights,roles),
      'causal_swaps':{f'{c}_delay{d}':paired_swaps(weights,roles,c,d)
        for c in ['handoff','write'] for d in [0,1]}}
    arrays={'X_sign':x,'X_cached_roles':xr,'X_historical':xh,
      'feature_names':np.array(names),'nearest_sign':diag['nearest'],
      'theta':diag['theta'],'designed_errors':diag['designed_errors']}
    return report,arrays

# Population calibration and deployment; sklearn is needed only for fitting.
ROLE_NAMES=['dead','always_on','event_pointer','late_pointer','max_detector','late_marker','comparator','other']

def write_json(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,default=convert))

def weight_hash(P):
    h=hashlib.sha256()
    for w in P:h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()

def reference_bank(P,path,samples,seed):
    """Independent evaluation labels; never used as target prediction features."""
    digest=weight_hash(P)
    if path.exists():
        with np.load(path) as z:
            if ('weight_sha256' in z and str(z['weight_sha256'])==digest
                    and int(z['samples'])==samples and 'seed' in z and int(z['seed'])==seed):
                return {k:z[k] for k in z.files}
    rng=np.random.default_rng(seed);cnt=np.zeros(60,np.int64)
    err=np.zeros((len(P[0]),60),np.int64);start=time.perf_counter()
    for b in range(0,samples,4096):
        x=rng.standard_normal((min(4096,samples-b),10),dtype=np.float32)
        lab=task_strata(x);truth=target(x);cnt+=np.bincount(lab,minlength=60)
        for j in range(0,len(P[0]),64):
            e=forward(tuple(w[j:j+64] for w in P),x).argmax(-1)!=truth[None]
            for k,row in enumerate(e):err[j+k]+=np.bincount(lab,weights=row,minlength=60).astype(np.int64)
    c,mass=region_contributions(err,cnt);acc=1-c.sum(1)
    cov=(np.eye(5)[None]*c[:,:,None]-c[:,:,None]*c[:,None,:])/(samples-1)
    r=dict(accuracy=acc,accuracy_se=np.sqrt(acc*(1-acc)/(samples-1)),contributions=c,
           masses=mass,contribution_covariance=cov,region_se=np.sqrt(np.diagonal(cov,axis1=1,axis2=2)),
           strata_err=err,strata_n=cnt,samples=samples,seed=seed,weight_sha256=digest,
           seconds=time.perf_counter()-start,region_names=np.array(REGION_NAMES))
    np.savez_compressed(path,**r);return r

def metrics(pred,Y):
    e=pred-Y;total=e.sum(1);second=e.T@e/len(e)
    return dict(rmse_pp=100*np.sqrt(np.mean(total**2)),bias_pp=-100*total.mean(),
        region_rmse_pp=100*np.sqrt(np.mean(e**2,0)),
        region_R2=1-np.mean(e**2,0)/np.maximum(Y.var(0),1e-30),
        residual_second_moment_pp2=10000*second,residual_mse_allocation_pp2=10000*second.sum(1),
        diagonal_mse_pp2=10000*np.trace(second),cross_region_mse_pp2=10000*(second.sum()-np.trace(second)),
        abs_accuracy_error_quantiles_pp=100*np.quantile(abs(total),[.5,.9,.95,.99]),
        fraction_within_point3=(abs(total)<=.003).mean(),
        oracle_clean_middle_rmse_pp=100*np.sqrt(np.mean(e[:,:3].sum(1)**2)))

def score_predictions(raw,Y,mass,fresh=None):
    clipped=np.clip(raw,0,mass)
    out=dict(raw=metrics(raw,Y),clipped=metrics(clipped,Y),clipping_count_by_region=(raw!=clipped).sum(0))
    if fresh is not None:
        out.update(fresh_raw=metrics(raw,fresh),fresh_clipped=metrics(clipped,fresh))
    return out

def region_columns(names,core=0):
    groups=[('negative_S',),('MS_tie','clean_sparse','clean_crowded'),
        ('SR_tie','clean_sparse','clean_crowded'),
        ('middle','clean_sparse','clean_crowded','MS_tie','SR_tie'),('clean_sparse','clean_crowded')]
    common={'direct_second','max_handoff','S_last','M_last','R_last'}
    return [np.array(list(range(core))+[core+i for i,name in enumerate(names)
        if name.split('/')[0] in set(group)|common],dtype=int) for group in groups]

def make_model(kind):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler,SplineTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor
    if kind=='gbm':return HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=15,random_state=0)
    layers=[SplineTransformer(n_knots=3,degree=2)] if kind=='additive' else []
    return make_pipeline(*layers,StandardScaler(),RidgeCV(alphas=np.logspace(-2,5,15)))

def fit_design(X,ZX,columns,Y,mass,kind,reference,zreference):
    from sklearn.model_selection import KFold
    scores={};oof={}
    for seed in (0,1729):
        prediction=np.zeros_like(Y,dtype=float)
        for train,test in KFold(5,shuffle=True,random_state=seed).split(Y):
            for j,c in enumerate(columns):
                fit=make_model(kind).fit(X[train][:,c],Y[train,j])
                prediction[test,j]=fit.predict(X[test][:,c])
        oof[str(seed)]=prediction
        scores[str(seed)]=score_predictions(prediction,Y,mass,reference)
    fits=[make_model(kind).fit(X[:,c],Y[:,j]) for j,c in enumerate(columns)]
    zp=np.stack([f.predict(ZX[:,c]) for f,c in zip(fits,columns)],1)
    result=dict(scores=scores,feature_columns=[len(c) for c in columns],
        zoo_raw_contributions=zp,zoo_clipped_contributions=np.clip(zp,0,mass),
        zoo_raw_accuracy=1-zp.sum(1),zoo_clipped_accuracy=1-np.clip(zp,0,mass).sum(1),
        zoo_scores=score_predictions(zp,zreference,mass),
        zoo_errors_pp=100*(1-np.clip(zp,0,mass).sum(1)-(1-zreference.sum(1))))
    return result,oof,zp,fits

def save_calibrator(path,design,fits,columns,names,mass):
    """Fold standardization into a plain NumPy affine map; no sklearn at inference."""
    coef=np.zeros((5,len(names)));intercept=np.empty(5)
    for j,(fit,c) in enumerate(zip(fits,columns)):
        scale,ridge=fit.steps[-2][1],fit.steps[-1][1]
        coef[j,c]=ridge.coef_/scale.scale_
        intercept[j]=ridge.intercept_-np.dot(ridge.coef_,scale.mean_/scale.scale_)
    np.savez_compressed(path,coef=coef,intercept=intercept,region_mass=mass,
        design=np.array(design),feature_names=np.array(names),region_names=np.array(REGION_NAMES),
        probe_version=np.array('3; max-second shifts3,6; six regimes,10 rotations'))

def feature_cache(P,path):
    start=time.perf_counter();X,names,bag,bn,F,fn,diag=extract_designed_features(P)
    elapsed=time.perf_counter()-start
    np.savez_compressed(path,X=X,names=np.array(names),bag=bag,bag_names=np.array(bn),
        neuron_features=F,neuron_names=np.array(fn),weight_sha256=np.array(weight_hash(P)),**diag)
    return X,names,bag,bn,F,fn,diag,elapsed

def predict_target(path,calibrator):
    P=tuple(w[None] for w in load_one(path))
    with np.load(calibrator) as z:cal={k:z[k] for k in z.files}
    start=time.perf_counter();X,names,B,bn,F,fn,diag=extract_designed_features(P)
    width=cal['coef'].shape[1]
    if width==B.shape[1]:features,expected=B,bn
    elif width==B.shape[1]+X.shape[1]:features,expected=np.c_[B,X],bn+names
    else:
        W=baseline_features(*P);features=np.c_[W,B,X]
        expected=[f'weight_{i}' for i in range(W.shape[1])]+bn+names
    if width!=features.shape[1]:raise ValueError('Calibrator feature dimension does not match this extractor.')
    if 'feature_names' in cal and list(cal['feature_names'])!=expected:
        raise ValueError('Calibrator feature names/order differ from this extractor.')
    raw=features@cal['coef'].T+cal['intercept'];bounded=np.clip(raw,0,cal['region_mass'])
    elapsed=time.perf_counter()-start
    neurons=[dict(index=j,role=ROLE_NAMES[diag['role'][0,j]],
        parameters={name:float(F[0,j,k]) for k,name in enumerate(fn)}) for j in range(F.shape[1])]
    return dict(target=str(path),calibrator=str(calibrator),accuracy=float(1-bounded.sum()),
        raw_accuracy=float(1-raw.sum()),region_names=REGION_NAMES,
        raw_contributions=raw[0],contributions=bounded[0],region_masses=cal['region_mass'],
        conditional_error=bounded[0]/cal['region_mass'],neurons=neurons,
        diagnostic_inputs=60,seconds=elapsed,euler_absolute_error=diag['euler_abs_error'],
        note='Population-calibrated prediction; no pointwise unbiasedness or universal accuracy guarantee.')

def supplemental_mechanisms(P,cached_roles,label_cache,Xnames,features):
    """Compact checks of readout alignment and limitations of role counts."""
    wi,wh,wo=P;n,h=wi.shape;power=np.broadcast_to(np.eye(h),(n,h,h)).copy();pulse=[]
    for age in range(10):pulse.append(wo@power);power=wh@power
    pulse=np.stack(pulse,1);alignment={}
    for delay in (0,1):
        good=(pulse[:,:10-delay].argmax(2)==(9-delay-np.arange(10-delay))[None,:,None]).mean(1)
        alignment[str(delay)]=dict(best_age_accuracy_percentiles=np.percentile(good.max(1),[5,50,95]),
                                   fraction_with_perfect_neuron=np.any(good==1,axis=1).mean())
    result=dict(ungated_positive_pulse_alignment=alignment,
        caveat='Ungated impulse paths need not remain open in the actual recurrent ReLU network.')
    if cached_roles is not None and 'feats' in label_cache and 'names' in label_cache:
        F,names=label_cache['feats'],list(label_cache['names'])
        key=['w_in','rate','eta_ageM','eta_ageS','corr_M','corr_S','out_dc','out_domshare']
        result['cached_role_means']={ROLE_NAMES[k]:dict(neurons=int((cached_roles==k).sum()),
            features={name:float(F[cached_roles==k,names.index(name)].mean()) for name in key}) for k in (1,7)}
        for k in (1,7):
            mask=cached_roles==k;cm=F[mask,names.index('corr_M')]**2;eta=F[mask,names.index('eta_ageM')]
            result['cached_role_means'][ROLE_NAMES[k]].update(neurons_per_network=float(mask.sum(1).mean()),
                corr_M_squared_mean=float(cm.mean()),eta_ageM_mean=float(eta.mean()),
                fraction_corr_M_squared_above_point25=float((cm>.25).mean()),
                fraction_eta_ageM_above_point25=float((eta>.25).mean()),
                fraction_both_above_point25=float(((cm>.25)&(eta>.25)).mean()))
        result['cached_role_overlap_scope']='Overlapping marginal descriptions, not an additive variance decomposition.'
        w=F[:,:,names.index('w_in')]
        eligible=(F[:,:,names.index('on_neither')]>=.7)&(F[:,:,names.index('on_newmax')]<=.4)
        # All other role priority conditions are fixed under this rescaling.
        possible=np.isin(cached_roles,[6,7])&eligible
        result['comparator_count_under_uniform_hidden_scale']={str(s):float(((s*w < -2)&possible).sum(1).mean()) for s in (.1,1,10)}
    return result

def zoo_pulse_alignment(Z,names):
    """Best signed ungated path, ordered by age accuracy then centered normalized margin."""
    wi,wh,wo=(w.astype(float) for w in Z);n,h=wi.shape
    power=np.broadcast_to(np.eye(h),(n,h,h)).copy();pulse=[]
    for age in range(10):pulse.append(wo@power);power=wh@power
    pulse=np.stack(pulse,1);result={}
    for i,name in enumerate(names):
        candidates=[]
        for sign in (1,-1):
            for delay in (0,1):
                ages=np.arange(10-delay);wanted=9-delay-ages
                for neuron in range(h):
                    v=sign*pulse[i,:10-delay,:,neuron];wrong=v.copy();wrong[ages,wanted]=-np.inf
                    margin=v[ages,wanted]-wrong.max(1)
                    normalized=margin/np.maximum(np.linalg.norm(v-v.mean(1,keepdims=True),axis=1),1e-30)
                    correct=int((v.argmax(1)==wanted).sum())
                    candidates.append(dict(neuron=neuron,sign=sign,delay=delay,correct=correct,ages=len(ages),
                        fraction=correct/len(ages),raw_margins=margin,normalized_margins=normalized,
                        mean_normalized_margin=float(normalized.mean()),worst_normalized_margin=float(normalized.min())))
        rank=lambda d:(d['fraction'],d['mean_normalized_margin'])
        result[name]=dict(best_signed=max(candidates,key=rank),
            best_positive=max((c for c in candidates if c['sign']==1),key=rank))
    return result

def phase_counterexample(Z,names):
    """A cyclic readout shift preserves the original24 z2 features, but not accuracy."""
    x=np.random.default_rng(20261018340).standard_normal((65536,10),dtype=np.float32)
    logits=forward(Z,x);truth=target(x)
    old=np.abs(np.fft.rfft(Z[2].astype(float),axis=1))
    new=np.abs(np.fft.rfft(np.roll(Z[2],1,axis=1).astype(float),axis=1))
    return dict(names=names,samples=len(x),seed=20261018340,
        original_accuracy=(logits.argmax(-1)==truth[None]).mean(1),
        shifted_readout_accuracy=(np.roll(logits,1,axis=-1).argmax(-1)==truth[None]).mean(1),
        max_DFT_magnitude_difference=np.max(abs(old-new)),
        scope='Exact insufficiency of the old phase-free features for arbitrary networks; not an in-population RMSE lower bound.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--population',type=Path);ap.add_argument('--snapshot')
    ap.add_argument('--labels',type=Path);ap.add_argument('--buckets',type=Path)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'roles_results')
    ap.add_argument('--predict',type=Path);ap.add_argument('--calibrator',type=Path)
    ap.add_argument('--gbm',action='store_true');ap.add_argument('--mechanisms',action='store_true')
    ap.add_argument('--fresh-pop',action='store_true',help='Independent2^16 evaluation labels per population model.')
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(a.threads)
    except ImportError:pass
    if a.predict:
        if not a.calibrator:ap.error('--predict requires --calibrator.')
        result=predict_target(a.predict,a.calibrator);write_json(a.out/'prediction.json',result)
        print(json.dumps({k:result[k] for k in ['accuracy','raw_accuracy','diagnostic_inputs','seconds']},indent=2));return
    pop=a.population or a.root/'pop1k/snap_2e28.npz'
    labels=a.labels or pop.with_suffix('.roles.npz')
    buckets=a.buckets or pop.with_suffix('.buckets.npz')
    with np.load(pop) as z:
        step=int(str(a.snapshot).removeprefix('s')) if a.snapshot else max(int(k[1:-4]) for k in z.files if k.endswith('_Wih'))
    P=load_population(pop,'s'+str(step))
    n=len(P[0])
    if tuple(w.shape for w in P)!=((n,16),(n,16,16),(n,10,16)):raise ValueError('Expected16-hidden-unit,10-output RNN weights.')
    if n<5:raise ValueError('Five-fold validation needs at least5 networks.')
    with np.load(labels) as z:cache={k:z[k] for k in z.files}
    E,cnt=cache['strata_err'],cache['strata_n']
    if E.shape!=(len(P[0]),60) or cnt.shape!=(60,):raise ValueError('Expected N×60 error counts and60 stratum counts.')
    if not np.isfinite(E).all() or not np.isfinite(cnt).all() or np.any(cnt<0) or cnt.sum()<2 or np.any(E<0) or np.any(E>cnt):
        raise ValueError('Invalid error/stratum counts.')
    if 'snapshot' in cache and int(cache['snapshot'])!=step:raise ValueError('Label snapshot differs from weight snapshot.')
    if 'acc' in cache and not np.allclose(cache['acc'],1-E.sum(1)/cnt.sum(),atol=1e-7,rtol=0):
        raise ValueError('Cached accuracy and error counts disagree.')
    Y,mass=region_contributions(E,cnt);cached_roles=None
    if buckets.exists():
        with np.load(buckets) as z:
            cached_roles=z['roles']
            if 'snapshot' in z and int(z['snapshot'])!=step:raise ValueError('Bucket snapshot differs from weights.')
        if cached_roles.shape!=P[0].shape:raise ValueError('Cached role shape differs from population.')
    zn,Z=load_zoo(a.root)
    f=feature_cache(P,a.out/'designed_v3_population.npz');zf=feature_cache(Z,a.out/'designed_v3_zoo.npz')
    X,names,B,bn,F,fn,diag,elapsed=f;ZX,_,ZB,_,_,_,_,_=zf
    W,ZW=baseline_features(*P),baseline_features(*Z);wn=[f'weight_{i}' for i in range(W.shape[1])]
    invariant,zi=np.c_[B,X],np.c_[ZB,ZX];hybrid,zh=np.c_[W,B,X],np.c_[ZW,ZB,ZX]
    ic=region_columns(names,B.shape[1]);hc=region_columns(names,W.shape[1]+B.shape[1])
    designs={'weights_additive':(W,ZW,[np.arange(W.shape[1])]*5,'additive',wn),
        'roles60_ridge':(B,ZB,[np.arange(B.shape[1])]*5,'ridge',bn),
        'invariant_matched_ridge':(invariant,zi,ic,'ridge',bn+names),
        'hybrid_matched_ridge':(hybrid,zh,hc,'ridge',wn+bn+names)}
    if a.gbm:
        designs.update(weights_gbm=(W,ZW,[np.arange(W.shape[1])]*5,'gbm',wn),
                       hybrid_gbm=(hybrid,zh,[np.arange(hybrid.shape[1])]*5,'gbm',wn+bn+names))
    zr=reference_bank(Z,a.out/'zoo_reference.npz',2**19,202610180733)
    fresh=reference_bank(P,a.out/'population_reference.npz',2**16,202610180734) if a.fresh_pop else None
    result=dict(config=vars(a),snapshot=step,population_count=len(P[0]),region_names=REGION_NAMES,
        region_mass=mass,mean_contribution_pp=100*Y.mean(0),contribution_sd_pp=100*Y.std(0),
        label_noise_rms_pp=100*np.sqrt(np.mean((1-Y.sum(1))*Y.sum(1)/(cnt.sum()-1))),
        probe_count=60,feature_seconds=elapsed,neuron_parameter_names=fn,
        role_counts_mean=[float((diag['role']==k).sum(1).mean()) for k in range(8)],role_names=ROLE_NAMES,
        zoo_names=zn,zoo_reference_accuracy=zr['accuracy'],zoo_reference_se=zr['accuracy_se'],models={},
        validation='Fixed five-fold network CV, seeds0 and1729; learned preprocessing stays within each outer training fold. Method development used this archive; new lineages are needed for confirmation.')
    raw=dict(contributions=Y,masses=mass)
    for name,(features,zfeatures,columns,kind,feature_names) in designs.items():
        start=time.perf_counter()
        item,oof,zp,fits=fit_design(features,zfeatures,columns,Y,mass,kind,
            None if fresh is None else fresh['contributions'],zr['contributions'])
        item['fit_seconds']=time.perf_counter()-start;result['models'][name]=item
        for seed,pred in oof.items():raw[name+'_oof_'+seed]=pred
        raw[name+'_zoo']=zp
        if kind=='ridge':
            label={'roles60_ridge':'roles','invariant_matched_ridge':'invariant','hybrid_matched_ridge':'hybrid'}[name]
            save_calibrator(a.out/f'calibrated_{label}.npz',name,fits,columns,feature_names,mass)
        print(name,{s:float(v['clipped']['rmse_pp']) for s,v in item['scores'].items()},flush=True)
        write_json(a.out/'results.json',result);np.savez_compressed(a.out/'out_of_fold.npz',**raw)
    if a.mechanisms:
        chosen=cached_roles if cached_roles is not None else diag['role']
        report,arrays=analyze(P,chosen)
        report['role_source']='cached4,096-probe taxonomy' if cached_roles is not None else 'new60-probe taxonomy'
        report.update(supplemental_mechanisms(P,cached_roles,cache,names,X))
        report['zoo_ungated_pulse_alignment']=zoo_pulse_alignment(Z,zn)
        report['phase_counterexample']=phase_counterexample(Z,zn)
        write_json(a.out/'mechanisms.json',report);np.savez_compressed(a.out/'mechanisms.npz',**arrays)
    print('Results written to',a.out,flush=True)

if __name__=='__main__':main()
