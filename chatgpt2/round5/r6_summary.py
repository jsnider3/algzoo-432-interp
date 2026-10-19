"""Assemble tables and the development-selected frontier from saved results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path('r6_results')
def get(file): return json.loads((P/file).read_text())
records = {}
for split in ['dev', 'held']:
    rows = []
    for family in ['low', 'event', 'anchors', 'exact_anchor']:
        for r in get(f'{family}_{split}.json'):
            label = family+':'+':'.join(str(r.get(k, '')) for k in ['n','K','q','bins'])
            m = r[split]
            rows.append(dict(id=label, family=family, cost=r['cost'], rms=m['rms'], mean=m['mean'],
                sampling=m.get('sampling_at_network_costs',m['sampling']),
                ratio=m.get('ratio_at_network_costs',m['ratio']), zoo=r['zoo_errors'],
                lower_bound=r.get('cost_is_lower_bound',False)))
    for r in get(f'correction_combinations_{split}.json'):
        if 'k3_n3_K8_q8' in r['correction']: continue  # benchmark, outside frozen shortlist
        label = r['anchor']+str(r['n'])+':'+r['correction'].replace('_'+split+'_','_')
        rows.append(dict(id=label,family='two_level',cost=r['cost'],rms=r['rms'],mean=r['mean'],
                         sampling=r['sampling_rms'],ratio=r['mse_ratio'],zoo=r['zoo'],
                         lower_bound=r['cost_is_lower_bound'],
                         exact_anchor_omits_scalar_work=r.get('exact_anchor_omits_scalar_work',False),
                         mac_only_cost_lower_bound=r.get('mac_only_cost_lower_bound'),
                         sampling_rms_at_mac_cost=r.get('sampling_rms_at_mac_cost'),
                         mse_ratio_at_mac_cost=r.get('mse_ratio_at_mac_cost')))
    r=get(f'correction_{split}_k3_n3_K8_q8_keep8.json')['same_grid']
    rows.append(dict(id='z38',family='z38',cost=r['cost'],rms=r['rms'],mean=r['mean'],
                     sampling=r['sampling_rms'],ratio=r['mse_ratio'],zoo=r['zoo'],lower_bound=False))
    records[split] = rows

# Selection is a function of development results only; held values are just paired.
best = float('inf'); frontier=[]
held = {r['id']:r for r in records['held']}
for r in sorted(records['dev'],key=lambda x:x['cost']):
    if r['rms'] < best:
        frontier.append(dict(dev=r,held=held[r['id']]))
        best=r['rms']
summary=dict(units='percentage points',selection='development RMS Pareto frontier; no held retuning',
             arithmetic_caveat='Exact-line costs omit scalar work. Other costs use declared scalar allowances, not measured hardware upper bounds.',
             frontier=frontier,all_methods=records)
(P/'summary.json').write_text(json.dumps(summary,indent=2))

fig,ax=plt.subplots(figsize=(10.6,6.5))
c=np.geomspace(40,1e7,400)
ax.loglog(c,100*np.sqrt(.0698014658949404/c),'--',color='#222222',lw=1.8,label='Sampling (held-out variance)')
for split,color,marker in [('dev','#df8b23','o'),('held','#176d94','s')]:
    ax.loglog([r[split]['cost'] for r in frontier],[r[split]['rms'] for r in frontier],
              color=color,marker=marker,lw=1.7,ms=5,label=f'Frozen frontier: {split}')
for split,color in [('dev','#df8b23'),('held','#176d94')]:
    r=next(x for x in records[split] if x['id']=='z38')
    ax.scatter(r['cost'],r['rms'],color=color,marker='X',s=95,zorder=5)
ax.annotate('z38 reference rule',(2.984e6,.621),xytext=(7e5,1.25),
            arrowprops=dict(arrowstyle='-',color='#777777'),fontsize=10)
ax.axhline(.025800749,color='#999999',ls=':',lw=1.2,label='Reference RMS noise (not a hard error floor)')
ax.set(xlim=(40,1e7),ylim=(.006,12),xlabel='Counted forward equivalents per network (log scale)',
       ylabel='RMS accuracy error, percentage points (log scale)',
       title='Two-level quadrature improves the estimate, but remains above sampling')
ax.grid(which='major',alpha=.22);ax.legend(loc='lower left',fontsize=9)
fig.text(.12,.012,'64 development + 64 held-out networks. Exact-line x-coordinates are optimistic MAC counts; scalar work is additional.',fontsize=8,color='#444444')
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(P/'curve.png',dpi=180)
for item in frontier:
    print(item['dev']['id'], {s:{k:item[s][k] for k in ['cost','rms','mean','sampling','ratio']} for s in ['dev','held']})
