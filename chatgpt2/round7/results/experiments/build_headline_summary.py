"""Collect only the prespecified headline rows, retaining their source paths."""
import json
from pathlib import Path

ROOT=Path(__file__).parent
PQ={'unif':0.08415976,'zoo':0.0691045116264}
rows=[]

def add(family,pop,k,fe,cf,source,*,external=None,parts=None,supports=None,d=None,detail=None):
    rows.append(dict(family=family,population=pop,k=k,total_fe=fe,
                     crossfit=dict(rms_points=cf.get('rms',cf.get('rms_points')),
                                   mean_points=cf.get('mean',cf.get('mean_points')),
                                   mse_ratio=cf.get('ratio',cf.get('mse_ratio'))),
                     source_file=source,external=external or {},cost_parts=parts or {},
                     supports=supports,dimensions=d,source_detail=detail))

source='audit_baselines.json'
for r in json.loads((ROOT/source).read_text()):
    ext={k:r[k] for k in ('b128','b64','mystery_error','released_errors') if k in r}
    add('g',r['recipe'],r['k'],r['cost'],r['population'],source,external=ext,
        parts=dict(network_fe=r['net'],other_fe=r['cost']-r['net']),supports=1024,d=r['d'])

for pop in ('unif','zoo'):
    for family in ('compact','mf4'):
        for k in (1,2,4):
            ending='ends_mfboth' if family=='compact' else 'all_mf4'
            source=f'combo_results/{pop}_k{k}_{ending}_p1_sup128.json'
            r=json.loads((ROOT/source).read_text())
            add(family,pop,k,r['total_fe'],r['metrics']['crossfit'],source,
                external={k:v for k,v in r['metrics'].items() if k!='crossfit'},
                parts={k:r[k] for k in ('net_fe','feature_fe','readout_fe','rows','cells')},
                supports=r['supports'],d=r['dimensions'])
    for k in (8,16):
        source=f'extended_results/{pop}_k{k}_mf4_pulse_h0_sup128.json'
        r=json.loads((ROOT/source).read_text())
        add('mf4',pop,k,r['total_fe'],r['metrics']['crossfit'],source,
            external={k:v for k,v in r['metrics'].items() if k!='crossfit'},
            parts={k:r[k] for k in ('net_fe','feature_fe','readout_fe','rows','cells')},
            supports=r['supports'],d=r['dimensions'])

for k in (2,4):
    source=f'aggregates_mf4pulse_unif_k{k}.json'
    r=json.loads((ROOT/source).read_text())['results']['groups']
    add('groups','unif',k,r['total_fe'],r['summary']['crossfit'],source,
        external={k:v for k,v in r['summary'].items() if k!='crossfit'},
        parts={k:r[k] for k in ('net_fe','add_feature_ops','anchor_rows','pulse_rows')},
        supports=1024,d=r['d'],detail='results.groups')

meta=dict(title='Accuracy versus target compute',rms_unit='percentage points',
          compute_unit='forward equivalents',sampling_mean_pq=PQ,
          sampling_formula='100 * sqrt(mean_pq / total_FE)',
          validation='Eight outer folds on 1024 networks per recipe; internal readout tuning.',
          selection_note='Selected tested model families; lines connect increasing K.',
          recipe_labels=dict(unif='Mystery recipe',zoo='AlgZoo recipe'),
          external_note='Zoo released_errors contain mystery then seeds 0-4; mystery uses another recipe.',
          pulse_probe_note='Pulse row counts in source files denote two trajectories; 20 age outputs are decoded.',
          baseline_cost_note='G uses the audited advertised arithmetic budget, with scalar input products cached.')
(ROOT/'headline_summary.json').write_text(json.dumps(dict(**meta,records=rows),indent=2))
points=[dict(family=r['family'],population=r['population'],k=r['k'],
             fe=r['total_fe'],rmse_points=r['crossfit']['rms_points'],
             mean_points=r['crossfit']['mean_points'],mse_ratio=r['crossfit']['mse_ratio'],
             source_file=r['source_file'],source_detail=r['source_detail']) for r in rows]
(ROOT/'round7_frontier_data.json').write_text(json.dumps(dict(**meta,points=points),indent=2))
print('Saved',len(rows),'selected records and plot points.')
