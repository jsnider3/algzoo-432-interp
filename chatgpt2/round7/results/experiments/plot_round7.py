"""Recreate the static scientific figure from round7_frontier_data.json."""
import json, os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/round7_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

root=Path(__file__).parent
d=json.loads((root/'round7_frontier_data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                     'axes.labelsize':11,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
styles={'g':('#747b85','o','Original G · 1,024 supports'),
        'compact':('#c57923','s','Compact probes + pulses · 128 supports'),
        'mf4':('#286ba5','o','MF4 + pulses · 128 supports'),
        'groups':('#248164','D','MF4 + pulses + timing groups · 1,024 supports')}
fig,axes=plt.subplots(1,2,figsize=(14,5.65),sharey=True)
for ax,pop in zip(axes,('unif','zoo')):
    cost=np.geomspace(100,10000,250)
    ax.plot(cost,100*np.sqrt(d['sampling_mean_pq'][pop]/cost),ls='--',lw=1.6,
            color='#42464c',label='Random sampling',zorder=1)
    for key,(color,marker,label) in styles.items():
        a=sorted([p for p in d['points'] if p['population']==pop and p['family']==key],
                 key=lambda p:p['k'])
        if not a: continue
        ax.plot([p['fe'] for p in a],[p['rmse_points'] for p in a],color=color,
                marker=marker,ms=6.4,lw=1.8,label=label,zorder=3,
                markerfacecolor='white' if key=='g' else color,markeredgewidth=1.5)
    ax.set(xscale='log',yscale='log',xlim=(100,10000),ylim=(.25,3.4),
           xlabel='Total target compute (forward equivalents)',
           title=d['recipe_labels'][pop]+' population')
    ax.xaxis.set_major_locator(FixedLocator([100,300,1000,3000,10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{int(x):,}'))
    ax.yaxis.set_major_locator(FixedLocator([.3,.4,.6,1,2,3]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='major',color='#e2e5e9',lw=.65); ax.set_axisbelow(True)
    ax.tick_params(which='both',labelsize=10,color='#aeb4bd')
axes[0].set_ylabel('Cross-fit RMS error (percentage points)')
for p,xytext in zip([p for p in d['points'] if p['family']=='groups'],[(335,.405),(1200,.31)]):
    axes[0].annotate(f"{p['fe']:,.0f} FE · {p['rmse_points']:.3f} pp",
                     (p['fe'],p['rmse_points']),xytext=xytext,color=styles['groups'][0],
                     fontsize=10,arrowprops=dict(arrowstyle='-',color=styles['groups'][0],lw=.9))
fig.suptitle(d['title'],fontsize=17,fontweight='semibold',y=.98)
fig.text(.5,.915,'Nested cross-fit across 1,024 networks per recipe. Lower error is better.',
         ha='center',fontsize=10.5,color='#59616d')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,
           frameon=False,fontsize=9.7,columnspacing=1.8,handlelength=2.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.25,top=.835,wspace=.16)
for ext in ('png','pdf','svg'):
    fig.savefig(root/f'round7_frontier.{ext}',dpi=200,facecolor='white')
print('Saved PNG, PDF, and SVG scientific figures.')
