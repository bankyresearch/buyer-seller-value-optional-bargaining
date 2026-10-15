"""Six vector journal figures; all plotted values are numerical, never AI imagery."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import importlib.util
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('original_figures',WORK/'focused_bargaining_20261002/figures/build_figures.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
WIDTH=180/25.4

def save(fig,name,out,records):
    number=int(name.split('_')[1]);fig.set_size_inches(WIDTH,fig.get_size_inches()[1])
    for i,ax in enumerate(fig.axes):
        for line in ax.lines:
            line.set_linewidth(max(1.,line.get_linewidth()))
            line.set_markeredgewidth(max(1.,line.get_markeredgewidth()))
        for spine in ax.spines.values():spine.set_linewidth(1.)
        for grid in ax.get_xgridlines()+ax.get_ygridlines():grid.set_linewidth(1.)
        ax.tick_params(width=1.)
        if number in [2,3,4]:
            ax.text(-.06 if number in [3,4] else -.025,1.08,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
    alias=f'Fig{number}'
    for ext in ['eps','svg','png']:
        fig.savefig(out/f'{alias}.{ext}',format=ext,dpi=600,facecolor='white')
    fig.savefig(out/f'{alias}_preview.png',dpi=150,facecolor='white')
    eps=(out/f'{alias}.eps').read_text(encoding='latin1')
    assert '/FontType 42 def' in eps
    assert 'colorimage' not in eps, 'Unexpected raster data in scientific EPS'
    records.append(dict(figure=alias,width_mm=180,height_mm=float(fig.get_size_inches()[1]*25.4),minimum_visible_line_pt=1,
        png_dpi=600,vector_eps=True,sha256={e:hashlib.sha256((out/f'{alias}.{e}').read_bytes()).hexdigest() for e in ['eps','svg','png']}))
    plt.close(fig);print('Saved',alias,flush=True)

def additional(inputs,out):
    boundaries=pd.read_csv(inputs/'cost_boundaries.csv');products=pd.read_csv(inputs/'product_outcomes.csv')
    records=[];cmax=2.;region_colors=['#E8EBEF','#BCD6E6','#FFE0B0','#BCE0D7']
    fig,axes=plt.subplots(3,3,figsize=(WIDTH,7.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.11,right=.985,bottom=.16,top=.95,hspace=.42,wspace=.16)
    plotted=[]
    for i,regime in enumerate(old.REGIMES):
        for j,r in enumerate([0.,.5,1.]):
            ax=axes[i,j]
            cut=boundaries[(boundaries.regime==regime)&(boundaries.policy=='Full-state FQI')&(abs(boundaries.fallback-r)<1e-9)].set_index('stakeholder')
            cs=float(cut.loc['seller','value'])*100;cb=float(cut.loc['buyer','value'])*100
            xs=[0.,float(np.clip(cs,0,cmax)),cmax];ys=[0.,float(np.clip(cb,0,cmax)),cmax]
            for a in range(2):
                for b in range(2):
                    if xs[a+1]<=xs[a] or ys[b+1]<=ys[b]:continue
                    midx=(xs[a]+xs[a+1])/2;midy=(ys[b]+ys[b+1])/2
                    code=int(midx<=cs)+2*int(midy<=cb)
                    ax.add_patch(Rectangle((xs[a],ys[b]),xs[a+1]-xs[a],ys[b+1]-ys[b],facecolor=region_colors[code],edgecolor='none'))
            if 0<=cs<=cmax:ax.axvline(cs,color='#254B65',linewidth=1.3)
            if 0<=cb<=cmax:ax.axhline(cb,color='#975713',linewidth=1.3)
            ax.plot(.5,.5,'x',color='#222222',markersize=5,markeredgewidth=1.2)
            ax.set(xlim=(0,cmax),ylim=(0,cmax),xticks=[0,1,2],yticks=[0,1,2])
            ax.set_title(f'{regime}\nFallback {r:g}',fontsize=9)
            ax.text(-.06,1.10,chr(97+i*3+j),transform=ax.transAxes,fontweight='bold',fontsize=11)
            plotted.append(dict(regime=regime,fallback=r,seller_ceiling=cs/100,buyer_ceiling=cb/100))
    fig.supxlabel('Seller interaction cost (% of listing price per round)',y=.082,fontsize=9)
    fig.supylabel('Buyer interaction cost (% of listing price per round)',x=.018,fontsize=9)
    fig.legend([Patch(facecolor=c) for c in region_colors],['Neither gains','Seller only','Buyer only','Both gain'],ncol=4,
        loc='lower center',bbox_to_anchor=(.53,.005),frameon=False,fontsize=8,handlelength=1.1,columnspacing=1.0)
    save(fig,'Figure_5_feasibility',out,records)
    pd.DataFrame(plotted).to_csv(out/'Fig5_data.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(WIDTH,7.1),sharey=True)
    fig.subplots_adjust(left=.11,right=.98,bottom=.17,top=.95,hspace=.55,wspace=.26)
    ecdf=[]
    for i,regime in enumerate(old.REGIMES):
        for j,party in enumerate(['seller','buyer']):
            ax=axes[i,j];ax.axvline(0,color='#8B969D',linewidth=1.)
            for policy in old.POLICIES:
                vals=np.sort(products[(products.regime==regime)&(products.policy==policy)][party+'_increment'].to_numpy())*100
                y=np.arange(1,len(vals)+1)/len(vals)
                ax.step(vals,y,where='post',color=old.COLORS[policy],linestyle=old.STYLES[policy],linewidth=1.3,label=policy)
                ecdf.extend(dict(regime=regime,policy=policy,stakeholder=party,increment=v/100,cumulative_fraction=float(fr)) for v,fr in zip(vals,y))
            ax.set_title(regime+'\n'+party.capitalize(),fontsize=9.5)
            ax.set(ylim=(0,1.02),yticks=[0,.5,1]);ax.grid(axis='y',linewidth=1.)
            ax.text(-.05,1.09,chr(97+i*2+j),transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.supxlabel('Product mean increment (% of listing price per original opportunity)',y=.095,fontsize=9)
    fig.supylabel('Cumulative fraction of 193 product contexts',x=.017,fontsize=9)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.009),ncol=2,frameon=False,fontsize=8.5)
    save(fig,'Figure_6_product_distributions',out,records)
    pd.DataFrame(ecdf).to_csv(out/'Fig6_data.csv',index=False)
    return records

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'axes.linewidth':1.,'grid.linewidth':1.,'xtick.major.width':1.,'ytick.major.width':1.})
    old.save=save
    old.figures(WORK/'focused_bargaining_20261002/analysis/results',out)
    extra=additional(ROOT/'results',out)
    report=json.loads((out/'figure_verification.json').read_text());report['figures'].extend(extra)
    report.update(figure_count=6,width_mm=180,minimum_visible_line_pt=1,
        confidence_intervals='Fig2 marginal 95% paired product-bootstrap intervals; Fig3-6 descriptive point estimates; Fig5 boundary intervals in supplementary cost_boundaries.csv',
        source='Recorded scientific outputs; no generative imagery',visual_inspection='Pending')
    (out/'figure_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK/'output/hssc_submission_20261005/Figures');a=p.parse_args();run(a.output)
