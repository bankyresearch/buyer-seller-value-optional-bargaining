"""Rebuild publication figures from immutable numerical analysis intermediates."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent
POLICIES = ['Full-state FQI', 'Clock-only FQI', 'Time concession', 'Buyer responsive']
REGIMES = ['Reference', 'Stubborn offers', 'Low valuations']
COLORS = dict(zip(POLICIES, ['#163D5A', '#167D8D', '#C77735', '#697586']))
STYLES = dict(zip(POLICIES, ['-', '--', '-.', ':']))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#89939E', 'axes.linewidth': .6, 'grid.color': '#E0E5E8',
    'grid.linewidth': .55, 'text.color': '#183341', 'axes.labelcolor': '#183341',
    'xtick.color': '#465560', 'ytick.color': '#465560',
    'svg.fonttype': 'none', 'ps.fonttype': 42, 'pdf.fonttype': 42,
    'ps.useafm': False, 'savefig.facecolor': 'white'})


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name, out, records):
    fig.canvas.draw()
    fig.savefig(out/(name+'.png'), dpi=1200, facecolor='white')
    fig.savefig(out/(name+'.eps'), format='eps', facecolor='white')
    fig.savefig(out/(name+'.svg'), format='svg', facecolor='white')
    fig.savefig(out/(name+'_preview.png'), dpi=150, facecolor='white')
    with Image.open(out/(name+'.png')) as im:
        assert im.size[0] == 7560
        pixels = list(im.size)
        assert abs(im.info['dpi'][0]-1200) < .1
    svg = ET.parse(out/(name+'.svg')).getroot()
    assert svg.findall('.//{http://www.w3.org/2000/svg}text'), 'Editable SVG text missing'
    eps = (out/(name+'.eps')).read_text(encoding='latin1')
    assert '/FontType 42 def' in eps, 'Embedded TrueType font missing'
    publisher_alias = 'Fig'+name.split('_')[1]
    for suffix in ['.png', '.eps', '.svg', '_preview.png']:
        shutil.copy2(out/(name+suffix), out/(publisher_alias+suffix))
    records.append(dict(figure=name, width_inches=6.3, height_inches=float(fig.get_size_inches()[1]),
        png_pixels=pixels, png_dpi=1200, preview_dpi=150, eps_type42_fonts=True,
        svg_editable_text=True, publisher_alias=publisher_alias, file_sha256={suffix: sha(out/(name+suffix)) for suffix in
            ['.png', '.eps', '.svg', '_preview.png']}))
    plt.close(fig)
    print('SAVED', name, flush=True)


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x')
    ax.axvline(0, color='#657786', linewidth=.8, zorder=1)
    ax.tick_params(axis='y', length=0)


def workflow(out, records):
    fig, ax = plt.subplots(figsize=(6.3, 7.2))
    fig.subplots_adjust(left=.02, right=.98, bottom=.015, top=.985)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, title, body, dashed=False, fontsize=8.6, title_offset=.025, body_offset=.058):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.009,rounding_size=0.012',
            facecolor='white', edgecolor='#647784' if dashed else '#167D8D',
            linewidth=1., linestyle='--' if dashed else '-')
        ax.add_patch(rect)
        ax.text(x+w/2, y+h-title_offset, title, ha='center', va='top', fontsize=9.2, fontweight='bold')
        ax.text(x+w/2, y+h-body_offset, body, ha='center', va='top', fontsize=fontsize, linespacing=1.45)
    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
            color='#546C7D', linewidth=1., linestyle='--' if dashed else '-'))
    box(.055, .874, .89, .105, 'Public corpora and listing contexts',
        'Declared data splits; separate simulator training; frozen policy selection')
    box(.035, .675, .435, .142, 'Four frozen seller policies',
        'Full-state FQI / Clock-only FQI\nFQI is trained in the simulator\nTime concession / Buyer responsive')
    box(.555, .675, .41, .142, 'Matched simulated opportunities',
        '193 products × 80 draws\nPrivate values stay in the evaluator')
    arrow((.28, .865), (.25, .825)); arrow((.75, .865), (.76, .825))
    box(.035, .48, .435, .137, 'Negotiation potential outcomes',
        'Agreement / transaction price / rounds\nCommon draws; five frozen FQI seeds')
    box(.555, .48, .41, .137, 'Parallel checkout counterfactual',
        'Full-price sale if value ≥ listing\nSame original opportunities')
    arrow((.25, .667), (.25, .625)); arrow((.76, .667), (.76, .625))
    arrow((.555, .712), (.465, .607))
    box(.055, .277, .89, .14, 'Conditional route accounting',
        'Stipulated opt-in q; failed-deal fallback r; separate seller and buyer costs\n'
        'Direct checkout / negotiation sale / fallback sale / no sale')
    arrow((.25, .472), (.32, .425)); arrow((.76, .472), (.68, .425))
    box(.055, .115, .89, .115, 'Per-opportunity value and feasibility boundaries',
        'New margin − discounts − lost checkout − interaction cost\n'
        'Seller / buyer increments; cost ceilings; fallback boundaries')
    arrow((.50, .269), (.50, .238))
    box(.10, .020, .80, .067, 'Future customer pilot and endogenous choice',
        'Not completed in this study', dashed=True, fontsize=8.1, title_offset=.014, body_offset=.043)
    arrow((.50, .107), (.50, .095), dashed=True)
    save(fig, 'Figure_1_workflow', out, records)


def figures(inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(inputs/'primary_intervals.csv')
    grid = pd.read_csv(inputs/'grid_point_estimates.csv')
    primary = intervals[(intervals.acceptance_coefficient == .25) & (intervals.selection == 'Exogenous')]
    pgrid = grid[(grid.acceptance_coefficient == .25) & (grid.selection == 'Exogenous') &
                 (grid.base_opt_in == .5) & (grid.fallback == .5) & (grid.cost == .005)]
    assert len(pgrid) == 12 and len(primary) == 120
    numeric = []; records = []
    def row(regime, policy, metric, acceptance=.25):
        found = intervals[(intervals.regime == regime) & (intervals.policy == policy) &
            (intervals.metric == metric) & (intervals.acceptance_coefficient == acceptance) &
            (intervals.selection == 'Exogenous')]
        assert len(found) == 1
        return found.iloc[0]
    def point(figure, regime, policy, metric, value, acceptance=.25, lo=None, hi=None, sign=1):
        matching = grid[(grid.regime == regime) & (grid.policy == policy) &
            (grid.acceptance_coefficient == acceptance) & (grid.selection == 'Exogenous') &
            (grid.base_opt_in == .5) & (grid.fallback == .5) & (grid.cost == .005)]
        assert len(matching) == 1 and abs(float(matching[metric].iloc[0])-value) < 1e-12
        numeric.append(dict(figure=figure, regime=regime, policy=policy, metric=metric,
            acceptance_coefficient=acceptance, source_mean=value, source_lo=lo, source_hi=hi,
            plotted_value=100*sign*value, plotted_lo=None if lo is None else 100*lo,
            plotted_hi=None if hi is None else 100*hi,
            units='percent_listing_price_per_original_opportunity', sign=sign,
            primary_source='primary_intervals.csv' if lo is not None else 'grid_point_estimates.csv'))
    workflow(out, records)
    # Figure2: confidence intervals are the computed paired-product marginal intervals.
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.3), sharex=True)
    fig.subplots_adjust(left=.235, right=.965, top=.91, bottom=.14, wspace=.16, hspace=.62)
    bounds = primary[primary.metric.isin(['seller_increment', 'buyer_increment'])]
    limit = math.ceil(max(abs(bounds.lo.min()), abs(bounds.hi.max()))*100*2)/2
    for r, regime in enumerate(REGIMES):
        for c, metric in enumerate(['seller_increment', 'buyer_increment']):
            ax = axes[r, c]; style(ax)
            ax.set_title(regime+'\n'+('Seller' if c == 0 else 'Buyer'), fontsize=9.5, pad=7)
            for y, policy in enumerate(POLICIES):
                d = row(regime, policy, metric)
                ax.errorbar(d['mean']*100, y, xerr=[[100*(d['mean']-d.lo)], [100*(d.hi-d['mean'])]],
                    fmt='o', color=COLORS[policy], markersize=4.4, capsize=2.3, linewidth=1.2, zorder=3)
                point(2, regime, policy, metric, float(d['mean']), lo=float(d.lo), hi=float(d.hi))
            ax.set_yticks(range(4), POLICIES if c == 0 else ['']*4)
            ax.set_ylim(3.55, -.55); ax.set_xlim(-limit, limit); ax.set_xticks(np.arange(-2, 2.1, 1))
    fig.supxlabel('Increment relative to checkout\n(% of listing price per original opportunity)', y=.045, fontsize=9)
    save(fig, 'Figure_2_seller_buyer_forest', out, records)
    # Figure3: components share a common denominator; diamonds are computed net sums.
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.6), sharex=True)
    fig.subplots_adjust(left=.235, right=.97, top=.945, bottom=.20, hspace=.50)
    negative = ['discount_cannibalization', 'lost_checkout_margin', 'seller_interaction_cost']
    component_colors = {'new_margin': '#167D8D', 'discount_cannibalization': '#C77735',
                        'lost_checkout_margin': '#C3CBD2', 'seller_interaction_cost': '#697586'}
    maximum = float(max(pgrid.new_margin.max(), pgrid[negative].sum(axis=1).max()))*100
    limit = math.ceil(maximum*1.08)
    for ax, regime in zip(axes, REGIMES):
        style(ax); ax.set_title(regime, loc='left', pad=7, fontsize=10)
        for y, policy in enumerate(POLICIES):
            d = pgrid[(pgrid.regime == regime) & (pgrid.policy == policy)].iloc[0]
            ax.barh(y, 100*d.new_margin, color=component_colors['new_margin'], height=.55)
            point(3, regime, policy, 'new_margin', float(d.new_margin))
            left = 0.
            for metric in negative:
                amount = -100*float(d[metric])
                ax.barh(y, amount, left=left, color=component_colors[metric], height=.55)
                point(3, regime, policy, metric, float(d[metric]), sign=-1); left += amount
            expected = 100*d.new_margin+left
            assert abs(expected-100*d.seller_increment) < 1e-10
            ax.plot(expected, y, marker='D', markersize=4.6, color='#102F46', markeredgecolor='white', markeredgewidth=.55)
            point(3, regime, policy, 'seller_increment', float(d.seller_increment))
        ax.set_yticks(range(4), POLICIES); ax.set_ylim(3.55, -.55); ax.set_xlim(-limit, limit)
    handles = [Patch(color=component_colors[x], label=label) for x, label in
        [('new_margin', 'New-sale margin'), ('discount_cannibalization', 'Discount displacement'),
         ('lost_checkout_margin', 'Lost checkout margin'), ('seller_interaction_cost', 'Seller interaction cost')]]
    handles.append(Line2D([0], [0], marker='D', color='white', markerfacecolor='#102F46', markersize=5, label='Net increment'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.56, .08), ncol=2, frameon=False,
               columnspacing=1.4, handlelength=1.6, labelspacing=.5)
    fig.supxlabel('% of listing price per original opportunity', y=.025, fontsize=9)
    save(fig, 'Figure_3_seller_decomposition', out, records)
    # Figure4: markers are evaluated settings; segments are visual guides, not fitted curves.
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.97, top=.95, bottom=.20, hspace=.42)
    acceptances = [0., .25, .5, 1.]
    selected = intervals[(intervals.metric == 'seller_increment') & (intervals.selection == 'Exogenous')]
    lower = math.floor(selected['mean'].min()*100*2)/2
    upper = math.ceil(selected['mean'].max()*100*2)/2
    for ax, regime in zip(axes, REGIMES):
        ax.set_title(regime, loc='left', fontsize=10, pad=7)
        ax.grid(axis='y'); ax.set_axisbelow(True); ax.axhline(0, color='#657786', linewidth=.8)
        for policy in POLICIES:
            means = []
            for acceptance in acceptances:
                d = row(regime, policy, 'seller_increment', acceptance)
                means.append(100*d['mean'])
                point(4, regime, policy, 'seller_increment', float(d['mean']), acceptance)
            ax.plot(acceptances, means, marker='o', markersize=3.7, linewidth=1.3,
                    color=COLORS[policy], linestyle=STYLES[policy], label=policy)
        ax.set_ylim(lower-.10, upper+.10); ax.set_xlim(-.02, 1.02)
        ax.set_xticks(acceptances, ['0', '0.25', '0.50', '1.00'])
    axes[-1].set_xlabel('Acceptance coefficient (evaluated settings)', labelpad=6)
    fig.supylabel('Seller increment (% of listing price per original opportunity)', x=.035, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.55, .07),
               ncol=2, frameon=False, handlelength=2.5, columnspacing=1.8)
    fig.text(.55, .032, 'Segments connect evaluated points; no response curve is fitted.', ha='center', fontsize=8)
    save(fig, 'Figure_4_acceptance_sensitivity', out, records)
    pd.DataFrame(numeric).to_csv(out/'numeric_plot_data.csv', index=False)
    assert len(numeric) == 132
    report = dict(status='passed', numeric_points_verified=len(numeric), figure_count=4,
        source_hashes={name: sha(inputs/name) for name in ['primary_intervals.csv', 'grid_point_estimates.csv']},
        build_source_sha256=sha(Path(__file__)), figures=records,
        confidence_intervals='Figure2 only, product-bootstrap95%; Figures3 and4 are point estimates',
        line_segments='Figure4 segments are visual connections between evaluated settings, not interpolation estimates',
        visual_inspection='Pending manual preview inspection; record separately after view_image')
    (out/'figure_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('VERIFIED', len(numeric), 'plotted values; editable SVG and embedded EPS fonts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=ROOT.parent/'analysis/results')
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args(); figures(args.inputs, args.output)
