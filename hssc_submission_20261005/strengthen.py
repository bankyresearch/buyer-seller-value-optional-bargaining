"""Exploratory checks with frozen policies. Run from a preserved repository tree."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import importlib.util
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import beta
import joblib

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ca = module('channel_analysis', WORK/'focused_bargaining_20261002/analysis/channel_analysis.py')
gt = module('generate_trajectories', WORK/'focused_bargaining_20261002/integration/generate_trajectories.py')
BASE = WORK/'focused_bargaining_20261002/integration/results'
OLD = WORK/'focused_bargaining_20261002/analysis/results'
PROTOCOL = json.loads((ROOT/'protocol.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, data): p.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')

def summarize(frame, identifying, weights):
    co, raw, _ = ca.aggregate_coefficients(frame)
    co = ca.weighted(co, .5, 'Exogenous')
    values = ca.outcome(co, .5, .005, .005)
    oracle = ca.route_oracle(frame, .5, .5, .005, .005, 'Exogenous')
    direct = np.column_stack([.5*(raw['a_s']+.5*raw['u_s']-.005*raw['h']),
        .5*(raw['a_b']+.5*raw['u_b']-.005*raw['h']),
        .5*(raw['a_d']+.5*raw['u_d']), .5*raw['h']])
    np.testing.assert_allclose(direct, oracle, atol=1e-12, rtol=0)
    product = pd.DataFrame(values, index=co.index).reset_index().assign(**identifying)
    intervals = [dict(**identifying, metric=m, **ca.ci(values[m], weights)) for m in
                 ['seller_increment', 'buyer_increment', 'joint_increment', 'conversion_increment']]
    return product, intervals

def existing(out):
    weights = ca.bootstrap_weights(193)
    products, intervals, seeds, loco, distributions, boundaries = [], [], [], [], [], []
    mapping = pd.read_csv(gt.lc.legacy.DATA/'amazon_products.csv').set_index('product').category
    for regime in gt.REGIMES:
        for policy in gt.POLICIES:
            path = BASE/f'base_{gt.slug(regime)}_a0.25_{policy["key"]}.csv.gz'
            frame = pd.read_csv(path)
            ident = dict(regime=regime, policy=policy['label'])
            product, cis = summarize(frame, ident, weights)
            product['category'] = product['product'].map(mapping)
            products.append(product); intervals.extend(cis)
            for seed, subset in frame.groupby('training_seed'):
                p, _ = summarize(subset, ident, weights)
                seeds.append(dict(**ident, training_seed=int(seed), **p[['seller_increment', 'buyer_increment', 'joint_increment']].mean().to_dict()))
            for category in sorted(product.category.unique()):
                subset = product[product.category != category]
                loco.append(dict(**ident, excluded_category=category, remaining_products=len(subset), **subset[['seller_increment', 'buyer_increment', 'joint_increment']].mean().to_dict()))
            s, b = product.seller_increment, product.buyer_increment
            distributions.append(dict(**ident, products=len(product), both_positive=int(((s>0)&(b>0)).sum()),
                seller_negative=int((s<0).sum()), buyer_negative=int((b<0).sum()), both_negative=int(((s<0)&(b<0)).sum()),
                **{f'{party}_{stat}':float(vals.quantile(q)) for party, vals in [('seller',s),('buyer',b)] for stat,q in [('p10',.1),('median',.5),('p90',.9)]}))
            co, _, _ = ca.aggregate_coefficients(frame); co = ca.weighted(co, .5, 'Exogenous')
            for fallback in np.linspace(0, 1, 101):
                for party in ['s','b']:
                    boundaries.append(dict(**ident, fallback=float(fallback), stakeholder='seller' if party=='s' else 'buyer',
                        **ca.ratio_ci(co['a_'+party]+fallback*co['u_'+party], co.h, weights)))
    for name, data in [('product_outcomes',pd.concat(products)),('primary_intervals',intervals),('seed_means',seeds),
                       ('leave_category_out',loco),('product_distribution',distributions),('cost_boundaries',boundaries)]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
    canonical = pd.read_csv(OLD/'primary_intervals.csv').query("acceptance_coefficient == .25 and selection == 'Exogenous'")
    for r in intervals:
        c = canonical[(canonical.regime==r['regime'])&(canonical.policy==r['policy'])&(canonical.metric==r['metric'])].iloc[0]
        for field in ['mean','lo','hi']: assert abs(c[field]-r[field])<1e-12
    print('Existing analyses complete; canonical estimates and intervals reproduced',flush=True)

def scenarios(products, setting):
    opening = pd.read_csv(gt.lc.legacy.DATA/'craigslist_dialogues.csv').query("split == 'train'").buyer_first.dropna().clip(.3,.85).values
    rows, randoms = [], []
    for r in products.itertuples():
        seed = int(hashlib.sha256((r.product+':revision-v1:'+gt.STREAM).encode()).hexdigest()[:12],16)
        rng = np.random.default_rng(seed)
        for j in range(80):
            u = rng.random()
            quantile = u if setting['distribution']=='uniform' else beta.ppf(u,setting['alpha'],setting['beta'])
            value = setting['low']+(setting['high']-setting['low'])*quantile
            initial = min(float(rng.choice(opening)), .9*value)
            deadline=int(rng.choice([3,5,8])); randoms.append(rng.random(8))
            floor=float(np.clip(r.lowest/r.current*setting['floor_factor'],.2,.95))
            rows.append((r.product,j,r.category,r.current,floor,value,initial,deadline,j%4))
    df=pd.DataFrame(rows,columns=['product','rep','category','listing','floor','value','opening','deadline','kind'])
    cfg=dict(dropout=.05,floor_factor=setting['floor_factor'],vlow=setting['low'],vhigh=setting['high'],horizon=8,behavior='reference',ask_dropout=False)
    return df,np.stack(randoms),cfg

def structural(out):
    products=pd.read_csv(gt.lc.legacy.DATA/'amazon_products.csv').query("split == 'test'").sort_values('product').reset_index(drop=True)
    weights=ca.bootstrap_weights(193)
    rules=json.loads((gt.lc.OLD/'results/selected_rules.json').read_text())
    bundles, hashes={},{}
    for policy in gt.POLICIES[2:]:
        for seed in gt.SEEDS:
            path=gt.STRENGTH/f'learning_results/fqi_{policy["specification"]}_40_{seed}.joblib'
            hashes[str(path.relative_to(WORK))]=sha(path)
            bundle=joblib.load(path); bundle['model'].n_jobs=2
            bundles[policy['key'],seed]=bundle
    intervals, product_rows, manifest=[],[],[]
    for setting in PROTOCOL['structural_settings']:
        df, randoms, cfg=scenarios(products,setting)
        if setting['name']=='Reference replay':
            original, oldrandom, _=gt.lc.legacy.scenarios(products,80,'Reference',stream=gt.STREAM)
            pd.testing.assert_frame_equal(df,original,check_exact=False,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(randoms,oldrandom)
        for policy in gt.POLICIES:
            parts=[]
            for seed in policy['seeds']:
                if seed==-1:
                    b=gt.pc.negotiation(df,randoms,cfg,policy['label'],rules[policy['label']],.25)
                    b['total_surplus']=b.seller_surplus+b.buyer_surplus
                else: b=gt.learned_negotiation(df,randoms,cfg,bundles[policy['key'],seed],.25)
                b['training_seed']=seed; b['policy']=policy['label']; b['regime']=setting['name']; b['acceptance_coefficient']=.25
                gt.check_base(b,193*80); parts.append(b)
            frame=pd.concat(parts,ignore_index=True).sort_values(['product','rep','training_seed']).reset_index(drop=True)
            if setting['name']=='Reference replay':
                original=pd.read_csv(BASE/f'base_reference_a0.25_{policy["key"]}.csv.gz').sort_values(['product','rep','training_seed']).reset_index(drop=True)
                for field in gt.METRICS+['price_ratio','floor','value','opening']:
                    np.testing.assert_allclose(frame[field],original[field],atol=1e-12,rtol=0,equal_nan=True)
            dest=out/'trajectories'/f'{gt.slug(setting["name"])}_{policy["key"]}.csv.gz'
            frame.to_csv(dest,index=False)
            p, cis=summarize(frame,dict(setting=setting['name'],policy=policy['label']),weights)
            product_rows.append(p);intervals.extend(cis)
            manifest.append(dict(setting=setting['name'],policy=policy['label'],file=dest.name,rows=len(frame),sha256=sha(dest)))
            pd.DataFrame(intervals).to_csv(out/'structural_intervals.csv',index=False)
            pd.concat(product_rows).to_csv(out/'structural_product_outcomes.csv',index=False)
            pd.DataFrame(manifest).to_csv(out/'structural_manifest.csv',index=False)
            print('Structural completed:',setting['name'],policy['label'],len(frame),flush=True)
    for path,digest in hashes.items(): assert sha(WORK/path)==digest
    dump(out/'structural_verification.json',dict(settings=8,files=len(manifest),executions=sum(x['rows'] for x in manifest),
        reference_replay='passed to 1e-12 for all policies and seeds',route_oracle='passed every row',bounds='passed every row',
        checkpoint_hashes=hashes,checkpoint_immutability=True,protocol_sha256=sha(ROOT/'protocol.json')))

def run(out, stage):
    out.mkdir(parents=True,exist_ok=True);(out/'trajectories').mkdir(exist_ok=True)
    if stage in ['all','existing']:existing(out)
    if stage in ['all','structural']:structural(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--stage',choices=['all','existing','structural'],default='all')
    args=p.parse_args();run(args.output,args.stage)
