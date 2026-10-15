"""Frozen learned and rule sellers in the SAME optional-channel scenarios.

No model fitting or channel-design selection. Numerical outputs are sufficient
per-opportunity inputs for the separate independent channel-accounting analysis.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
STRENGTH = WORKSPACE/'bargaining_strengthening_20260926'
sys.path.insert(0, str(STRENGTH))
import learning_controls as lc
import platform_controls as pc

SEEDS = [271, 811, 1601, 2027, 4093]
REGIMES = ['Reference', 'Stubborn offers', 'Low valuations']
COEFFICIENTS = [.25, 0., .5, 1.]
POLICIES = [
    dict(key='time', label='Time concession', seeds=[-1]),
    dict(key='responsive', label='Buyer responsive', seeds=[-1]),
    dict(key='fqi_full', label='Full-state FQI', seeds=SEEDS, specification='mixed_enriched_128'),
    dict(key='fqi_clock', label='Clock-only FQI', seeds=SEEDS, specification='clock_mixed_enriched_128')]
METRICS = ['deal', 'seller_surplus', 'buyer_surplus', 'total_surplus', 'rounds']
STREAM = 'platform-controls-20260926'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def slug(value):
    return str(value).lower().replace(' ', '_')


def learned_negotiation(df, randoms, cfg, bundle, coefficient):
    """Equivalent to frozen learned rollout at a=.25, differing only in a."""
    n = len(df)
    listing = df.listing.to_numpy()
    floor = np.ceil(df.floor.to_numpy()*listing*100-1e-9)/(listing*100)
    value = df.value.to_numpy()
    previous_ask = np.ones(n)
    previous_bid = df.opening.to_numpy().copy()
    minimum = np.full(n, np.inf)
    maximum = np.full(n, -np.inf)
    alive = np.ones(n, bool)
    deal = np.zeros(n, bool)
    price = np.full(n, np.nan)
    rounds = np.zeros(n, int)
    for t in range(cfg['horizon']):
        if not alive.any():
            break
        bid = lc.legacy.customer_bid(t, df, cfg, previous_ask, previous_bid)
        state = lc.legacy.public_state(t, cfg['horizon'], floor, bid, previous_ask, previous_bid, minimum, maximum)
        ids = np.flatnonzero(alive)
        action = lc.q_values(bundle['model'], state[ids], bundle['mode'], bundle['actions']).argmax(axis=1)
        available = lc.quote_candidates(t, cfg['horizon'], floor[ids], previous_ask[ids], bundle['actions'])
        ask = previous_ask.copy()
        ask[ids] = available[np.arange(len(ids)), action]
        ask = np.maximum(floor, np.minimum(previous_ask, np.ceil(ask*listing*100-1e-9)/(listing*100)))
        assert np.all(ask[alive] >= floor[alive]-1e-10)
        assert np.all(ask[alive] <= previous_ask[alive]+1e-10)
        assert np.max(np.abs(ask[alive]*listing[alive]*100-np.round(ask[alive]*listing[alive]*100))) < 1e-5
        seller_accept = alive & (bid >= ask-1e-12)
        buyer_accept = alive & ~seller_accept & (ask <= bid+coefficient*np.maximum(value-bid, 0)+1e-12) & (ask <= value+1e-12)
        agreed = seller_accept | buyer_accept
        price[seller_accept] = bid[seller_accept]
        price[buyer_accept] = ask[buyer_accept]
        deal |= agreed
        rounds[alive] = t+1
        dropout = cfg['dropout']
        if cfg['ask_dropout']:
            dropout = np.clip(dropout+.4*np.maximum(ask-bid, 0), 0, .5)
        alive = alive & ~agreed & (t+1 < df.deadline.to_numpy()) & (randoms[:, t] >= dropout) & (t+1 < cfg['horizon'])
        minimum = np.minimum(minimum, bid)
        maximum = np.maximum(maximum, bid)
        previous_ask = ask
        previous_bid = bid
    result = df.copy()
    result['floor'] = floor
    result['deal'] = deal.astype(int)
    result['rounds'] = rounds
    result['price_ratio'] = price
    result['seller_surplus'] = np.where(deal, price-floor, 0)
    result['buyer_surplus'] = np.where(deal, value-price, 0)
    result['total_surplus'] = result.seller_surplus+result.buyer_surplus
    return result


def check_base(base, expected_rows):
    assert len(base) == expected_rows
    assert not base.duplicated(['product', 'rep', 'training_seed']).any()
    assert base[base.price_ratio.isna()].deal.eq(0).all()
    assert base.loc[base.deal == 1, 'price_ratio'].notna().all()
    assert base['rounds'].between(1, 8).all()
    assert (base.seller_surplus >= -1e-10).all() and (base.buyer_surplus >= -1e-10).all()
    assert (base.loc[base.deal == 0, ['seller_surplus', 'buyer_surplus', 'total_surplus']] == 0).all().all()
    np.testing.assert_allclose(base.total_surplus, base.seller_surplus+base.buyer_surplus, rtol=0, atol=1e-12)
    d = base.deal == 1
    np.testing.assert_allclose(base.loc[d, 'total_surplus'], base.loc[d, 'value']-base.loc[d, 'floor'], rtol=0, atol=1e-12)
    assert (base.loc[d, 'price_ratio'] >= base.loc[d, 'floor']-1e-10).all()
    assert (base.loc[d, 'price_ratio'] <= base.loc[d, 'value']+1e-10).all()
    prices = base.loc[d, 'price_ratio']*base.loc[d, 'listing']*100
    np.testing.assert_allclose(prices, np.round(prices), rtol=0, atol=1e-5)
    assert np.isfinite(base[['listing', 'floor', 'value', 'opening', 'rounds']+METRICS].to_numpy()).all()


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    product_source = lc.legacy.DATA/'amazon_products.csv'
    all_products = pd.read_csv(product_source)
    sets = [set(all_products.loc[all_products.split == x, 'product']) for x in ['train', 'dev', 'test']]
    assert not any(sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)])
    products = all_products.query("split == 'test'").sort_values('product').reset_index(drop=True)
    assert len(products) == 193 and products['product'].is_unique
    settings_path = STRENGTH/'learning_results/frozen_settings.json'
    settings = json.loads(settings_path.read_text())
    assert settings['seeds'] == SEEDS
    rule_path = lc.OLD/'results/selected_rules.json'
    rules = json.loads(rule_path.read_text())
    bundles, checkpoint_hashes = {}, {}
    for policy in POLICIES[2:]:
        name = policy['specification']
        assert settings['selected_leaf'][name] == 40
        for seed in SEEDS:
            checkpoint = STRENGTH/f'learning_results/fqi_{name}_40_{seed}.joblib'
            bundle = joblib.load(checkpoint)
            assert bundle['leaf'] == 40 and bundle['seed'] == seed
            assert bundle['spec'] == lc.SPECS[name]
            bundles[(policy['key'], seed)] = bundle
            checkpoint_hashes[str(checkpoint.relative_to(WORKSPACE))] = sha(checkpoint)
    sources = [product_source, lc.legacy.DATA/'craigslist_dialogues.csv', rule_path, settings_path,
               STRENGTH/'learning_controls.py', STRENGTH/'platform_controls.py', lc.OLD/'experiment.py', lc.legacy.BASE/'simulate.py']
    source_hashes = {str(path.relative_to(WORKSPACE)): sha(path) for path in sources}
    protocol_hash = sha(ROOT/'protocol.json')
    manifests, products_out, seed_out = [], [], []
    scenario_checks, rule_checks, learned_checks = 0, 0, 0
    begin = time.perf_counter()
    for regime in REGIMES:
        df, randoms, cfg = lc.legacy.scenarios(products, 80, regime, stream=STREAM)
        scenario_slug = slug(regime)
        df.to_csv(out/f'scenarios_{scenario_slug}.csv.gz', index=False)
        np.savez_compressed(out/f'randoms_{scenario_slug}.npz', abandonment=randoms)
        previous_scenario = pd.read_csv(STRENGTH/f'platform_results/scenarios_{scenario_slug}.csv.gz')
        assert df['product'].equals(previous_scenario['product'])
        for column in ['rep', 'listing', 'floor', 'value', 'opening', 'deadline', 'kind']:
            np.testing.assert_allclose(df[column], previous_scenario[column], rtol=0, atol=1e-12)
        with np.load(STRENGTH/f'platform_results/randoms_{scenario_slug}.npz') as original_randoms:
            np.testing.assert_array_equal(randoms, original_randoms['abandonment'])
        scenario_checks += 1
        for coefficient in COEFFICIENTS:
            for policy in POLICIES:
                start = time.perf_counter()
                blocks = []
                for seed in policy['seeds']:
                    if seed == -1:
                        base = pc.negotiation(df, randoms, cfg, policy['label'], rules[policy['label']], coefficient)
                        base['total_surplus'] = base.seller_surplus+base.buyer_surplus
                        prior = STRENGTH/f'platform_results/base_{scenario_slug}_{coefficient:g}_{slug(policy["label"])}.csv.gz'
                        previous = pd.read_csv(prior)
                        for field in ['deal', 'rounds', 'price_ratio', 'seller_surplus', 'buyer_surplus']:
                            np.testing.assert_allclose(base[field], previous[field], rtol=0, atol=1e-12, equal_nan=True)
                        rule_checks += 1
                    else:
                        bundle = bundles[(policy['key'], seed)]
                        base = learned_negotiation(df, randoms, cfg, bundle, coefficient)
                        if coefficient == .25:
                            prior = lc.rollout(df, randoms, cfg, lc.SPECS[policy['specification']], bundle=bundle)
                            for field in METRICS+['price_ratio']:
                                np.testing.assert_allclose(base[field], prior[field], rtol=0, atol=1e-12, equal_nan=True)
                            learned_checks += 1
                    base['policy'] = policy['label']
                    base['training_seed'] = seed
                    base['regime'] = regime
                    base['acceptance_coefficient'] = coefficient
                    check_base(base, 193*80)
                    seed_out.append(dict(regime=regime, acceptance_coefficient=coefficient, policy=policy['label'], training_seed=seed,
                                         **base[METRICS].mean().to_dict()))
                    blocks.append(base)
                merged = pd.concat(blocks, ignore_index=True).sort_values(['product', 'rep', 'training_seed']).reset_index(drop=True)
                check_base(merged, 193*80*len(policy['seeds']))
                # Each seed is a potential seller for the identical customer draw.
                matching = merged.groupby(['product', 'rep'])[['listing', 'floor', 'value', 'opening', 'deadline', 'kind']].nunique(dropna=False)
                assert matching.to_numpy().max() == 1
                filename = f'base_{scenario_slug}_a{coefficient:g}_{policy["key"]}.csv.gz'
                merged.to_csv(out/filename, index=False)
                product_rows = merged.groupby('product')[METRICS].mean().reset_index()
                product_rows['policy'] = policy['label']
                product_rows['regime'] = regime
                product_rows['acceptance_coefficient'] = coefficient
                products_out.append(product_rows)
                entry = dict(regime=regime, acceptance_coefficient=coefficient, policy=policy['label'], policy_key=policy['key'],
                             training_seeds=json.dumps(policy['seeds']), seed_count=len(policy['seeds']), base_file=filename,
                             rows=len(merged), products=193, opportunities_per_product=80, elapsed_seconds=time.perf_counter()-start,
                             sha256=sha(out/filename))
                manifests.append(entry)
                pd.DataFrame(manifests).to_csv(out/'trajectory_manifest.csv', index=False)
                pd.DataFrame(seed_out).to_csv(out/'seed_negotiation_outcomes.csv', index=False)
                pd.concat(products_out, ignore_index=True).to_csv(out/'product_negotiation_outcomes.csv', index=False)
                print('BASE COMPLETE', {k:entry[k] for k in ['regime', 'acceptance_coefficient', 'policy', 'base_file', 'rows', 'elapsed_seconds']}, flush=True)
    assert len(manifests) == 48 and sum(row['rows'] for row in manifests) == 2223360
    # Numerical source files and frozen checkpoints are never modified.
    for path, digest in {**source_hashes, **checkpoint_hashes}.items():
        assert sha(WORKSPACE/path) == digest
    summary = dict(protocol_sha256=protocol_hash, implementation_sha256=sha(Path(__file__)), source_sha256=source_hashes,
                   checkpoint_sha256=checkpoint_hashes, potential_trajectories=2223360, files=48, products=193,
                   opportunities_per_product=80, policy_instances_per_regime_coefficient=12, seeds=SEEDS,
                   regimes=REGIMES, acceptance_coefficients=sorted(COEFFICIENTS), stream=STREAM,
                   denominator='Original opportunity; failures retained. Learned policy families average over five fixed seller seeds, not independent customers.',
                   units='Surplus and ratio fields use listing-price units; listing retains recorded currency magnitude.',
                   rule_parameters={k:rules[k] for k in ['Time concession', 'Buyer responsive']},
                   checks=dict(disjoint_product_splits=True, scenario_equivalence_to_original=scenario_checks,
                               all_rule_cell_equivalence=rule_checks, learned_seed_regime_equivalence_at_point25=learned_checks,
                               bounds_cents_surplus_and_failure_invariants=True, source_and_checkpoint_immutability=True,
                               matched_customer_draws_all_policies=True),
                   elapsed_seconds=time.perf_counter()-begin,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__))
    write_json(out/'integration_summary.json', summary)
    print('COMPLETE', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    run(parser.parse_args().output)
