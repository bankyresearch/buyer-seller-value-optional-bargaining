"""Synthetic opt-in, checkout fallback and per-round cost sensitivity.

The stored base trajectories plus design_grid.csv reconstruct every expected
per-opportunity outcome. All choice/fallback parameters are assumptions.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'multidataset_bargaining'
REV = ROOT.parent / 'bargaining_framework_revision'
sys.path.insert(0, str(REV))
sys.path.insert(0, str(BASE))
from experiment import scenarios, customer_bid, rollout as legacy_rollout
from simulate import propose

METRICS = ['agreement', 'seller_gross', 'buyer_gross', 'seller_net', 'buyer_net',
           'joint_net', 'negotiation_rounds', 'opt_in', 'new_sales_margin',
           'discount_cannibalization', 'lost_checkout_margin',
           'seller_incremental_gross', 'seller_incremental_net', 'buyer_incremental_net']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def negotiation(df, randoms, cfg, policy, params, coefficient):
    """Frozen observable rule, with a configurable customer acceptance coefficient."""
    n = len(df); listing = df.listing.values; value = df.value.values
    floor = np.ceil(df.floor.values*listing*100-1e-9)/(listing*100)
    previous_ask = np.ones(n); previous_bid = df.opening.values.copy()
    alive = np.ones(n, bool); deal = np.zeros(n, bool)
    price = np.full(n, np.nan); rounds = np.zeros(n, int)
    for t in range(cfg['horizon']):
        if not alive.any(): break
        bid = customer_bid(t, df, cfg, previous_ask, previous_bid)
        ask = propose(policy, params, t, cfg['horizon'], floor, bid, previous_ask)
        ask = np.maximum(floor, np.minimum(previous_ask, np.ceil(ask*listing*100-1e-9)/(listing*100)))
        assert np.all(ask[alive] >= floor[alive]-1e-12)
        assert np.all(ask[alive] <= previous_ask[alive]+1e-12)
        seller_accept = alive & (bid >= ask-1e-12)
        buyer_accept = alive & ~seller_accept & (ask <= bid+coefficient*np.maximum(value-bid, 0)+1e-12) & (ask <= value+1e-12)
        agreed = seller_accept | buyer_accept
        price[seller_accept] = bid[seller_accept]; price[buyer_accept] = ask[buyer_accept]
        deal |= agreed; rounds[alive] = t+1
        alive &= ~agreed & (t+1 < df.deadline.values) & (randoms[:, t] >= cfg['dropout']) & (t+1 < cfg['horizon'])
        previous_ask = ask; previous_bid = bid
    result = df.copy(); result['floor'] = floor; result['deal'] = deal.astype(int)
    result['rounds'] = rounds; result['price_ratio'] = price
    result['seller_surplus'] = np.where(deal, price-floor, 0.)
    result['buyer_surplus'] = np.where(deal, value-price, 0.)
    assert (result.seller_surplus >= -1e-10).all() and (result.buyer_surplus >= -1e-10).all()
    assert np.allclose((price[deal]*listing[deal]*100), np.round(price[deal]*listing[deal]*100), atol=1e-7, rtol=0)
    assert np.allclose((result.seller_surplus+result.buyer_surplus)[deal], (value-floor)[deal])
    return result


def choice_probability(value, probability, mechanism):
    if mechanism == 'Exogenous': return np.full(len(value), probability)
    odds = probability/(1-probability)*np.where(value < 1., 2., .5)
    return odds/(1+odds)


def opportunity_outcomes(result, mode, probability=0., fallback=0., cost=0., selection='Exogenous'):
    """Probability-integrated mutually exclusive purchase routes, per initial opportunity."""
    v = result.value.to_numpy(); f = result.floor.to_numpy()
    d = result.deal.to_numpy().astype(bool); price = result.price_ratio.to_numpy()
    d0 = v >= 1.; s0 = np.where(d0, 1-f, 0.); b0 = np.maximum(v-1, 0.)
    if mode == 'Checkout only': q = np.zeros(len(result)); fallback = 0.
    elif mode == 'Forced negotiation': q = np.ones(len(result)); fallback = 0.
    else: q = choice_probability(v, probability, selection)
    direct = (1-q)*d0
    negotiation_sale = q*d
    fallback_sale = q*(~d)*fallback*d0
    agreement = direct+negotiation_sale+fallback_sale
    seller = (1-q)*s0+q*(result.seller_surplus.to_numpy()+(~d)*fallback*s0)
    buyer = (1-q)*b0+q*(result.buyer_surplus.to_numpy()+(~d)*fallback*b0)
    rounds = q*result['rounds'].to_numpy()
    burden = cost*rounds
    new_margin = q*(~d0)*result.seller_surplus.to_numpy()
    discount = q*d0*np.where(d, 1-price, 0.)
    lost_checkout = q*d0*(~d)*(1-fallback)*s0
    assert np.all((agreement >= -1e-12) & (agreement <= 1+1e-12))
    assert np.allclose(seller-s0, new_margin-discount-lost_checkout, atol=1e-12, rtol=0)
    seller_net = seller-burden; buyer_net = buyer-burden
    values = np.column_stack([agreement, seller, buyer, seller_net, buyer_net,
        seller_net+buyer_net, rounds, q, new_margin, discount, lost_checkout,
        seller-s0, seller_net-s0, buyer_net-b0])
    assert np.isfinite(values).all()
    return values


def designs():
    yield dict(mode='Checkout only', selection='None', base_opt_in=0., fallback=0., cost=0.)
    for cost in [0., .005, .02]:
        yield dict(mode='Forced negotiation', selection='All', base_opt_in=1., fallback=0., cost=cost)
    for selection, probability, fallback, cost in itertools.product(
            ['Exogenous', 'Value-dependent odds'], [.25, .5, .75], [0., .5, 1.], [0., .005, .02]):
        yield dict(mode='Choice enabled', selection=selection, base_opt_in=probability, fallback=fallback, cost=cost)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    sources = [BASE/'data/processed/amazon_products.csv', BASE/'data/processed/craigslist_dialogues.csv',
               REV/'results/selected_rules.json', REV/'experiment.py', BASE/'simulate.py']
    products = pd.read_csv(sources[0]).query("split == 'test'").sort_values('product').reset_index(drop=True)
    assert len(products) == 193 and not products['product'].duplicated().any()
    selected = json.loads(sources[2].read_text())
    all_grids, all_product_values, raw_rows = [], [], 0
    legacy_checks = 0
    for regime in ['Reference', 'Stubborn offers', 'Low valuations']:
        df, randoms, cfg = scenarios(products, 80, regime, stream='platform-controls-20260926')
        assert df[['product', 'rep']].drop_duplicates().shape[0] == 193*80
        assert list(df['product'].drop_duplicates()) == products['product'].tolist()
        scenario_key = regime.lower().replace(' ', '_')
        df.to_csv(out/f'scenarios_{scenario_key}.csv.gz', index=False)
        np.savez_compressed(out/f'randoms_{scenario_key}.npz', abandonment=randoms)
        for coefficient, policy in itertools.product([0., .25, .5, 1.], ['Time concession', 'Buyer responsive']):
            base = negotiation(df, randoms, cfg, policy, selected[policy], coefficient)
            if coefficient == .25:
                old = legacy_rollout(df, randoms, cfg, policy, selected[policy])
                for field in ['deal', 'rounds', 'price_ratio', 'seller_surplus', 'buyer_surplus']:
                    assert np.allclose(base[field], old[field], equal_nan=True, atol=1e-12, rtol=0), field
                legacy_checks += 1
            key = f'{scenario_key}_{coefficient:g}_{policy.lower().replace(" ", "_")}'
            base.to_csv(out/f'base_{key}.csv.gz', index=False)
            raw_rows += len(base)
            for design in designs():
                values = opportunity_outcomes(base, design['mode'], design['base_opt_in'],
                    design['fallback'], design['cost'], design['selection'])
                product_values = values.reshape(193, 80, len(METRICS)).mean(axis=1)
                all_product_values.append(product_values)
                all_grids.append(dict(design_id=len(all_grids), regime=regime,
                    acceptance_coefficient=coefficient, policy=policy, base_file=f'base_{key}.csv.gz', **design))
            print('PLATFORM EVALUATED', regime, coefficient, policy, flush=True)
    matrix = np.stack(all_product_values)  # designs, products, metrics
    grid = pd.DataFrame(all_grids)
    grid.to_csv(out/'design_grid.csv', index=False)
    np.savez_compressed(out/'product_outcomes.npz', values=matrix, products=products['product'].to_numpy(dtype=str),
                        metrics=np.asarray(METRICS), design_id=grid.design_id.to_numpy())
    # Product-level data table supports inspection and independent recalculation.
    product_frame = pd.DataFrame(matrix.reshape(-1, len(METRICS)), columns=METRICS)
    product_frame.insert(0, 'product', np.tile(products['product'].values, len(grid)))
    product_frame.insert(0, 'design_id', np.repeat(grid.design_id.values, 193))
    product_frame.to_csv(out/'product_outcomes.csv.gz', index=False)
    indices = np.random.default_rng(20260926).integers(193, size=(2000, 193))
    weights = np.zeros((2000, 193))
    for i in range(2000): weights[i] = np.bincount(indices[i], minlength=193)/193
    summary_rows = []
    for k, metric in enumerate(METRICS):
        estimates = matrix[:, :, k].mean(axis=1)
        bootstrap = weights @ matrix[:, :, k].T
        lo, hi = np.quantile(bootstrap, [.025, .975], axis=0)
        for i in range(len(grid)):
            summary_rows.append(dict(design_id=i, metric=metric, mean=float(estimates[i]),
                lo=float(lo[i]), hi=float(hi[i]), clusters=193))
    long_summary = pd.DataFrame(summary_rows).merge(grid, on='design_id', validate='many_to_one')
    long_summary.to_csv(out/'summary.csv', index=False)
    main = long_summary.query("acceptance_coefficient == .25 and ((mode == 'Choice enabled' and selection == 'Exogenous' and base_opt_in == .5 and fallback == .5 and cost == .005) or (mode == 'Forced negotiation' and cost == .005) or mode == 'Checkout only')")
    summary = dict(protocol_sha256=sha(ROOT/'PLATFORM_PROTOCOL.md'), source_sha256={str(p.relative_to(ROOT.parent)): sha(p) for p in sources},
        evaluation_products=193, opportunities_per_product=80, unique_potential_negotiation_trajectories=raw_rows,
        assumptions=['Reference', 'Stubborn offers', 'Low valuations'], acceptance_coefficients=[0., .25, .5, 1.],
        design_combinations=len(grid), policy_parameters={p:selected[p] for p in ['Time concession', 'Buyer responsive']},
        accounting='Expected outcomes integrate synthetic opt-in and fallback probabilities; combinations are not new independent trajectories. Costs charged separately to buyer and seller.',
        uncertainty='2000 product bootstrap draws; paired incremental metrics; conditional on fixed behavioral assumptions, not adoption uncertainty',
        primary_configuration='a=.25; p=.5; fallback=.5; cost=.005 per participant per negotiation round; exogenous opt-in',
        main_results=main.to_dict(orient='records'),
        checks=dict(legacy_coefficient_equivalence=legacy_checks, no_duplicate_opportunities=True,
            cent_rounding_and_bounds=True, mutually_exclusive_routes=True, cannibalization_reconciliation=True, no_nonfinite_outputs=True),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    print('COMPLETE', raw_rows, 'base trajectories;', len(grid), 'design combinations', flush=True)
    print(main.loc[main.metric.isin(['seller_incremental_net', 'buyer_incremental_net']),
        ['regime', 'policy', 'mode', 'metric', 'mean', 'lo', 'hi']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'platform_results')
    run(parser.parse_args().output)
