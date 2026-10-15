"""Checkout-relative analysis of fixed policy trajectories; no simulator retuning.

The equations are conditional accounting identities, not an endogenous-choice
model. JSON/CSV outputs are scientific numerical intermediates, not workbooks.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
POLICIES = ['Time concession', 'Buyer responsive', 'Full-state FQI', 'Clock-only FQI']
SELECTIONS = ['Exogenous', 'Value-dependent odds']
COEFS = ['a_s', 'a_b', 'u_s', 'u_b', 'h', 'new_margin', 'discount', 'a_d', 'u_d', 'mass',
         'gross_s', 'gross_b', 'baseline_s', 'baseline_b']
SEEDS = [271, 811, 1601, 2027, 4093]
EPS = 1e-12


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(p, selection):
    if selection == 'Exogenous': return p, p
    assert selection == 'Value-dependent odds'
    return 2*p/(1+p), p/(2-p)


def route_oracle(frame, p, r, seller_cost, buyer_cost, selection):
    """Independent route-based receipts/value calculation, not surplus-mixture code."""
    low, high = probabilities(p, selection)
    value = frame.value.to_numpy(); floor = frame.floor.to_numpy()
    eligible = value >= 1.; deal = frame.deal.to_numpy().astype(bool)
    price = np.where(deal, frame.price_ratio.to_numpy(), 0.)
    q = np.where(eligible, high, low)
    routes = np.column_stack([(1-q)*eligible, q*deal, q*(~deal)*r*eligible])
    purchase_probability = routes.sum(axis=1)
    assert ((purchase_probability >= -EPS) & (purchase_probability <= 1+EPS)).all()
    receipts = routes[:, 0]+routes[:, 1]*price+routes[:, 2]
    proxy_expense = floor*purchase_probability
    gross_customer_value = value*purchase_probability
    seller_net = receipts-proxy_expense-seller_cost*q*frame['rounds'].to_numpy()
    buyer_net = gross_customer_value-receipts-buyer_cost*q*frame['rounds'].to_numpy()
    checkout_seller = eligible*(1-floor)
    checkout_buyer = eligible*(value-1)
    expected_joint = (value-floor)*purchase_probability-(seller_cost+buyer_cost)*q*frame['rounds'].to_numpy()
    assert np.allclose(seller_net+buyer_net, expected_joint, atol=EPS, rtol=0)
    return np.column_stack([seller_net-checkout_seller, buyer_net-checkout_buyer,
                            purchase_probability-eligible, q*frame['rounds'].to_numpy()])


def raw_coefficients(frame):
    value = frame.value.to_numpy(); floor = frame.floor.to_numpy()
    deal = frame.deal.to_numpy().astype(bool); eligible = value >= 1.
    price = np.where(deal, frame.price_ratio.to_numpy(), 0.)
    seller = np.where(deal, price-floor, 0.); buyer = np.where(deal, value-price, 0.)
    s0 = eligible*(1-floor); b0 = eligible*(value-1)
    assert np.allclose(frame.seller_surplus, seller, atol=EPS, rtol=0)
    assert np.allclose(frame.buyer_surplus, buyer, atol=EPS, rtol=0)
    assert np.allclose(frame.total_surplus, seller+buyer, atol=EPS, rtol=0)
    assert np.isfinite(frame[['listing', 'floor', 'value', 'rounds']]).all().all()
    assert (frame.listing > 0).all() and ((frame.floor >= 0) & (frame.floor <= 1)).all()
    assert ((frame['rounds'] >= 1) & (frame['rounds'] <= frame.deadline)).all()
    assert np.all(seller >= -EPS) and np.all(buyer >= -EPS)
    assert np.all(price[deal] <= 1+EPS) and np.all(price[deal] >= floor[deal]-EPS)
    cents = price[deal]*frame.listing.to_numpy()[deal]*100
    assert np.allclose(cents, np.round(cents), atol=1e-5, rtol=0)
    data = dict(a_s=seller-s0, a_b=buyer-b0, u_s=(~deal)*s0, u_b=(~deal)*b0,
        h=frame['rounds'].to_numpy(), new_margin=(~eligible)*seller,
        discount=eligible*deal*(1-price), a_d=deal.astype(float)-eligible,
        u_d=(~deal)*eligible, mass=np.ones(len(frame)), gross_s=seller, gross_b=buyer,
        baseline_s=s0, baseline_b=b0)
    assert np.allclose(data['a_s'], data['new_margin']-data['discount']-data['u_s'], atol=EPS, rtol=0)
    return data, eligible


def aggregate_coefficients(frame):
    raw, eligible = raw_coefficients(frame)
    columns = {}
    for group, mask in [('low', ~eligible), ('high', eligible)]:
        columns.update({group+'_'+k: values*mask for k, values in raw.items()})
    table = pd.DataFrame(columns)
    table.insert(0, 'training_seed', frame.training_seed.to_numpy())
    table.insert(0, 'product', frame['product'].to_numpy())
    counts = table.groupby(['product', 'training_seed']).size()
    assert (counts == 80).all()
    # Both eligible groups keep the original opportunity denominator.
    by_seed = table.groupby(['product', 'training_seed']).mean()
    return by_seed.groupby('product').mean().sort_index(), raw, eligible


def weighted(frame, p, selection):
    low, high = probabilities(p, selection)
    return pd.DataFrame({k: low*frame['low_'+k]+high*frame['high_'+k] for k in COEFS}, index=frame.index)


def outcome(co, r, cost_s, cost_b):
    seller = co.a_s+r*co.u_s-cost_s*co.h
    buyer = co.a_b+r*co.u_b-cost_b*co.h
    return dict(seller_increment=seller.to_numpy(), buyer_increment=buyer.to_numpy(),
        joint_increment=(seller+buyer).to_numpy(), conversion_increment=(co.a_d+r*co.u_d).to_numpy(),
        negotiation_rounds=co.h.to_numpy(), effective_opt_in=co.mass.to_numpy(),
        new_margin=co.new_margin.to_numpy(), discount_cannibalization=co.discount.to_numpy(),
        lost_checkout_margin=((1-r)*co.u_s).to_numpy(), seller_interaction_cost=(cost_s*co.h).to_numpy())


def bootstrap_weights(n):
    draws = np.random.default_rng(20261002).integers(n, size=(2000, n))
    return np.stack([np.bincount(x, minlength=n)/n for x in draws])


def ci(values, weights):
    values = np.asarray(values, dtype=float)
    bootstrap = weights @ values
    return dict(mean=float(values.mean()), lo=float(np.quantile(bootstrap, .025)),
                hi=float(np.quantile(bootstrap, .975)), clusters=len(values))


def ratio_ci(numerator, denominator, weights):
    numerator = np.asarray(numerator, dtype=float); denominator = np.asarray(denominator, dtype=float)
    average_d = float(denominator.mean())
    point = float(numerator.mean()/average_d) if abs(average_d) > EPS else None
    bn, bd = weights@numerator, weights@denominator
    valid = np.abs(bd) > EPS
    ratios = bn[valid]/bd[valid]
    return dict(value=point, lo=float(np.quantile(ratios, .025)) if len(ratios) else None,
        hi=float(np.quantile(ratios, .975)) if len(ratios) else None,
        undefined_bootstrap_fraction=float(1-valid.mean()), denominator=average_d,
        denominator_positive_fraction=float((bd > EPS).mean()),
        denominator_negative_fraction=float((bd < -EPS).mean()),
        denominator_sign_instability=bool((bd > EPS).mean() > .025 and (bd < -EPS).mean() > .025))


def fallback_status(a, u, h, cost):
    if abs(u) <= EPS:
        return 'fallback_irrelevant_already_nonnegative' if a-cost*h >= -EPS else 'fallback_irrelevant_infeasible'
    threshold = (cost*h-a)/u
    if threshold <= 0: return 'no_fallback_required'
    if threshold <= 1: return 'feasible_fallback_threshold'
    return 'infeasible_even_at_full_fallback'


def inspect_base(path, paired_latents):
    frame = pd.read_csv(path).sort_values(['product', 'training_seed', 'rep']).reset_index(drop=True)
    assert not frame.duplicated(['product', 'rep', 'training_seed']).any()
    policy = frame.policy.unique().tolist(); regime = frame.regime.unique().tolist()
    coefficient = frame.acceptance_coefficient.unique().tolist()
    assert len(policy) == len(regime) == len(coefficient) == 1
    expected = [-1] if policy[0] in POLICIES[:2] else SEEDS
    assert sorted(frame.training_seed.unique()) == expected
    assert frame['product'].nunique() == 193 and len(frame) == 193*80*len(expected)
    latent_columns = ['product', 'rep', 'listing', 'floor', 'value', 'opening', 'deadline', 'kind']
    unique = frame[latent_columns].drop_duplicates().sort_values(['product', 'rep']).reset_index(drop=True)
    assert len(unique) == 193*80
    pair_key = (regime[0], coefficient[0])
    if pair_key not in paired_latents: paired_latents[pair_key] = unique
    else: pd.testing.assert_frame_equal(unique, paired_latents[pair_key], check_exact=False, atol=EPS, rtol=0)
    return frame, (regime[0], float(coefficient[0]), policy[0])


def run(inputs, out, allow_partial=False):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = inputs/'trajectory_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    assert not manifest.base_file.duplicated().any()
    files = [inputs/name for name in sorted(manifest.base_file)]
    if not allow_partial: assert len(files) == 48, f'Expected 48 completed base files, found {len(files)}'
    assert files, 'No integrated trajectories found'
    cohorts = {}; paired_latents = {}; provenance = {}; verified_rows = 0
    products = None
    for path in files:
        frame, key = inspect_base(path, paired_latents)
        assert key not in cohorts
        co, raw, eligible = aggregate_coefficients(frame)
        if products is None: products = co.index
        else: assert products.equals(co.index)
        cohorts[key] = co
        provenance[path.name] = dict(sha256=sha(path), rows=len(frame))
        recorded = manifest.loc[manifest.base_file == path.name].iloc[0]
        assert provenance[path.name]['sha256'] == recorded.sha256 and len(frame) == recorded.rows
        for selection in SELECTIONS:
            q_low, q_high = probabilities(.5, selection)
            q = np.where(eligible, q_high, q_low)
            implied = np.column_stack([q*(raw['a_s']+.5*raw['u_s']-.005*raw['h']),
                q*(raw['a_b']+.5*raw['u_b']-.005*raw['h']),
                q*(raw['a_d']+.5*raw['u_d']), q*raw['h']])
            oracle = route_oracle(frame, .5, .5, .005, .005, selection)
            assert np.allclose(implied, oracle, atol=EPS, rtol=0)
        verified_rows += len(frame)
        print('COEFFICIENTS', key, len(frame), flush=True)
    weights = bootstrap_weights(len(products))
    coefficient_frames = []
    grid_rows = []; primary_rows = []; threshold_rows = []; composition_rows = []
    primary_cache = {}
    for key, base in cohorts.items():
        regime, acceptance, policy = key
        identifying = dict(regime=regime, acceptance_coefficient=acceptance, policy=policy)
        cf = base.reset_index().assign(**identifying); coefficient_frames.append(cf)
        for selection, p, r, cost in itertools.product(SELECTIONS, [.25, .5, .75], [0., .5, 1.], [0., .005, .02]):
            co = weighted(base, p, selection)
            values = outcome(co, r, cost, cost)
            row = dict(**identifying, selection=selection, base_opt_in=p, fallback=r, cost=cost)
            row.update({metric: float(value.mean()) for metric, value in values.items()})
            grid_rows.append(row)
        for selection in SELECTIONS:
            co = weighted(base, .5, selection)
            primary_cache[(regime, acceptance, policy, selection)] = outcome(co, .5, .005, .005)
            for metric, values in primary_cache[(regime, acceptance, policy, selection)].items():
                primary_rows.append(dict(**identifying, selection=selection, metric=metric, **ci(values, weights)))
            for r in [0., .5, 1.]:
                seller_n = co.a_s+r*co.u_s; buyer_n = co.a_b+r*co.u_b
                for party, numerator in [('seller', seller_n), ('buyer', buyer_n),
                                          ('joint_equal_cost', (seller_n+buyer_n)/2)]:
                    threshold_rows.append(dict(**identifying, selection=selection, fallback=r,
                        threshold='maximum_cost', stakeholder=party, **ratio_ci(numerator, co.h, weights)))
                bootstrap_s = (weights@seller_n.to_numpy())/(weights@co.h.to_numpy())
                bootstrap_b = (weights@buyer_n.to_numpy())/(weights@co.h.to_numpy())
                both = np.minimum(bootstrap_s, bootstrap_b)
                threshold_rows.append(dict(**identifying, selection=selection, fallback=r,
                    threshold='maximum_cost', stakeholder='both_average_participants',
                    value=float(min(seller_n.mean(), buyer_n.mean())/co.h.mean()),
                    lo=float(np.quantile(both, .025)), hi=float(np.quantile(both, .975)),
                    undefined_bootstrap_fraction=0., denominator=float(co.h.mean())))
            for party in ['s', 'b']:
                ratio = ratio_ci(.005*co.h-co['a_'+party], co['u_'+party], weights)
                threshold_rows.append(dict(**identifying, selection=selection, cost=.005,
                    threshold='minimum_fallback', stakeholder='seller' if party == 's' else 'buyer',
                    status=fallback_status(co['a_'+party].mean(), co['u_'+party].mean(), co.h.mean(), .005), **ratio))
        # Actual composition crossing, not an invented exogenous uptake threshold.
        for party in ['s', 'b']:
            low = base['low_a_'+party]+.5*base['low_u_'+party]-.005*base.low_h
            high = base['high_a_'+party]+.5*base['high_u_'+party]-.005*base.high_h
            denominator = 2*low-high; numerator = 4*low+high
            ratio = ratio_ci(numerator, denominator, weights)
            point = ratio['value']
            if point is not None and EPS < point < 1-EPS:
                q_low, q_high = probabilities(point, 'Value-dependent odds')
                assert abs(q_low*low.mean()+q_high*high.mean()) < 1e-10
            composition_rows.append(dict(**identifying, stakeholder='seller' if party == 's' else 'buyer',
                threshold='base_probability_composition_crossing', feasible=point is not None and 0 < point < 1,
                low_contribution=float(low.mean()), high_contribution=float(high.mean()), **ratio))
        # Exogenous scaling is exact at every product, not just overall means.
        for p in [.25, .75]:
            small = outcome(weighted(base, p, 'Exogenous'), .5, .005, .005)
            full = outcome(weighted(base, 1., 'Exogenous'), .5, .005, .005)
            for metric in small: assert np.allclose(small[metric], p*full[metric], atol=EPS, rtol=0)
    pd.concat(coefficient_frames).to_csv(out/'product_coefficients.csv.gz', index=False)
    grid = pd.DataFrame(grid_rows); grid.to_csv(out/'grid_point_estimates.csv', index=False)
    pd.DataFrame(primary_rows).to_csv(out/'primary_intervals.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(out/'thresholds.csv', index=False)
    pd.DataFrame(composition_rows).to_csv(out/'composition_thresholds.csv', index=False)
    contrasts = []; crossings = []; rankings = []; mechanisms = []; negative_controls = 0
    conditions = sorted({(k[0], k[1]) for k in cohorts})
    for regime, acceptance in conditions:
        for selection in SELECTIONS:
            rank_rows = []
            for policy in POLICIES:
                base_key = (regime, acceptance, policy)
                if base_key not in cohorts: continue
                base = cohorts[base_key]
                gross = float((base.low_gross_s+base.high_gross_s).mean())
                net = float(primary_cache[(*base_key, selection)]['seller_increment'].mean())
                rank_rows.append(dict(regime=regime, acceptance_coefficient=acceptance,
                    selection=selection, policy=policy, conditional_gross_seller=gross,
                    primary_seller_increment=net))
            ranked = pd.DataFrame(rank_rows)
            ranked['gross_rank'] = ranked.conditional_gross_seller.rank(ascending=False, method='min').astype(int)
            ranked['channel_rank'] = ranked.primary_seller_increment.rank(ascending=False, method='min').astype(int)
            rankings.extend(ranked.to_dict('records'))
            for left, right in [('Full-state FQI', 'Time concession'), ('Full-state FQI', 'Clock-only FQI'),
                                ('Full-state FQI', 'Buyer responsive')]:
                lkey = (regime, acceptance, left, selection); rkey = (regime, acceptance, right, selection)
                if lkey not in primary_cache or rkey not in primary_cache: continue
                for metric in ['seller_increment', 'buyer_increment', 'joint_increment', 'conversion_increment']:
                    contrasts.append(dict(regime=regime, acceptance_coefficient=acceptance, selection=selection,
                        left=left, right=right, metric=metric,
                        **ci(primary_cache[lkey][metric]-primary_cache[rkey][metric], weights)))
            for left, right in itertools.combinations(POLICIES, 2):
                if (regime, acceptance, left) not in cohorts or (regime, acceptance, right) not in cohorts: continue
                a = weighted(cohorts[(regime, acceptance, left)], .5, selection)
                b = weighted(cohorts[(regime, acceptance, right)], .5, selection)
                delta_gross = a.a_s+.5*a.u_s-b.a_s-.5*b.u_s
                delta_h = a.h-b.h
                abase = cohorts[(regime, acceptance, left)]; bbase = cohorts[(regime, acceptance, right)]
                unweighted_gap = (abase.low_gross_s+abase.high_gross_s-bbase.low_gross_s-bbase.high_gross_s)
                weighted_gap = a.gross_s-b.gross_s
                composition_term = weighted_gap-.5*unweighted_gap
                fallback_term = .5*(a.u_s-b.u_s)
                cost_term = -.005*delta_h
                net_gap = primary_cache[(regime, acceptance, left, selection)]['seller_increment']-primary_cache[(regime, acceptance, right, selection)]['seller_increment']
                assert np.allclose(net_gap, .5*unweighted_gap+composition_term+fallback_term+cost_term, atol=EPS, rtol=0)
                mechanisms.append(dict(regime=regime, acceptance_coefficient=acceptance, selection=selection,
                    left=left, right=right, conditional_gross_difference=float(unweighted_gap.mean()),
                    constant_opt_in_gross_difference=float(.5*unweighted_gap.mean()),
                    selection_composition_adjustment=float(composition_term.mean()),
                    fallback_adjustment=float(fallback_term.mean()),
                    interaction_cost_adjustment=float(cost_term.mean()),
                    primary_net_difference=float(np.mean(net_gap))))
                threshold = ratio_ci(delta_gross, delta_h, weights)
                if selection == 'Exogenous':
                    # The common baseline cancels: its subtraction cannot reverse rankings.
                    assert np.allclose(a.baseline_s, b.baseline_s, atol=EPS, rtol=0)
                    assert np.allclose(a.a_s-b.a_s, a.gross_s-b.gross_s, atol=EPS, rtol=0)
                    negative_controls += 1
                value = threshold['value']
                crossings.append(dict(regime=regime, acceptance_coefficient=acceptance,
                    selection=selection, left=left, right=right,
                    crossing_within_evaluated_cost_range=value is not None and 0 < value < .02,
                    gross_difference=float(delta_gross.mean()), rounds_difference=float(delta_h.mean()),
                    **threshold))
    pd.DataFrame(contrasts).to_csv(out/'paired_policy_contrasts.csv', index=False)
    pd.DataFrame(crossings).to_csv(out/'policy_cost_crossings.csv', index=False)
    pd.DataFrame(rankings).to_csv(out/'descriptive_rankings.csv', index=False)
    pd.DataFrame(mechanisms).to_csv(out/'ranking_mechanisms.csv', index=False)
    # Descriptive finite-set hindsight loss: checkout is always an available zero increment.
    context = ['regime', 'acceptance_coefficient', 'selection', 'base_opt_in', 'fallback', 'cost']
    best = grid.groupby(context).seller_increment.max().clip(lower=0).rename('best_mean_increment').reset_index()
    regret = grid.merge(best, on=context, validate='many_to_one')
    regret['hindsight_loss'] = regret.best_mean_increment-regret.seller_increment
    checkout = best.assign(policy='Checkout only', seller_increment=0.)
    checkout['hindsight_loss'] = checkout.best_mean_increment
    regret = pd.concat([regret, checkout], ignore_index=True)
    regret.to_csv(out/'descriptive_hindsight_loss.csv', index=False)
    robust = regret.groupby(['selection', 'base_opt_in', 'fallback', 'cost', 'policy']).agg(
        worst_condition_increment=('seller_increment', 'min'), max_hindsight_loss=('hindsight_loss', 'max'),
        mean_hindsight_loss=('hindsight_loss', 'mean')).reset_index()
    robust.to_csv(out/'finite_condition_summary.csv', index=False)
    summary = dict(protocol_sha256=sha(ROOT/'protocol.json'), implementation_sha256=sha(Path(__file__)),
        input_manifest_sha256=sha(manifest_path),
        input_files=provenance, potential_trajectories=verified_rows, conditions=len(conditions),
        products=193, repetitions=80, learned_training_seeds=SEEDS, partial=allow_partial,
        primary_design=dict(opt_in=.5, fallback=.5, seller_cost=.005, buyer_cost=.005),
        primary_results=[x for x in primary_rows if x['acceptance_coefficient'] == .25],
        primary_policy_contrasts=[x for x in contrasts if x['acceptance_coefficient'] == .25],
        main_thresholds=[x for x in threshold_rows if x['acceptance_coefficient'] == .25 and x['selection'] == 'Exogenous'
            and (x['threshold'] == 'minimum_fallback' or x.get('fallback') == .5)],
        checks=dict(route_oracle_rows_per_selection=verified_rows, selections_checked=2,
            source_surplus_recomputed=True, receipt_value_cancellation=True, monetary_bounds=True,
            matched_latent_opportunities=True, original_opportunity_denominator=True,
            exogenous_opt_in_scaling=True, negative_control_policy_pairs=negative_controls,
            common_baseline_subtraction_cannot_reverse_order=True, no_policy_retuning=True),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        limitations='Same-team analysis and stipulated behavioral/cost/selection assumptions. Conditional thresholds and descriptive hindsight loss are not adoption estimates, new theory, an optimized selector, individual Pareto improvement, or field validation.')
    dump(out/'summary.json', summary)
    print('COMPLETE', verified_rows, 'potential trajectories;', len(grid_rows), 'point-estimate designs', flush=True)
    print(pd.DataFrame(primary_rows).query("acceptance_coefficient == .25 and selection == 'Exogenous' and metric == 'seller_increment'")
        [['regime', 'policy', 'mean', 'lo', 'hi']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=ROOT.parent/'integration/results')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(); run(args.inputs, args.output, args.allow_partial)
