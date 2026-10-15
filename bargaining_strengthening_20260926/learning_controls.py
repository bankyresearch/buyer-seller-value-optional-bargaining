"""Locally frozen exploratory controls; imports unchanged public-data simulator.

Python JSON/CSV numerical intermediates. No human or field observations created.
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
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'bargaining_framework_revision'
sys.path.insert(0, str(OLD))
import experiment as legacy

SEEDS = [271, 811, 1601, 2027, 4093]
SPECS = {
    'uniform_grid_32': dict(exploration='uniform', actions='grid', reps=32, mode='full'),
    'mixed_grid_32': dict(exploration='mixed', actions='grid', reps=32, mode='full'),
    'uniform_enriched_32': dict(exploration='uniform', actions='enriched', reps=32, mode='full'),
    'mixed_enriched_32': dict(exploration='mixed', actions='enriched', reps=32, mode='full'),
    'mixed_enriched_128': dict(exploration='mixed', actions='enriched', reps=128, mode='full'),
    'clock_mixed_enriched_128': dict(exploration='mixed', actions='enriched', reps=128, mode='clock'),
}
METRICS = ['deal', 'seller_surplus', 'buyer_surplus', 'total_surplus', 'rounds']
KEYS = ['state', 'action', 'reward', 'successor', 'terminal', 'episode', 'round']


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def data_key(spec):
    return f"{spec['exploration']}_{spec['actions']}_{spec['reps']}"


def candidates(state, action_set):
    a = np.broadcast_to(legacy.ACTIONS, (len(state), 9)).copy()
    if action_set == 'enriched':
        floor, prev = state[:, 2].astype(float), state[:, 4].astype(float)
        clock = np.clip(1. - (1. - floor) * state[:, 0], floor, prev)
        fraction = np.divide(prev-clock, prev-floor, out=np.zeros(len(state)), where=prev-floor > 1e-12)
        a = np.column_stack([a, np.clip(fraction, 0, 1)])
    return a


def features(state, fraction, mode):
    x = state if mode == 'full' else state[:, legacy.CLOCK_COLUMNS]
    return np.column_stack([x, fraction]).astype(np.float32)


def q_values(model, state, mode, action_set):
    actions = candidates(state, action_set)
    repeated = np.repeat(state, actions.shape[1], axis=0)
    q = model.predict(features(repeated, actions.ravel(), mode)).reshape(actions.shape)
    return np.clip(q, 0, 1. - state[:, 2:3])


def quote_candidates(t, horizon, floor, previous, action_set):
    asks = previous[:, None] - legacy.ACTIONS[None, :] * (previous-floor)[:, None]
    if action_set == 'enriched':
        exact = np.maximum(floor, np.minimum(previous, 1.-(1.-floor)*(t+1)/horizon))
        asks = np.column_stack([asks, exact])
    return asks


def rollout(df, randoms, cfg, spec, bundle=None, training_seed=None, trace=False):
    """Same customer dynamics and rounding as legacy; controlled action/exploration."""
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
    transitions, trace_rows = [], []
    rng = np.random.default_rng(training_seed)
    training = training_seed is not None
    for t in range(cfg['horizon']):
        if not alive.any():
            break
        bid = legacy.customer_bid(t, df, cfg, previous_ask, previous_bid)
        state = legacy.public_state(t, cfg['horizon'], floor, bid, previous_ask, previous_bid, minimum, maximum)
        ids = np.flatnonzero(alive)
        available_asks = quote_candidates(t, cfg['horizon'], floor[ids], previous_ask[ids], spec['actions'])
        if training:
            action = rng.integers(0, available_asks.shape[1], len(ids))
            if spec['exploration'] == 'mixed':
                episode_number = df.rep.to_numpy()[ids]
                component = ((episode_number // 4)+(episode_number % 4)) % 4
                terminal = np.where(component == 1, .5, np.where(component == 2, 1., 0.))
                target = np.clip(1.-(1.-floor[ids])*terminal*(t+1)/cfg['horizon'], floor[ids], previous_ask[ids])
                target_action = np.abs(available_asks-target[:, None]).argmin(axis=1)
                follow = (rng.random(len(ids)) < .75) & (component != 0)
                action = np.where(follow, target_action, action)
        else:
            action = q_values(bundle['model'], state[ids], spec['mode'], spec['actions']).argmax(axis=1)
        ask = previous_ask.copy()
        ask[ids] = available_asks[np.arange(len(ids)), action]
        # Use exact fractions for the first nine actions to preserve legacy fitting.
        fraction = legacy.ACTIONS[np.minimum(action, 8)].copy()
        dynamic = action == 9
        if dynamic.any():
            gap = previous_ask[ids]-floor[ids]
            exact_fraction = np.divide(previous_ask[ids]-ask[ids], gap, out=np.zeros(len(ids)), where=gap > 1e-12)
            fraction[dynamic] = np.clip(exact_fraction[dynamic], 0, 1)
        ask = np.maximum(floor, np.minimum(previous_ask, np.ceil(ask*listing*100-1e-9)/(listing*100)))
        assert np.all(ask[alive] >= floor[alive]-1e-10)
        assert np.all(ask[alive] <= previous_ask[alive]+1e-10)
        assert np.max(np.abs(ask[alive]*listing[alive]*100-np.round(ask[alive]*listing[alive]*100))) < 1e-5
        seller_accept = alive & (bid >= ask-1e-12)
        buyer_accept = alive & ~seller_accept & (ask <= bid+.25*np.maximum(value-bid, 0)+1e-12) & (ask <= value+1e-12)
        agreed = seller_accept | buyer_accept
        price[seller_accept] = bid[seller_accept]
        price[buyer_accept] = ask[buyer_accept]
        deal |= agreed
        rounds[alive] = t+1
        dropout = cfg['dropout']
        if cfg['ask_dropout']:
            dropout = np.clip(dropout+.4*np.maximum(ask-bid, 0), 0, .5)
        next_alive = alive & ~agreed & (t+1 < df.deadline.to_numpy()) & (randoms[:, t] >= dropout) & (t+1 < cfg['horizon'])
        minimum = np.minimum(minimum, bid)
        maximum = np.maximum(maximum, bid)
        if training:
            next_bid = legacy.customer_bid(t+1, df, cfg, ask, bid)
            successor = legacy.public_state(t+1, cfg['horizon'], floor, next_bid, ask, bid, minimum, maximum)
            reward = np.where(agreed[ids], price[ids]-floor[ids], 0)
            transitions.append((state[ids], fraction, reward, successor[ids], ~next_alive[ids], ids, np.full(len(ids), t)))
        if trace:
            trace_rows.append(pd.DataFrame(dict(episode=ids, round=t+1, action_index=action,
                                               action_fraction=fraction, previous_ask=previous_ask[ids], ask=ask[ids], bid=bid[ids], floor=floor[ids])))
        alive = next_alive
        previous_ask = ask
        previous_bid = bid
    if training:
        return tuple(np.concatenate([x[k] for x in transitions]) for k in range(7))
    result = df[['product', 'rep', 'category', 'kind', 'listing', 'floor', 'value', 'deadline']].copy()
    result['floor'] = floor
    result['deal'] = deal.astype(int)
    result['rounds'] = rounds
    result['price_ratio'] = price
    result['seller_surplus'] = np.where(deal, price-floor, 0)
    result['buyer_surplus'] = np.where(deal, value-price, 0)
    result['total_surplus'] = result.seller_surplus+result.buyer_surplus
    assert result.seller_surplus.min() >= -1e-9 and result.buyer_surplus.min() >= -1e-9
    assert np.allclose(result.loc[deal, 'total_surplus'], value[deal]-floor[deal])
    assert not result.duplicated(['product', 'rep']).any()
    if trace:
        return result, pd.concat(trace_rows, ignore_index=True)
    return result


def fit(transitions, spec, leaf, seed):
    state, action, reward, successor, terminal, _, _ = transitions
    x = features(state, action, spec['mode'])
    target = reward.copy()
    model = None
    for iteration in range(8):
        if model is not None:
            target = reward.copy()
            target[~terminal] += q_values(model, successor[~terminal], spec['mode'], spec['actions']).max(axis=1)
            target = np.clip(target, 0, 1.-state[:, 2])
        model = ExtraTreesRegressor(n_estimators=24, max_depth=12, min_samples_leaf=leaf, random_state=seed, n_jobs=2)
        model.fit(x, target)
    return dict(model=model, mode=spec['mode'], actions=spec['actions'], spec=spec, leaf=leaf, seed=seed,
                state_names=legacy.STATE_NAMES, clock_columns=legacy.CLOCK_COLUMNS)


def coverage(transitions, spec, seed):
    state = transitions[0]
    rows = []
    for t in range(8):
        s = state[transitions[6] == t]
        expected = 1.-(1.-s[:, 2])*t/8
        close = np.abs(s[:, 4]-expected) <= .02
        rows.append(dict(training=data_key(spec), seed=seed, round=t+1, transitions=len(s), near_linear=int(close.sum()),
                         near_linear_fraction=float(close.mean()) if len(s) else None,
                         previous_ask_mean=float(s[:, 4].mean()) if len(s) else None))
    return rows


def train(out, spec_names, seeds):
    products = pd.read_csv(legacy.DATA/'amazon_products.csv')
    sets = [set(products.loc[products.split == s, 'product']) for s in ['train', 'dev', 'test']]
    assert not any(sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)])
    dev = legacy.scenarios(products[products.split == 'dev'], 40, stream='development')
    scores_path = out/'candidate_scores.csv'
    scores = pd.read_csv(scores_path).to_dict('records') if scores_path.exists() else []
    completed = {(r['specification'], int(r['seed']), int(r['leaf'])) for r in scores}
    coverage_path = out/'training_coverage.csv'
    coverage_rows = pd.read_csv(coverage_path).to_dict('records') if coverage_path.exists() else []
    coverage_done = {(r['training'], int(r['seed'])) for r in coverage_rows}
    for name in spec_names:
        spec = SPECS[name]
        for seed in seeds:
            if all((name, seed, leaf) in completed for leaf in [40, 160]):
                continue
            key = data_key(spec)
            tr_path = out/f'transitions_{key}_{seed}.npz'
            if tr_path.exists():
                with np.load(tr_path) as z:
                    transitions = tuple(z[k] for k in KEYS)
            else:
                scenario = legacy.scenarios(products[products.split == 'train'], spec['reps'], stream='training-'+str(seed))
                transitions = rollout(*scenario, spec, training_seed=seed)
                np.savez_compressed(tr_path, **dict(zip(KEYS, transitions)))
                scenario[0].to_csv(out/f'training_scenarios_{key}_{seed}.csv.gz', index=False)
            if (key, seed) not in coverage_done:
                coverage_rows.extend(coverage(transitions, spec, seed))
                pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
                coverage_done.add((key, seed))
            for leaf in [40, 160]:
                if (name, seed, leaf) in completed:
                    continue
                started = time.perf_counter()
                bundle = fit(transitions, spec, leaf, seed)
                fit_seconds = time.perf_counter()-started
                checkpoint = out/f'fqi_{name}_{leaf}_{seed}.joblib'
                joblib.dump(bundle, checkpoint, compress=3)
                result = rollout(*dev, spec, bundle=bundle)
                score = float(result.groupby('product').seller_surplus.mean().mean())
                row = dict(specification=name, seed=seed, leaf=leaf, transitions=len(transitions[0]),
                           seller_surplus=score, fit_seconds=fit_seconds, checkpoint_bytes=checkpoint.stat().st_size)
                scores.append(row)
                pd.DataFrame(scores).to_csv(scores_path, index=False)
                print('CANDIDATE', row, flush=True)
    full_scores = pd.DataFrame(scores)
    selected = {}
    for name in SPECS:
        subset = full_scores[full_scores.specification == name]
        if len(subset) == len(SEEDS)*2 and set(subset.seed) == set(SEEDS):
            selected[name] = int(subset.groupby('leaf').seller_surplus.mean().idxmax())
    settings = dict(specifications=SPECS, selected_leaf=selected, seeds=SEEDS, trees=24, depth=12, iterations=8,
                    development_reps=40, evaluation_reps=80, training_stream='revision-v1:training-SEED',
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                    protocol_sha256=hashlib.sha256((ROOT/'LEARNING_PROTOCOL.md').read_bytes()).hexdigest(),
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/'frozen_settings.json', settings)
    print('SELECTED', selected, flush=True)


def evaluate(out):
    settings = json.loads((out/'frozen_settings.json').read_text())
    assert set(settings['selected_leaf']) == set(SPECS), 'Finish every prespecified training candidate first.'
    products = pd.read_csv(legacy.DATA/'amazon_products.csv').query("split == 'test'")
    rules = json.loads((OLD/'results/selected_rules.json').read_text())
    imitation = joblib.load(legacy.BASE/'results/Gradient_boosting.joblib')
    summaries, contrasts, seed_rows, product_rows = {}, {}, [], []
    total = 0
    for regime in legacy.REGIMES:
        scenario = legacy.scenarios(products, 80, regime)
        records = []
        names = list(SPECS) if regime == 'Reference' else ['mixed_enriched_128', 'clock_mixed_enriched_128']
        for label, params in {**rules, 'Posted-price checkout': {}, 'Offer imitation': {}}.items():
            r = legacy.rollout(*scenario, label, params, imitation=imitation)
            r['training_seed'] = -1
            records.append(r)
        for name in names:
            spec = SPECS[name]
            for seed in SEEDS:
                bundle = joblib.load(out/f'fqi_{name}_{settings["selected_leaf"][name]}_{seed}.joblib')
                r = rollout(*scenario, spec, bundle=bundle)
                r['policy'] = name
                r['training_seed'] = seed
                records.append(r)
                seed_rows.append(dict(regime=regime, specification=name, seed=seed, **r[METRICS].mean().to_dict()))
        all_rows = pd.concat(records, ignore_index=True)
        all_rows['regime'] = regime
        all_rows.to_csv(out/('episodes_'+regime.lower().replace(' ', '_')+'.csv.gz'), index=False)
        total += len(all_rows)
        grouped = all_rows.groupby(['policy', 'product'])[METRICS].mean()
        product_rows.append(grouped.reset_index().assign(regime=regime))
        for name, g in grouped.groupby(level=0):
            summaries[regime+'|'+name] = {m: legacy.interval(g[m]) for m in METRICS}
        pairs = [('mixed_enriched_128', 'Time concession'), ('mixed_enriched_128', 'clock_mixed_enriched_128'),
                 ('mixed_enriched_128', 'Offer imitation'), ('mixed_enriched_128', 'Posted-price checkout')]
        if regime == 'Reference':
            pairs += [('mixed_grid_32', 'uniform_grid_32'), ('mixed_enriched_32', 'uniform_enriched_32'),
                      ('uniform_enriched_32', 'uniform_grid_32'), ('mixed_enriched_32', 'mixed_grid_32'),
                      ('mixed_enriched_128', 'mixed_enriched_32')]
            pairs += [(name, 'Time concession') for name in names if name != 'mixed_enriched_128']
        for left, right in pairs:
            delta = grouped.loc[left]-grouped.loc[right]
            contrasts[regime+'|'+left+' minus '+right] = {m: legacy.interval(delta[m]) for m in METRICS}
        print('EVALUATED', regime, {p: round(summaries[regime+'|'+p]['seller_surplus']['mean'], 6) for p in grouped.index.get_level_values(0).unique()}, flush=True)
    pd.concat(product_rows, ignore_index=True).to_csv(out/'product_outcomes.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seed_sensitivity.csv', index=False)
    write_json(out/'summary.json', dict(test_products=len(products), seeds=SEEDS, total_trajectories=total,
                                      reference_policy_instances=36, stress_policy_instances=16,
                                      primary='Reference|mixed_enriched_128 minus Time concession',
                                      interval_scope='2,000 paired product resamples after repetitions and five training seeds averaged within product; conditional on simulator and fixed seeds',
                                      summary=summaries, contrasts=contrasts))
    print('COMPLETE', total, flush=True)


def verify(out):
    checks = {}
    for seed in SEEDS[:3]:
        with np.load(out/f'transitions_uniform_grid_32_{seed}.npz') as new, np.load(OLD/f'results/transitions_{seed}.npz') as old:
            for key in KEYS:
                target = legacy.ACTIONS[old[key]] if key == 'action' else old[key]
                np.testing.assert_array_equal(new[key], target)
    checks['legacy_training_arrays_shared_three_seeds'] = 'exact'
    # Candidate containment includes the time rule at every round and merchant floor.
    rng = np.random.default_rng(719)
    floor = rng.uniform(.2, .95, 5000)
    previous = rng.uniform(floor, 1.)
    for horizon in [4, 8]:
        for t in range(horizon):
            candidates_ask = quote_candidates(t, horizon, floor, previous, 'enriched')
            expected = legacy.propose('Time concession', dict(terminal=1., exponent=1.), t, horizon, floor, np.zeros(5000), previous)
            np.testing.assert_array_equal(candidates_ask[:, 9], expected)
    checks['enriched_exact_linear_quote_containment'] = '60,000 states passed'
    for reps in [32, 128]:
        episode = np.arange(reps)
        component = ((episode // 4)+(episode % 4)) % 4
        balanced = pd.crosstab(episode % 4, component).to_numpy()
        assert np.all(balanced == reps//16)
    checks['exploration_component_orthogonal_to_customer_style'] = 'each of 16 cells has 2 or 8 trajectories per training product'
    # Clock features must be unaffected by changes to every excluded buyer feature.
    state = rng.uniform(0, 1, (100, 9)).astype(np.float32)
    changed = state.copy()
    changed[:, [3, 5, 6, 7, 8]] += 10
    np.testing.assert_array_equal(features(state, np.zeros(100), 'clock'), features(changed, np.zeros(100), 'clock'))
    np.testing.assert_array_equal(candidates(state, 'enriched'), candidates(changed, 'enriched'))
    checks['clock_feature_and_action_exclusion'] = 'passed'
    settings = json.loads((out/'frozen_settings.json').read_text())
    products = pd.read_csv(legacy.DATA/'amazon_products.csv').query("split == 'test'")
    scenario = legacy.scenarios(products.iloc[:3], 80)
    total_replayed = 0
    reference = pd.read_csv(out/'episodes_reference.csv.gz')
    for name in SPECS:
        for seed in SEEDS:
            bundle = joblib.load(out/f'fqi_{name}_{settings["selected_leaf"][name]}_{seed}.joblib')
            r = rollout(*scenario, SPECS[name], bundle=bundle)
            saved = reference[(reference.policy == name) & (reference.training_seed == seed) & reference['product'].isin(products.iloc[:3]['product'])]
            saved = saved.sort_values(['product', 'rep'])
            r = r.sort_values(['product', 'rep'])
            np.testing.assert_allclose(r[METRICS+['price_ratio']], saved[METRICS+['price_ratio']], atol=1e-12, rtol=0, equal_nan=True)
            total_replayed += len(r)
    checks['selected_checkpoint_reference_replay_trajectories'] = total_replayed
    count = 0
    for file in out.glob('episodes_*.csv.gz'):
        frame = pd.read_csv(file)
        assert not frame.duplicated(['product', 'rep', 'policy', 'training_seed']).any()
        assert frame.seller_surplus.min() >= -1e-9 and frame.buyer_surplus.min() >= -1e-9
        np.testing.assert_allclose(frame.seller_surplus+frame.buyer_surplus, frame.total_surplus, atol=1e-12, rtol=0)
        agreed = frame.deal == 1
        np.testing.assert_allclose(frame.loc[agreed, 'total_surplus'], frame.loc[agreed, 'value']-frame.loc[agreed, 'floor'], atol=1e-12, rtol=0)
        prices = frame.loc[agreed, 'price_ratio']*frame.loc[agreed, 'listing']*100
        assert (np.abs(prices-np.round(prices)) < 1e-5).all()
        count += len(frame)
    checks['all_trajectory_invariants_passed'] = count
    summary = json.loads((out/'summary.json').read_text())
    assert count == summary['total_trajectories']
    costs = pd.read_csv(out/'candidate_scores.csv')
    checks['candidate_count'] = len(costs)
    assert len(costs) == 60
    checks['fit_seconds_total'] = float(costs.fit_seconds.sum())
    checks['checkpoint_bytes_all_candidates'] = int(costs.checkpoint_bytes.sum())
    checks['failure_count'] = 0
    write_json(out/'verification.json', checks)
    print('VERIFIED', checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=ROOT/'learning_results')
    parser.add_argument('--stage', choices=['train', 'evaluate', 'verify', 'all'], default='all')
    parser.add_argument('--specs', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ['train', 'all']:
        train(args.out_dir, args.specs, args.seeds)
    if args.stage in ['evaluate', 'all']:
        evaluate(args.out_dir)
    if args.stage in ['verify', 'all']:
        verify(args.out_dir)
