"""Exploratory reward-aligned extension. No customer or field data are generated as real observations."""
from pathlib import Path
import sys, json, hashlib, itertools, argparse, platform
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
import sklearn

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'multidataset_bargaining'
sys.path.insert(0, str(BASE))
from simulate import propose, REGIMES as ORIGINAL_REGIMES
from analyze import FEATURES
DATA = BASE / 'data/processed'
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [271, 811, 1601]
ACTIONS = np.linspace(0, 1, 9)
REGIMES = {**ORIGINAL_REGIMES,
    'Stubborn offers': {'behavior': 'stubborn'},
    'Last-round concessions': {'behavior': 'last'},
    'Oscillating offers': {'behavior': 'oscillating'},
    'Ask-sensitive abandonment': {'ask_dropout': True}}
STATE_NAMES = ['round_fraction', 'horizon_over_eight', 'merchant_floor', 'buyer_offer',
               'previous_ask', 'previous_bid', 'bid_change', 'minimum_bid', 'maximum_bid']
CLOCK_COLUMNS = [0, 1, 2, 4]
RULE_GRIDS = {
    'Uniform discount': [{'discount': x} for x in [0, .05, .1, .15, .2, .25, .3, .35, .4, .45]],
    'Time concession': [dict(terminal=q, exponent=b) for q, b in itertools.product([.25, .5, .75, 1.], [.5, 1., 2., 4.])],
    'Buyer responsive': [dict(terminal=q, exponent=b, weight=w, markup=m)
        for q, b, w, m in itertools.product([.5, .75, 1.], [1., 2.], [.25, .5, .75], [0., .1, .2])]}


def scenarios(products, reps, regime='Reference', stream='evaluation'):
    cfg = dict(dropout=.05, floor_factor=.8, vlow=.6, vhigh=1.15, horizon=8,
               behavior='reference', ask_dropout=False)
    cfg.update(REGIMES[regime])
    opening = pd.read_csv(DATA / 'craigslist_dialogues.csv').query("split == 'train'").buyer_first.dropna().clip(.3, .85).values
    rows, randoms = [], []
    for r in products.itertuples():
        seed = int(hashlib.sha256((r.product + ':revision-v1:' + stream).encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(seed)
        for j in range(reps):
            v = rng.uniform(cfg['vlow'], cfg['vhigh'])
            o = min(float(rng.choice(opening)), .9 * v)
            deadline = int(rng.choice([3, 5, 8]))
            u = rng.random(8)
            floor = float(np.clip(r.lowest / r.current * cfg['floor_factor'], .2, .95))
            rows.append((r.product, j, r.category, r.current, floor, v, o, deadline, j % 4))
            randoms.append(u)
    return pd.DataFrame(rows, columns=['product','rep','category','listing','floor','value','opening','deadline','kind']), np.stack(randoms), cfg


def customer_bid(t, df, cfg, previous_ask, previous_bid):
    value, opening, deadline, kind = [df[k].values for k in ['value','opening','deadline','kind']]
    progress = np.minimum(1., t / np.maximum(deadline - 1, 1))
    bid = opening + (value - opening) * progress ** np.choose(kind, [1., 3., .5, 1.])
    if t:
        bid = np.where(kind == 3, np.minimum(value, np.maximum(bid, previous_bid + .3 * np.maximum(previous_ask - previous_bid, 0))), bid)
    if cfg['behavior'] == 'stubborn': bid = opening.copy()
    if cfg['behavior'] == 'last': bid = np.where(t >= deadline - 1, value, opening)
    if cfg['behavior'] == 'oscillating' and t > 0 and t % 2 == 1: bid = np.maximum(0., bid - .10)
    return np.floor(np.minimum(bid, 1.) * df.listing.values * 100 + 1e-9) / (df.listing.values * 100)


def public_state(t, horizon, floor, bid, previous_ask, previous_bid, minimum, maximum):
    n = len(floor)
    return np.column_stack([np.full(n, (t + 1) / horizon), np.full(n, horizon / 8), floor,
        bid, previous_ask, previous_bid, bid - previous_bid, np.minimum(minimum, bid), np.maximum(maximum, bid)]).astype(np.float32)


def sa(state, action, mode):
    x = state if mode == 'full' else state[:, CLOCK_COLUMNS]
    return np.column_stack([x, ACTIONS[np.asarray(action, dtype=int)]]).astype(np.float32)


def q_values(model, state, mode):
    repeated = np.repeat(state, len(ACTIONS), axis=0)
    actions = np.tile(np.arange(len(ACTIONS)), len(state))
    q = model.predict(sa(repeated, actions, mode)).reshape(len(state), len(ACTIONS))
    return np.clip(q, 0., 1. - state[:, 2:3])


def learned_action(bundle, state):
    return q_values(bundle['model'], state, bundle['mode']).argmax(axis=1)


def rollout(df, randoms, cfg, policy, params=None, bundle=None, imitation=None, training_seed=None):
    params = params or {}
    n = len(df); listing = df.listing.values
    floor = np.ceil(df.floor.values * listing * 100 - 1e-9) / (listing * 100)
    value = df.value.values
    previous_ask = np.ones(n); previous_bid = df.opening.values.copy()
    minimum = np.full(n, np.inf); maximum = np.full(n, -np.inf)
    alive = np.ones(n, bool); deal = np.zeros(n, bool); price = np.full(n, np.nan); rounds = np.zeros(n, int)
    transitions = []; histories = []
    rng = np.random.default_rng(training_seed)
    if policy == 'Posted-price checkout':
        deal = value >= 1.; price[deal] = 1.; rounds[:] = 1
    else:
        for t in range(cfg['horizon']):
            if not alive.any(): break
            bid = customer_bid(t, df, cfg, previous_ask, previous_bid)
            state = public_state(t, cfg['horizon'], floor, bid, previous_ask, previous_bid, minimum, maximum)
            ids = np.flatnonzero(alive)
            if policy in ('Reward FQI', 'Clock-only FQI', 'Explore'):
                action = rng.integers(0, len(ACTIONS), len(ids)) if policy == 'Explore' else learned_action(bundle, state[ids])
                ask = previous_ask.copy()
                ask[ids] = previous_ask[ids] - ACTIONS[action] * (previous_ask[ids] - floor[ids])
            elif policy == 'Offer imitation':
                all_history = histories + [bid]
                X = pd.DataFrame({'log_listing': np.log(listing), 'last_own': previous_ask if t else np.ones(n),
                    'last_other': bid, 'last_public': bid, 'min_public': np.min(all_history, axis=0), 'max_public': np.max(all_history, axis=0),
                    'own_seen': int(t > 0), 'other_seen': 1, 'log_own_count': np.log1p(t), 'log_other_count': np.log1p(t+1),
                    'log_event': np.log1p(2*t+1), 'is_buyer': 0,
                    'category': ['electronics' if c == 'electronics' else 'unknown' for c in df.category]})
                ask = np.maximum(floor, np.minimum(previous_ask, np.clip(imitation.predict(X[FEATURES]), 0, 2)))
            else: ask = propose(policy, params, t, cfg['horizon'], floor, bid, previous_ask)
            ask = np.maximum(floor, np.minimum(previous_ask, np.ceil(ask * listing * 100 - 1e-9) / (listing * 100)))
            assert np.all(ask[alive] >= floor[alive] - 1e-10) and np.all(ask[alive] <= previous_ask[alive] + 1e-10)
            seller_accept = alive & (bid >= ask - 1e-12)
            buyer_accept = alive & ~seller_accept & (ask <= bid + .25 * np.maximum(value - bid, 0) + 1e-12) & (ask <= value + 1e-12)
            agreed = seller_accept | buyer_accept
            price[seller_accept] = bid[seller_accept]; price[buyer_accept] = ask[buyer_accept]
            deal |= agreed; rounds[alive] = t + 1
            dropout = cfg['dropout']
            if cfg['ask_dropout']: dropout = np.clip(dropout + .40 * np.maximum(ask - bid, 0), 0, .50)
            next_alive = alive & ~agreed & (t + 1 < df.deadline.values) & (randoms[:, t] >= dropout) & (t + 1 < cfg['horizon'])
            minimum = np.minimum(minimum, bid); maximum = np.maximum(maximum, bid)
            if policy == 'Explore':
                next_bid = customer_bid(t+1, df, cfg, ask, bid)
                next_state = public_state(t+1, cfg['horizon'], floor, next_bid, ask, bid, minimum, maximum)
                reward = np.where(agreed[ids], price[ids] - floor[ids], 0.)
                transitions.append((state[ids], action, reward, next_state[ids], ~next_alive[ids], ids, np.full(len(ids), t)))
            alive = next_alive; previous_ask = ask; previous_bid = bid
            histories.extend([bid, ask])
    result = df[['product','rep','category','kind','listing','floor','value','deadline']].copy()
    result['floor'] = floor; result['policy'] = policy; result['deal'] = deal.astype(int); result['rounds'] = rounds; result['price_ratio'] = price
    result['seller_surplus'] = np.where(deal, price-floor, 0.)
    result['buyer_surplus'] = np.where(deal, value-price, 0.)
    result['total_surplus'] = result.seller_surplus + result.buyer_surplus
    assert result.seller_surplus.min() >= -1e-9 and result.buyer_surplus.min() >= -1e-9
    assert np.allclose(result.loc[deal,'total_surplus'], value[deal]-floor[deal])
    if policy == 'Explore': return tuple(np.concatenate([x[k] for x in transitions]) for k in range(7))
    return result


def fit(transitions, mode, leaf, seed):
    state, action, reward, successor, done, _, _ = transitions
    X = sa(state, action, mode); target = reward.copy(); model = None
    for iteration in range(8):
        if model is not None:
            target = reward.copy()
            target[~done] += q_values(model, successor[~done], mode).max(axis=1)
            target = np.clip(target, 0., 1. - state[:, 2])
        model = ExtraTreesRegressor(n_estimators=24, max_depth=12, min_samples_leaf=leaf, random_state=seed, n_jobs=2)
        model.fit(X, target)
    return dict(model=model, mode=mode, leaf=leaf, seed=seed, state_names=STATE_NAMES, action_fractions=ACTIONS.tolist())


def select_and_train():
    products = pd.read_csv(DATA / 'amazon_products.csv')
    sets = [set(products.loc[products.split == s,'product']) for s in ['train','dev','test']]
    assert not any(sets[a] & sets[b] for a,b in [(0,1),(0,2),(1,2)])
    dev = scenarios(products[products.split == 'dev'], 40, stream='development')
    selected = {'Fixed price': {}}; rule_scores = []
    for policy, grid in RULE_GRIDS.items():
        scores = [float(rollout(*dev, policy, p).groupby('product').seller_surplus.mean().mean()) for p in grid]
        selected[policy] = grid[int(np.argmax(scores))]
        rule_scores += [dict(policy=policy, parameters=json.dumps(p,sort_keys=True), seller_surplus=s) for p,s in zip(grid,scores)]
    pd.DataFrame(rule_scores).to_csv(OUT/'rule_selection.csv',index=False)
    (OUT/'selected_rules.json').write_text(json.dumps(selected,indent=2))
    scores = []
    for seed in SEEDS:
        train = scenarios(products[products.split == 'train'], 32, stream='training-'+str(seed))
        tr = rollout(*train, 'Explore', training_seed=seed)
        np.savez_compressed(OUT/f'transitions_{seed}.npz', **dict(zip(['state','action','reward','successor','terminal','episode','round'],tr)))
        train[0].to_csv(OUT/f'training_scenarios_{seed}.csv.gz', index=False)
        for mode, leaf in itertools.product(['full','clock'], [40,160]):
            bundle = fit(tr, mode, leaf, seed)
            joblib.dump(bundle, OUT/f'fqi_{mode}_{leaf}_{seed}.joblib',compress=3)
            result = rollout(*dev, 'Reward FQI', bundle=bundle)
            score = float(result.groupby('product').seller_surplus.mean().mean())
            scores.append(dict(seed=seed,mode=mode,leaf=leaf,transitions=len(tr[0]),seller_surplus=score))
            print('FQI development',scores[-1],flush=True)
    scores = pd.DataFrame(scores); scores.to_csv(OUT/'fqi_selection.csv',index=False)
    leaves = {mode: int(scores[scores['mode']==mode].groupby('leaf').seller_surplus.mean().idxmax()) for mode in ['full','clock']}
    protocol = dict(selected_rules=selected, selected_leaf=leaves, seeds=SEEDS, trees=24,max_depth=12,iterations=8,
                    train_reps=32,development_reps=40,test_reps=80,regimes=REGIMES,state=STATE_NAMES,clock_columns=CLOCK_COLUMNS,
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
                    protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest())
    (OUT/'frozen_settings.json').write_text(json.dumps(protocol,indent=2))
    print('FROZEN',leaves,flush=True)


def interval(values, seed=719):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)))


def evaluate():
    settings = json.loads((OUT/'frozen_settings.json').read_text())
    products = pd.read_csv(DATA/'amazon_products.csv').query("split == 'test'")
    imitation = joblib.load(BASE/'results/Gradient_boosting.joblib')
    bundles = {(mode,seed):joblib.load(OUT/f'fqi_{mode}_{settings["selected_leaf"][mode]}_{seed}.joblib') for mode,seed in itertools.product(['full','clock'],SEEDS)}
    summaries, contrasts, seed_scores = {}, {}, []; episode_count=0; product_rows=[]
    for regime in REGIMES:
        scenario = scenarios(products,80,regime)
        records = []
        for policy,params in {**settings['selected_rules'],'Posted-price checkout':{},'Offer imitation':{}}.items():
            result = rollout(*scenario,policy,params,imitation=imitation);result['training_seed']=-1;records.append(result)
        for mode,label in [('full','Reward FQI'),('clock','Clock-only FQI')]:
            for seed in SEEDS:
                result=rollout(*scenario,label,bundle=bundles[(mode,seed)])
                result['training_seed']=seed;records.append(result)
                seed_scores.append(dict(regime=regime,policy=label,seed=seed,**result[['deal','seller_surplus','buyer_surplus','rounds']].mean().to_dict()))
        all_results = pd.concat(records,ignore_index=True);all_results['regime']=regime
        filename='episodes_'+regime.lower().replace(' ','_')+'.csv.gz'
        all_results.to_csv(OUT/filename,index=False,compression='gzip');episode_count+=len(all_results)
        metrics=['deal','seller_surplus','buyer_surplus','total_surplus','rounds']
        grouped=all_results.groupby(['policy','product'])[metrics].mean()
        for policy,g in grouped.groupby(level=0):
            summaries[regime+'|'+policy]={m:interval(g[m]) for m in metrics}
        for comparator in ['Time concession','Offer imitation','Clock-only FQI','Posted-price checkout']:
            d=grouped.loc['Reward FQI']-grouped.loc[comparator]
            contrasts[regime+'|Reward FQI minus '+comparator]={m:interval(d[m]) for m in metrics}
        g=grouped.reset_index();g['regime']=regime
        g=g.merge(products[['product','category']],on='product');product_rows.append(g)
        print('EVALUATED',regime,{p:round(summaries[regime+'|'+p]['seller_surplus']['mean'],5) for p in grouped.index.get_level_values(0).unique()},flush=True)
    pd.concat(product_rows,ignore_index=True).to_csv(OUT/'product_outcomes.csv',index=False)
    pd.DataFrame(seed_scores).to_csv(OUT/'training_seed_sensitivity.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(test_products=len(products),scenarios_per_policy_regime=len(products)*80,
        policy_labels=8,trained_policy_seeds=3,regimes=len(REGIMES),total_trajectories=episode_count,
        bootstrap_unit='product; seed-averaged FQI; conditional on simulator and three fixed training seeds',summary=summaries,contrasts=contrasts),indent=2))
    print('COMPLETE',episode_count,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','evaluate','all'],default='all');args=parser.parse_args()
    if args.stage in ['train','all']:select_and_train()
    if args.stage in ['evaluate','all']:evaluate()
