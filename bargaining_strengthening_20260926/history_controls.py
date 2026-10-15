"""Matched native-offer targets, different strictly prior price histories.

Run with bundled Python and --output DIR for a separate replication.
No source inputs or previous results are edited.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'multidataset_bargaining'
REV = ROOT.parent / 'bargaining_framework_revision'
sys.path.insert(0, str(BASE))
from analyze import FEATURES, aggregate_error

SEED = 20260925
BOOT_SEED = 20260926
CONTEXT = ['log_listing', 'log_event', 'is_buyer', 'category']
KEYS = ['dialogue', 'event']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline(kind, parameter, features):
    prep = ColumnTransformer([
        ('numeric', StandardScaler(), [x for x in features if x != 'category']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['category'])])
    estimator = (Ridge(alpha=parameter) if kind == 'Ridge' else
        HistGradientBoostingRegressor(loss='absolute_error', max_iter=150,
            max_leaf_nodes=parameter, min_samples_leaf=25, l2_regularization=1.,
            random_state=SEED, early_stopping=False))
    return make_pipeline(prep, estimator)


def matched_inputs():
    paths = [BASE/'data/processed/craigslist_price_events.csv', REV/'results/structured_offer_events.csv']
    mixed, native = [pd.read_csv(p).sort_values(KEYS).reset_index(drop=True) for p in paths]
    mixed = mixed.loc[mixed.source_action == 'offer'].reset_index(drop=True)
    assert not mixed.duplicated(KEYS).any() and not native.duplicated(KEYS).any()
    assert mixed[KEYS].equals(native[KEYS])
    for key in ['scenario', 'split', 'category', 'role']:
        assert mixed[key].equals(native[key]), key
    for key in ['listing', 'target_ratio', 'log_listing', 'log_event', 'is_buyer']:
        assert np.allclose(mixed[key], native[key], atol=1e-14, rtol=0), key
    assert mixed.groupby('split').size().to_dict() == {'dev': 479, 'test': 546, 'train': 4482}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not set(mixed.loc[mixed.split == a, 'scenario']) & set(mixed.loc[mixed.split == b, 'scenario'])
    return mixed, native, paths


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(means, .025)),
                hi=float(np.quantile(means, .975)), clusters=len(values))


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    mixed, native, paths = matched_inputs()
    selection, predictions, errors, models = [], [], {}, {}
    variants = [('Mixed history', mixed, FEATURES), ('Native-only history', native, FEATURES),
                ('No-price-history context', native, CONTEXT)]
    for view, data, features in variants:
        data.to_csv(out/(view.replace(' ', '_')+'_features.csv.gz'), index=False)
        train, dev, test = [data.loc[data.split == s] for s in ['train', 'dev', 'test']]
        for kind, grid in [('Ridge', [.1, 1., 10., 100.]), ('Gradient boosting', [7, 15, 31])]:
            best = (np.inf, None, None)
            for parameter in grid:
                model = pipeline(kind, parameter, features).fit(train[features], train.target_ratio)
                development_prediction = np.clip(model.predict(dev[features]), 0, 2)
                score = float(aggregate_error(dev, np.abs(development_prediction-dev.target_ratio)).mean())
                selection.append(dict(view=view, model=kind, parameter=parameter, development_mae=score))
                if score < best[0]: best = (score, model, parameter)
            label = view+'|'+kind
            models[label] = dict(parameter=best[2], development_mae=best[0], features=features)
            joblib.dump(best[1], out/(label.replace(' ', '_').replace('|', '__')+'.joblib'), compress=3)
            prediction = np.clip(best[1].predict(test[features]), 0, 2)
            result = test[['dialogue', 'scenario', 'event', 'category', 'role', 'listing', 'target_ratio']].copy()
            result['view'] = view; result['model'] = kind; result['prediction'] = prediction
            predictions.append(result)
            errors[label] = aggregate_error(test, np.abs(prediction-test.target_ratio)).sort_index()
            print('FITTED', label, best[2], flush=True)
    test = mixed.loc[mixed.split == 'test']
    median = float(mixed.loc[mixed.split == 'train', 'target_ratio'].median())
    errors['Training median'] = aggregate_error(test, np.abs(median-test.target_ratio)).sort_index()
    error_frame = pd.DataFrame(errors)
    assert len(error_frame) == 371 and error_frame.notna().all().all()
    indices = np.random.default_rng(BOOT_SEED).integers(len(error_frame), size=(2000, len(error_frame)))
    metrics = {k: interval(v, indices) for k, v in errors.items()}
    contrasts = {}
    for kind in ['Ridge', 'Gradient boosting']:
        for comparator in ['Native-only history', 'No-price-history context']:
            label = kind+'|Mixed history minus '+comparator
            contrasts[label] = interval(errors['Mixed history|'+kind]-errors[comparator+'|'+kind], indices)
    pd.DataFrame(selection).to_csv(out/'development_selection.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'predictions.csv.gz', index=False)
    error_frame.to_csv(out/'scenario_errors.csv')
    summary = dict(protocol_sha256=sha(ROOT/'HISTORY_PROTOCOL.md'), rows={'train': 4482, 'dev': 479, 'test': 546},
        test_scenarios=371, source_sha256={str(p.relative_to(ROOT.parent)): sha(p) for p in paths},
        model_selection=models, metrics=metrics, paired_contrasts=contrasts,
        interval='2000 paired scenario percentile bootstrap; mean within dialogue then scenario',
        interpretation='Matched targets and splits isolate the predictive contribution of history representation. Not parser accuracy or causal evidence.',
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        checks=dict(unique_targets=True, exact_target_alignment=True, context_alignment=True, scenario_disjoint=True,
            identical_training_development_test_rows=True, identical_hyperparameter_protocol=True))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'metrics': metrics, 'paired_contrasts': contrasts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'history_results')
    run(parser.parse_args().output)
