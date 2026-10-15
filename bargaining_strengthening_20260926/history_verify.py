"""Verify every matched prior-history row against raw events, without refitting."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'multidataset_bargaining'
sys.path.insert(0, str(BASE))
from prepare_data import price
from history_controls import matched_inputs, FEATURES, CONTEXT


def run(out, raw, replication=None):
    mixed, native, _ = matched_inputs()
    lookups = [x.set_index(['dialogue', 'event']) for x in [mixed, native]]
    checked = 0; source_hashes = {}
    for split in ['train', 'dev', 'test']:
        path = raw/(split+'.json')
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        allowed = set(mixed.loc[mixed.split == split, 'dialogue']); seen = set()
        for dialogue in json.loads(path.read_text(encoding='utf8')):
            uid = hashlib.sha256(str(dialogue.get('uuid')).encode()).hexdigest()
            if uid not in allowed or uid in seen: continue
            seen.add(uid)
            listing = float(dialogue['scenario']['kbs'][0]['item']['Price'])
            roles = [k['personal']['Role'] for k in dialogue['scenario']['kbs']]
            histories = [[], []]; own = [{'buyer': [], 'seller': []}, {'buyer': [], 'seller': []}]
            for i, event in enumerate(dialogue['events']):
                actor = event.get('agent')
                if actor not in (0, 1): continue
                value = price(event)
                if value is None or not 0 <= value/listing <= 2: continue
                role = roles[actor]; other = 'seller' if role == 'buyer' else 'buyer'
                if event.get('action') == 'offer':
                    # Crucially verify BEFORE appending the current target.
                    for view in range(2):
                        row = lookups[view].loc[(uid, i)]
                        h = histories[view]; a = own[view][role]; b = own[view][other]
                        expected = [a[-1] if a else 1., b[-1] if b else 1., h[-1] if h else 1.,
                            min(h) if h else 1., max(h) if h else 1., int(bool(a)), int(bool(b)),
                            math.log1p(len(a)), math.log1p(len(b))]
                        actual = row[['last_own', 'last_other', 'last_public', 'min_public', 'max_public',
                                      'own_seen', 'other_seen', 'log_own_count', 'log_other_count']].to_numpy(float)
                        assert np.allclose(actual, expected, atol=1e-13, rtol=0), (uid, i, view)
                        assert abs(row.target_ratio-value/listing) < 1e-13
                    checked += 1
                for view in range(2):
                    if view == 1 and event.get('action') != 'offer': continue
                    histories[view].append(value/listing); own[view][role].append(value/listing)
    assert checked == len(native) == 5507
    predictions = pd.read_csv(out/'predictions.csv.gz')
    for view, features, data in [('Mixed history', FEATURES, mixed), ('Native-only history', FEATURES, native),
                                ('No-price-history context', CONTEXT, native)]:
        test = data.query("split == 'test'")
        for family in ['Ridge', 'Gradient boosting']:
            label = view+'|'+family
            model = joblib.load(out/(label.replace(' ', '_').replace('|', '__')+'.joblib'))
            expected = np.clip(model.predict(test[features]), 0, 2)
            saved = predictions.loc[(predictions['view'] == view) & (predictions['model'] == family)]
            assert np.allclose(expected, saved.prediction, atol=1e-12, rtol=0)
    result = dict(native_target_rows_verified=checked, historical_views=2, models_replayed=6,
        raw_source_sha256=source_hashes, strict_prior_event_order=True,
        scope='Same-team verification against original events, not an independent external audit.')
    if replication is not None:
        first = json.loads((out/'summary.json').read_text()); repeated = json.loads((replication/'summary.json').read_text())
        assert first == repeated
        for filename in ['development_selection.csv', 'predictions.csv.gz', 'scenario_errors.csv']:
            a, b = [pd.read_csv(directory/filename) for directory in [out, replication]]
            pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=0)
        result['separate_directory_reproduction'] = dict(full_retraining=True, summary_exact=True,
            compared_tables=3, prediction_rows=3276, numerical_absolute_tolerance=1e-12)
    (out/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=ROOT/'history_results')
    p.add_argument('--raw', type=Path, default=ROOT.parent/'research_paper_bm3il/data/raw/craigslist')
    p.add_argument('--replication', type=Path)
    a = p.parse_args(); run(a.output, a.raw, a.replication)
