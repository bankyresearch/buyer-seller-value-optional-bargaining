"""Complete numeric comparison of alternate-directory trajectory generation."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def main(original, repeat):
    first = pd.read_csv(original/'trajectory_manifest.csv')
    second = pd.read_csv(repeat/'trajectory_manifest.csv')
    assert first.base_file.to_list() == second.base_file.to_list()
    checked, largest = 0, 0.
    for name in first.base_file:
        a, b = pd.read_csv(original/name), pd.read_csv(repeat/name)
        assert a.columns.to_list() == b.columns.to_list() and len(a) == len(b)
        for column in a:
            if pd.api.types.is_numeric_dtype(a[column]):
                np.testing.assert_allclose(a[column], b[column], rtol=0, atol=1e-12, equal_nan=True)
                error = np.abs(a[column].to_numpy()-b[column].to_numpy())
                if np.isfinite(error).any():
                    largest = max(largest, float(np.nanmax(error)))
            else:
                assert a[column].equals(b[column])
        checked += len(a)
    for regime in ['reference', 'stubborn_offers', 'low_valuations']:
        with np.load(original/f'randoms_{regime}.npz') as a, np.load(repeat/f'randoms_{regime}.npz') as b:
            np.testing.assert_array_equal(a['abandonment'], b['abandonment'])
    x, y = [json.loads((p/'integration_summary.json').read_text()) for p in [original, repeat]]
    for key in ['protocol_sha256', 'implementation_sha256', 'source_sha256', 'checkpoint_sha256', 'potential_trajectories', 'checks', 'rule_parameters']:
        assert x[key] == y[key]
    result = dict(status='passed', rows_compared=checked, files_compared=len(first), tolerance_absolute=1e-12,
                  maximum_numeric_difference=largest, random_draw_arrays='exact',
                  scope='Full same-environment alternate-directory trajectory replay, not external researcher replication.',
                  excluded='Elapsed time and gzip metadata')
    (original/'reproduction_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--repeat', type=Path, default=Path(__file__).resolve().parent/'reproduction')
    args = parser.parse_args()
    main(args.original, args.repeat)
