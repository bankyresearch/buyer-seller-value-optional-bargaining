"""Cross-cell checks on complete saved trajectories, separate from generator."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def main(out):
    manifest = pd.read_csv(out/'trajectory_manifest.csv')
    assert len(manifest) == 48 and manifest.rows.sum() == 2223360
    monotone_pairs, common_latent_checks = 0, 0
    for regime, regime_rows in manifest.groupby('regime'):
        scenario = pd.read_csv(out/f'scenarios_{regime.lower().replace(" ", "_")}.csv.gz').set_index(['product', 'rep'])
        for _, policy_rows in regime_rows.groupby('policy'):
            previous = None
            for row in policy_rows.sort_values('acceptance_coefficient').itertuples():
                current = pd.read_csv(out/row.base_file).sort_values(['product', 'rep', 'training_seed']).reset_index(drop=True)
                reference = scenario.loc[pd.MultiIndex.from_frame(current[['product', 'rep']])]
                for column in ['listing', 'value', 'opening', 'deadline', 'kind']:
                    np.testing.assert_allclose(current[column], reference[column], rtol=0, atol=1e-12)
                rounded_floor = np.ceil(reference.floor.to_numpy()*reference.listing.to_numpy()*100-1e-9)/(reference.listing.to_numpy()*100)
                np.testing.assert_allclose(current.floor, rounded_floor, rtol=0, atol=1e-12)
                assert len(current) == row.rows and current['product'].nunique() == 193
                assert not current.duplicated(['product', 'rep', 'training_seed']).any()
                common_latent_checks += len(current)
                if previous is not None:
                    assert current[['product', 'rep', 'training_seed']].equals(previous[['product', 'rep', 'training_seed']])
                    assert (current.deal >= previous.deal).all()
                    assert (current['rounds'] <= previous['rounds']).all()
                    monotone_pairs += len(current)
                previous = current
    result = dict(status='passed', complete_base_rows=common_latent_checks, adjacent_acceptance_monotonicity_pairs=monotone_pairs,
                  checks=['All potential trajectories use exactly the corresponding recorded latent scenario',
                          'No duplicate episode key or discarded product',
                          'Higher acceptance coefficient cannot reduce agreement or increase attempted rounds under identical paths until first acceptance',
                          'Executable merchant floor is exactly the cent-rounded scenario proxy'])
    (out/'grid_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    main(parser.parse_args().output)
