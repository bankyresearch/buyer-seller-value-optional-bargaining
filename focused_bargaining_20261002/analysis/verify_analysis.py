"""Verify all analysis tables, prior-rule continuity and alternate-directory replay."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import channel_analysis as ca

ROOT = Path(__file__).resolve().parent


def verify(current, replay, old):
    compared = []
    for path in sorted(current.glob('*.csv*')):
        duplicate = replay/path.name
        assert duplicate.exists(), duplicate
        a, b = pd.read_csv(path), pd.read_csv(duplicate)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=0)
        compared.append(dict(file=path.name, rows=len(a)))
    a = json.loads((current/'summary.json').read_text())
    b = json.loads((replay/'summary.json').read_text())
    assert a == b
    grid = pd.read_csv(current/'grid_point_estimates.csv')
    assert len(grid) == 2592
    np.testing.assert_allclose(grid.seller_increment,
        grid.new_margin-grid.discount_cannibalization-grid.lost_checkout_margin-grid.seller_interaction_cost,
        atol=1e-12, rtol=0)
    np.testing.assert_allclose(grid.joint_increment, grid.seller_increment+grid.buyer_increment, atol=1e-12, rtol=0)
    np.testing.assert_allclose(grid.seller_interaction_cost, grid.cost*grid.negotiation_rounds, atol=1e-12, rtol=0)
    prior = pd.read_csv(old/'summary.csv').query("mode == 'Choice enabled'")
    key = ['regime', 'acceptance_coefficient', 'policy', 'selection', 'base_opt_in', 'fallback', 'cost']
    metrics = dict(seller_increment='seller_incremental_net', buyer_increment='buyer_incremental_net',
        negotiation_rounds='negotiation_rounds', effective_opt_in='opt_in', new_margin='new_sales_margin',
        discount_cannibalization='discount_cannibalization', lost_checkout_margin='lost_checkout_margin')
    continuity = 0
    for new_metric, old_metric in metrics.items():
        left = grid[grid.policy.isin(ca.POLICIES[:2])][key+[new_metric]]
        right = prior[prior.metric == old_metric][key+['mean']]
        joined = left.merge(right, on=key, validate='one_to_one')
        assert len(joined) == 1296
        np.testing.assert_allclose(joined[new_metric], joined['mean'], atol=1e-12, rtol=0)
        continuity += len(joined)
    mechanism = pd.read_csv(current/'ranking_mechanisms.csv')
    np.testing.assert_allclose(mechanism.primary_net_difference,
        mechanism.constant_opt_in_gross_difference+mechanism.selection_composition_adjustment+
        mechanism.fallback_adjustment+mechanism.interaction_cost_adjustment, atol=1e-12, rtol=0)
    np.testing.assert_allclose(mechanism.query("selection == 'Exogenous'").selection_composition_adjustment,
                              0., atol=1e-12, rtol=0)
    # Hindsight benchmark is an explicitly finite post-outcome descriptor.
    regret = pd.read_csv(current/'descriptive_hindsight_loss.csv')
    assert (regret.hindsight_loss >= -1e-12).all()
    assert (regret.best_mean_increment >= -1e-12).all()
    context = ['regime', 'acceptance_coefficient', 'selection', 'base_opt_in', 'fallback', 'cost']
    assert (regret.groupby(context).size() == 5).all()
    report = dict(status='passed', tolerance_absolute=1e-12, replay_tables=compared,
        replay_table_count=len(compared), replay_table_rows=sum(x['rows'] for x in compared),
        summary_json_equal=True, original_rule_point_estimates_verified=continuity,
        opportunity_accounting_designs=len(grid), ranking_mechanism_contrasts=len(mechanism),
        hindsight_contexts=regret.groupby(context).ngroups,
        route_oracle_trajectory_selection_pairs=2*a['potential_trajectories'],
        negative_control_policy_pairs=a['checks']['negative_control_policy_pairs'],
        statement='Independent accounting implementation within the same team and an exact same-environment analysis rerun; not external researcher replication.')
    ca.dump(current/'analysis_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', type=Path, default=ROOT/'results')
    parser.add_argument('--replay', type=Path, default=ROOT/'replication')
    parser.add_argument('--old', type=Path, default=ROOT.parent.parent/'bargaining_strengthening_20260926/platform_results')
    args = parser.parse_args(); verify(args.current, args.replay, args.old)
