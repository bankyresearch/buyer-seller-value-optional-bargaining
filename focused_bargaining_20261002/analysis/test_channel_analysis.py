"""Independent accounting fixtures and boundary tests; no simulator imports."""
from pathlib import Path
import argparse
import json
import unittest
import numpy as np
import pandas as pd
import channel_analysis as ca


def fixture():
    return pd.DataFrame(dict(product=['x']*4, rep=list(range(4)), training_seed=[-1]*4,
        listing=[100.]*4, floor=[.7, .7, .6, .6], value=[1.2, 1.1, .9, .8],
        deal=[True, False, True, False], price_ratio=[.9, np.nan, .8, np.nan],
        rounds=[2, 3, 4, 5], deadline=[6]*4,
        seller_surplus=[.2, 0., .2, 0.], buyer_surplus=[.3, 0., .1, 0.],
        total_surplus=[.5, 0., .3, 0.]))


def analytical_example():
    data = pd.DataFrame(dict(value=[.95, .95], floor=[.7, .7], deal=[True, True],
        price_ratio=[.9, .88], rounds=[6, 1]))
    oracle = ca.route_oracle(data, .5, 0., .005, .005, 'Exogenous')
    return dict(label='Constructed analytical illustration, not observed results',
        listing_units=True, opt_in=.5, fallback=0., cost_per_round=.005,
        common_floor=.7, common_value=.95, prices=[.9, .88], rounds=[6, 1],
        gross_seller=[.2, .18], incremental_seller=oracle[:, 0].tolist(),
        exact_cost_crossing=.004,
        interpretation='A higher conditional gross margin is reversed by its five extra rounds; subtraction of a common checkout baseline alone cannot cause a reversal.')


class AccountingTests(unittest.TestCase):
    def test_manual_route_enumeration_with_unequal_costs(self):
        expected = np.array([[-.06, .03, 0., 1.], [-.09, -.055, -.25, 1.5],
                             [.08, .01, .5, 2.], [-.025, -.05, 0., 2.5]])
        np.testing.assert_allclose(ca.route_oracle(fixture(), .5, .5, .01, .02, 'Exogenous'),
                                   expected, atol=1e-12, rtol=0)

    def test_no_opt_in_is_checkout(self):
        for selection in ca.SELECTIONS:
            np.testing.assert_allclose(ca.route_oracle(fixture(), 0., .8, .02, .03, selection), 0., atol=1e-12)

    def test_full_opt_in_no_fallback_is_potential_trajectory(self):
        frame = fixture(); raw, _ = ca.raw_coefficients(frame)
        actual = ca.route_oracle(frame, 1., 0., 0., 0., 'Exogenous')
        np.testing.assert_allclose(actual, np.column_stack([raw['a_s'], raw['a_b'], raw['a_d'], raw['h']]), atol=1e-12)

    def test_full_fallback_restores_failed_checkout_only(self):
        delta = ca.route_oracle(fixture(), .5, 1., 0., 0., 'Exogenous')-ca.route_oracle(fixture(), .5, 0., 0., 0., 'Exogenous')
        np.testing.assert_allclose(delta, [[0, 0, 0, 0], [.15, .05, .5, 0], [0, 0, 0, 0], [0, 0, 0, 0]], atol=1e-12)

    def test_decomposition_all_bounded_fixture_combinations(self):
        frame = fixture(); raw, eligible = ca.raw_coefficients(frame)
        for selection in ca.SELECTIONS:
            for p in [0., .25, .5, .75, 1.]:
                low, high = ca.probabilities(p, selection); q = np.where(eligible, high, low)
                for r in [0., .5, 1.]:
                    for cost in [0., .005, .02]:
                        actual = ca.route_oracle(frame, p, r, cost, cost*2, selection)
                        expected = np.column_stack([q*(raw['a_s']+r*raw['u_s']-cost*raw['h']),
                            q*(raw['a_b']+r*raw['u_b']-2*cost*raw['h']),
                            q*(raw['a_d']+r*raw['u_d']), q*raw['h']])
                        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_group_aggregation_uses_original_opportunity_denominator(self):
        repeated = pd.concat([fixture() for _ in range(20)], ignore_index=True)
        repeated['rep'] = np.arange(80)
        first = repeated.assign(training_seed=271)
        second = repeated.assign(training_seed=811)
        co, _, _ = ca.aggregate_coefficients(pd.concat([first, second], ignore_index=True))
        self.assertAlmostEqual(co.low_mass.iloc[0], .5)
        self.assertAlmostEqual(co.high_mass.iloc[0], .5)
        self.assertAlmostEqual(co.low_gross_s.iloc[0], .05)
        self.assertAlmostEqual(co.high_gross_s.iloc[0], .05)
        self.assertAlmostEqual(ca.weighted(co, .5, 'Exogenous').mass.iloc[0], .5)

    def test_cost_ceiling_exact_root_and_sign(self):
        a, u, h, r = -.01, .04, 2., .5
        ceiling = (a+r*u)/h
        self.assertAlmostEqual(ceiling, .005)
        self.assertAlmostEqual(a+r*u-ceiling*h, 0.)
        self.assertGreater(a+r*u-(ceiling-.001)*h, 0.)
        self.assertLess(a+r*u-(ceiling+.001)*h, 0.)

    def test_fallback_floor_exact_root(self):
        a, u, h, c = -.01, .04, 2., .005
        minimum = (c*h-a)/u
        self.assertAlmostEqual(minimum, .5)
        self.assertAlmostEqual(a+minimum*u-c*h, 0.)

    def test_fallback_feasibility_labels(self):
        self.assertEqual(ca.fallback_status(.1, 0., 2., .005), 'fallback_irrelevant_already_nonnegative')
        self.assertEqual(ca.fallback_status(-.1, 0., 2., .005), 'fallback_irrelevant_infeasible')
        self.assertEqual(ca.fallback_status(.1, .1, 2., .005), 'no_fallback_required')
        self.assertEqual(ca.fallback_status(-.01, .04, 2., .005), 'feasible_fallback_threshold')
        self.assertEqual(ca.fallback_status(-.1, .01, 2., .005), 'infeasible_even_at_full_fallback')

    def test_ratio_uses_ratio_of_means_not_mean_ratios(self):
        result = ca.ratio_ci([1., 9.], [1., 3.], np.array([[.5, .5], [1., 0.], [0., 1.]]))
        self.assertAlmostEqual(result['value'], 2.5)
        self.assertNotEqual(result['value'], np.mean([1., 3.]))

    def test_undefined_ratio_retained(self):
        result = ca.ratio_ci([1., 2.], [0., 0.], np.array([[.5, .5], [1., 0.]]))
        self.assertIsNone(result['value']); self.assertIsNone(result['lo'])
        self.assertEqual(result['undefined_bootstrap_fraction'], 1.)

    def test_ratio_sign_instability_is_explicit(self):
        result = ca.ratio_ci([1., 2.], [-1., 2.], np.array([[.5, .5], [1., 0.], [0., 1.]]))
        self.assertTrue(result['denominator_sign_instability'])
        self.assertAlmostEqual(result['denominator_positive_fraction'], 2/3)
        self.assertAlmostEqual(result['denominator_negative_fraction'], 1/3)

    def test_composition_crossing_root_and_sign(self):
        low, high = .01, -.03
        point = (4*low+high)/(2*low-high)
        self.assertAlmostEqual(point, .2)
        def value(p):
            ql, qh = ca.probabilities(p, 'Value-dependent odds')
            return ql*low+qh*high
        self.assertAlmostEqual(value(point), 0.)
        self.assertGreater(value(.1), 0.); self.assertLess(value(.5), 0.)

    def test_constant_opt_in_cannot_change_sign_or_policy_order(self):
        s0 = np.array([.3, .2, 0., 0.]); a = np.array([.25, .15, .2, .1]); b = np.array([.2, .1, .15, .09])
        for p in [.1, .25, .5, .75, 1.]:
            np.testing.assert_allclose(p*(a-s0)-p*(b-s0), p*(a-b), atol=1e-12)
            self.assertGreater(np.mean(p*(a-s0)-p*(b-s0)), 0.)

    def test_constructed_cost_ranking_reversal(self):
        ex = analytical_example()
        self.assertGreater(ex['gross_seller'][0], ex['gross_seller'][1])
        np.testing.assert_allclose(ex['incremental_seller'], [.085, .0875], atol=1e-12)
        self.assertLess(ex['incremental_seller'][0], ex['incremental_seller'][1])
        difference = .5*(.2-.18)-.004*.5*(6-1)
        self.assertAlmostEqual(difference, 0.)

    def test_paired_bootstrap_is_repeatable(self):
        a = ca.bootstrap_weights(193); b = ca.bootstrap_weights(193)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a.sum(axis=1), 1., atol=1e-12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AccountingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        successful=result.wasSuccessful(), fixture_grid_cases=90,
        constructed_example=analytical_example(),
        distinction='Same-team independent accounting tests, not independent external scientific validation.')
    ca.dump(args.output/'accounting_tests.json', report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
