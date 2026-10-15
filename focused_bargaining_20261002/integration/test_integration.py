"""Small deterministic contract tests; no human/customer study observations."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd

import generate_trajectories as gen


class IntegrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        products = pd.read_csv(gen.lc.legacy.DATA/'amazon_products.csv').query("split == 'test'").sort_values('product').iloc[:2]
        cls.scenario = gen.lc.legacy.scenarios(products, 12, 'Reference', stream=gen.STREAM)
        cls.bundles = {}
        for name in ['mixed_enriched_128', 'clock_mixed_enriched_128']:
            for seed in gen.SEEDS:
                cls.bundles[(name, seed)] = joblib.load(gen.STRENGTH/f'learning_results/fqi_{name}_40_{seed}.joblib')

    def test_protocol_and_full_denominator(self):
        protocol = json.loads((gen.ROOT/'protocol.json').read_text())
        self.assertEqual(protocol['expected_potential_trajectories'], 193*80*3*4*(2+2*5))
        self.assertEqual(protocol['acceptance_coefficients'], [0, .25, .5, 1])
        self.assertEqual(protocol['scenario_stream'], gen.STREAM)

    def test_quarter_coefficient_matches_frozen_learning_rollout(self):
        for (name, _), bundle in self.bundles.items():
            new = gen.learned_negotiation(*self.scenario, bundle, .25)
            old = gen.lc.rollout(*self.scenario, gen.lc.SPECS[name], bundle=bundle)
            for column in gen.METRICS+['price_ratio']:
                np.testing.assert_allclose(new[column], old[column], rtol=0, atol=1e-12, equal_nan=True)

    def test_acceptance_endpoints_preserve_sequential_monotonicity(self):
        for bundle in self.bundles.values():
            low = gen.learned_negotiation(*self.scenario, bundle, 0.)
            high = gen.learned_negotiation(*self.scenario, bundle, 1.)
            self.assertTrue((high.deal >= low.deal).all())
            self.assertTrue((high['rounds'] <= low['rounds']).all())

    def test_no_overlap_infeasibility_keeps_failed_attempts(self):
        df, randoms, cfg = self.scenario
        impossible = df.copy()
        impossible['floor'] = .95
        impossible['value'] = .6
        impossible['opening'] = .3
        for bundle in self.bundles.values():
            base = gen.learned_negotiation(impossible, randoms, cfg, bundle, .25)
            self.assertEqual(len(base), len(df))
            self.assertEqual(int(base.deal.sum()), 0)
            self.assertTrue((base['rounds'] >= 1).all())
            self.assertTrue(base.price_ratio.isna().all())
            self.assertTrue((base.seller_surplus == 0).all())
            outcome = gen.pc.opportunity_outcomes(base, 'Forced negotiation', cost=.005)
            seller_net = outcome[:, gen.pc.METRICS.index('seller_net')]
            np.testing.assert_allclose(seller_net, -.005*base['rounds'], rtol=0, atol=1e-12)

    def test_clock_candidate_and_features_exclude_buyer_fields(self):
        state = np.asarray([[.5, 1, .7, .6, .85, .55, .05, .4, .6]], dtype=np.float32)
        altered = state.copy()
        altered[:, [3, 5, 6, 7, 8]] = .99
        np.testing.assert_array_equal(gen.lc.features(state, [0.], 'clock'), gen.lc.features(altered, [0.], 'clock'))
        np.testing.assert_array_equal(gen.lc.candidates(state, 'enriched'), gen.lc.candidates(altered, 'enriched'))

    def test_channel_zero_optin_equals_checkout(self):
        bundle = self.bundles[('mixed_enriched_128', gen.SEEDS[0])]
        base = gen.learned_negotiation(*self.scenario, bundle, .25)
        checkout = gen.pc.opportunity_outcomes(base, 'Checkout only')
        optional = gen.pc.opportunity_outcomes(base, 'Choice enabled', probability=0., fallback=1., cost=.02, selection='Exogenous')
        np.testing.assert_allclose(checkout, optional, rtol=0, atol=1e-12)

    def test_probability_accounting_scalar_oracle(self):
        bundle = self.bundles[('mixed_enriched_128', gen.SEEDS[0])]
        base = gen.learned_negotiation(*self.scenario, bundle, .25)
        actual = gen.pc.opportunity_outcomes(base, 'Choice enabled', probability=.5, fallback=.5, cost=.005, selection='Exogenous')
        for i, row in enumerate(base.itertuples()):
            s0 = (1-row.floor) if row.value >= 1 else 0.
            b0 = max(row.value-1, 0.)
            seller = .5*s0+.5*(row.seller_surplus+(1-row.deal)*.5*s0)-.005*.5*row.rounds
            buyer = .5*b0+.5*(row.buyer_surplus+(1-row.deal)*.5*b0)-.005*.5*row.rounds
            self.assertAlmostEqual(actual[i, gen.pc.METRICS.index('seller_net')], seller, places=12)
            self.assertAlmostEqual(actual[i, gen.pc.METRICS.index('buyer_net')], buyer, places=12)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationContracts)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    target = gen.ROOT/'results/test_verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    gen.write_json(target, dict(status='passed' if outcome.wasSuccessful() else 'failed', tests_run=outcome.testsRun,
                               failures=len(outcome.failures), errors=len(outcome.errors), skipped=len(outcome.skipped),
                               fixtures='Two evaluation products, 12 synthetic opportunities each, and explicit impossible-overlap unit fixtures; not an additional empirical sample.',
                               tests=unittest.defaultTestLoader.getTestCaseNames(IntegrationContracts)))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
