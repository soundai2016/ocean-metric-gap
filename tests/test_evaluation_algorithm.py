"""Evaluation-plan selection and validation."""
import json
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluation_plan import make_plan


def request():
    return {'record_id': 'test', 'claimed_tests': [4], 'comparison_target': 'matched_information',
            'prognostic_correction': False, 'probabilistic_claim': False, 'transfer_condition': None}


class EvaluationPlanTests(unittest.TestCase):
    def test_downstream_only_does_not_imply_coupled_integration(self):
        plan = make_plan(request())
        ids = [x['id'] for x in plan['checks']]
        self.assertIn('downstream', ids)
        for name in ('offline', 'initialized', 'coupled', 'correction_budget'):
            self.assertNotIn(name, ids)

    def test_each_tier_selects_its_own_measurements(self):
        for tier, expected in enumerate(('offline', 'initialized', 'coupled', 'downstream'), 1):
            q = request(); q['claimed_tests'] = [tier]
            plan = make_plan(q)
            ids = {x['id'] for x in plan['checks']}
            self.assertIn(expected, ids)
            self.assertEqual(len(ids & {'offline', 'initialized', 'coupled', 'downstream'}), 1)

    def test_multiple_tests_are_sorted_without_mutating_input(self):
        q = request(); q['claimed_tests'] = [4, 1, 3, 2]; original = deepcopy(q)
        plan = make_plan(q)
        self.assertEqual(plan['claimed_tests'], [1, 2, 3, 4])
        self.assertEqual(q, original)

    def test_budget_check_applies_to_initialized_corrections(self):
        q = request(); q['claimed_tests'] = [2]; q['prognostic_correction'] = True
        self.assertIn('correction_budget', {x['id'] for x in make_plan(q)['checks']})

    def test_probabilistic_claim_adds_conditional_reliability(self):
        q = request(); q['probabilistic_claim'] = True
        check = next(x for x in make_plan(q)['checks'] if x['id'] == 'probabilistic_verification')
        self.assertIn('conditional reliability', check['action'])
        self.assertIn('M08', check['reporting_items'])

    def test_transfer_keeps_frozen_and_retuned_separate(self):
        q = request(); q['transfer_condition'] = 'grid spacing'
        ids = [x['id'] for x in make_plan(q)['checks']]
        self.assertLess(ids.index('transfer_frozen'), ids.index('transfer_retuned'))

    def test_system_comparison_does_not_assert_matched_inputs(self):
        q = request(); q['comparison_target'] = 'complete_system'
        comparator = next(x for x in make_plan(q)['checks'] if x['id'] == 'comparator')
        self.assertIn('each complete system', comparator['action'])
        self.assertNotIn('Hold admissible inputs fixed', comparator['action'])

    def test_plans_do_not_fabricate_thresholds_or_results(self):
        plan = make_plan(request())
        self.assertFalse(plan['experiment_results_evaluated'])
        for check in plan['checks']:
            self.assertEqual(check['status'], 'pending')
            self.assertIsNone(check['criterion']); self.assertIsNone(check['result'])
            self.assertIsNone(check['source_locator'])
        self.assertEqual(plan['reporting_items'], [f'M{i:02}' for i in range(1, 11)])

    def test_invalid_tier_inputs(self):
        for tests in ([], [1, 1], [True], [0], [5], [1.], '1', [[1]]):
            q = request(); q['claimed_tests'] = tests
            with self.subTest(tests=tests), self.assertRaises(ValueError):
                make_plan(q)

    def test_invalid_boolean_comparison_and_transfer_inputs(self):
        for key, value in [('prognostic_correction', 1), ('probabilistic_claim', 'false'),
                           ('comparison_target', []), ('comparison_target', 'unspecified'),
                           ('transfer_condition', ''), ('transfer_condition', False),
                           ('record_id', ' ')]:
            q = request(); q[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                make_plan(q)

    def test_missing_and_unknown_fields_are_rejected(self):
        for q in (None, {}, {**request(), 'unexpected': 1}):
            with self.subTest(request=q), self.assertRaises(ValueError):
                make_plan(q)

    def test_released_plans_match_the_request_files(self):
        requests = json.loads((ROOT / 'data/evaluation_requests.json').read_text())
        released = json.loads((ROOT / 'results/evaluation_plans.json').read_text())
        self.assertEqual(released['plans'], [make_plan(q) for q in requests])


if __name__ == '__main__':
    unittest.main()
