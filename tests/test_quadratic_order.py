"""Numerical checks of the quadratic ranking criterion and its hypotheses."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from quadratic_order import inspect_metrics, reversal_witness


class QuadraticOrderTests(unittest.TestCase):
    def test_positive_proportional_metrics(self):
        W = np.array([[2., .2], [.2, 1.]])
        report = inspect_metrics(3 * W, W)
        self.assertTrue(report['positive_proportional_metric'])
        np.testing.assert_allclose(report['generalized_eigenvalues'], [3, 3], rtol=1e-13)

    def test_anisotropic_witness_has_strict_opposite_order(self):
        witness = reversal_witness(np.diag([1., 4.]), np.eye(2))
        self.assertLess(witness['field_squared_a'], witness['field_squared_b'])
        self.assertGreater(witness['downstream_a'], witness['downstream_b'])
        self.assertAlmostEqual(witness['field_squared_a'], .625)
        self.assertAlmostEqual(witness['downstream_a'], 2.5)

    def test_arbitrarily_small_quadratic_scaling(self):
        for amplitude in (1., .01, 1e-4, 1e-8):
            with self.subTest(amplitude=amplitude):
                w = reversal_witness(np.diag([1., 4.]), np.eye(2), amplitude=amplitude)
                self.assertLess(w['field_squared_a'], w['field_squared_b'])
                self.assertGreater(w['downstream_a'], w['downstream_b'])
                np.testing.assert_allclose(w['downstream_a'], 2.5 * amplitude ** 2, rtol=1e-13)

    def test_null_direction_allows_reversal(self):
        witness = reversal_witness(np.diag([0., 4.]), np.eye(2))
        self.assertEqual(witness['downstream_b'], 0)
        self.assertGreater(witness['downstream_a'], 0)

    def test_zero_response_does_not_preserve_strict_order(self):
        report = inspect_metrics(np.zeros((2, 2)), np.eye(2))
        self.assertTrue(report['proportional_within_tolerance'])
        self.assertFalse(report['positive_proportional_metric'])
        with self.assertRaises(ValueError):
            reversal_witness(np.zeros((2, 2)), np.eye(2))

    def test_positive_proportional_metric_has_no_reversal(self):
        with self.assertRaises(ValueError):
            reversal_witness(2 * np.eye(3), np.eye(3))

    def test_nonsingular_coordinate_change(self):
        S = np.array([[1., .3], [.2, 1.7]])
        Q, W = S.T @ np.diag([1., 4.]) @ S, S.T @ S
        report = inspect_metrics(Q, W)
        np.testing.assert_allclose(report['generalized_eigenvalues'], [1., 4.], rtol=1e-13)
        witness = reversal_witness(Q, W)
        a, b = np.array(witness['a']), np.array(witness['b'])
        self.assertLess(a @ W @ a, b @ W @ b)
        self.assertGreater(a @ Q @ a, b @ Q @ b)

    def test_weight_scaling_retains_metric_units(self):
        report = inspect_metrics(np.diag([1., 4.]), 10 * np.eye(2))
        np.testing.assert_allclose(report['generalized_eigenvalues'], [.1, .4])

    def test_one_dimensional_positive_metric(self):
        self.assertTrue(inspect_metrics([[4]], [[2]])['positive_proportional_metric'])

    def test_nonpositive_field_weights_are_rejected(self):
        for W in (np.zeros((2, 2)), np.diag([1., -1.])):
            with self.subTest(W=W), self.assertRaises(ValueError):
                inspect_metrics(np.eye(2), W)

    def test_negative_downstream_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            inspect_metrics(np.diag([-1., 2.]), np.eye(2))

    def test_asymmetric_nonfinite_or_incompatible_inputs(self):
        invalid = [([[1, 1], [0, 1]], np.eye(2)), ([[np.nan]], [[1]]),
                   (np.eye(2), np.eye(3)), ([], []), ([1, 2], [1, 2]),
                   (np.eye(2).astype(complex), np.eye(2))]
        for Q, W in invalid:
            with self.subTest(Q=Q), self.assertRaises(ValueError):
                inspect_metrics(Q, W)

    def test_invalid_tolerances_and_amplitudes(self):
        for value in (0, -1, np.nan, np.inf, True, 'small'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reversal_witness(np.diag([1., 4.]), np.eye(2), amplitude=value)
        for value in (0, -1, 1, np.nan, True, 'small'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect_metrics(np.eye(2), np.eye(2), rtol=value)

    def test_released_manufactured_checks_recompute(self):
        result = json.loads((ROOT / 'results/quadratic_order_checks.json').read_text())
        self.assertEqual(result['scope'], 'manufactured_quadratic_metrics')
        self.assertEqual(len(result['records']), 5)
        for record in result['records']:
            recomputed = inspect_metrics(record['Q'], record['W'])
            self.assertEqual(recomputed['resolved_anisotropy'], record['resolved_anisotropy'])
            for witness in record.get('witnesses', []):
                check = reversal_witness(record['Q'], record['W'], amplitude=witness['amplitude'])
                for key in ('field_squared_a', 'field_squared_b', 'downstream_a', 'downstream_b'):
                    np.testing.assert_allclose(check[key], witness[key], rtol=1e-13)


if __name__ == '__main__':
    unittest.main()
