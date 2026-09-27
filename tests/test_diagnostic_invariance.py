"""Checks under cell partitioning, component rotation and metric coordinates."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from physical_diagnostics import budget_residual, kinetic_increment
from quadratic_order import inspect_metrics, reversal_witness


class DiagnosticInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.velocity = np.array([[0.37, -0.16, 0.02], [-0.21, 0.43, -0.04],
                                  [0.08, 0.12, 0.07]])
        self.increment = np.array([[0.012, -0.007, 0.003], [-0.009, 0.004, 0.002],
                                   [0.006, -0.011, -0.001]])
        self.mass = np.array([1.7, 4.1, 0.9])

    def test_unequal_cell_masses_and_three_components(self):
        result = kinetic_increment(self.velocity, self.increment, self.mass, 7.5)
        before = 0.5 * np.sum(self.mass * np.sum(self.velocity ** 2, axis=1))
        after = 0.5 * np.sum(self.mass * np.sum((self.velocity + self.increment) ** 2, axis=1))
        self.assertAlmostEqual(result['total_energy_j'], after - before, places=14)

    def test_energy_increment_is_additive_over_cells(self):
        result = kinetic_increment(self.velocity, self.increment, self.mass, 7.5)
        parts = [kinetic_increment(self.velocity[i:i+1], self.increment[i:i+1],
                                   self.mass[i:i+1], 7.5) for i in range(len(self.mass))]
        for key in ('linear_energy_j', 'quadratic_energy_j', 'total_energy_j'):
            self.assertAlmostEqual(result[key], sum(p[key] for p in parts), places=14)

    def test_energy_is_invariant_under_component_rotation(self):
        rotation, _ = np.linalg.qr(np.array([[1., .3, .2], [.4, 1.2, -.1], [.2, .5, .9]]))
        original = kinetic_increment(self.velocity, self.increment, self.mass, 7.5)
        rotated = kinetic_increment(self.velocity @ rotation, self.increment @ rotation,
                                    self.mass, 7.5)
        for key in ('linear_energy_j', 'quadratic_energy_j', 'total_energy_j'):
            self.assertAlmostEqual(original[key], rotated[key], places=14)

    def test_variable_fluxes_on_irregular_intervals(self):
        times = np.array([0., 3., 11., 18.5])
        rates = np.array([1.2, -0.7, 2.3])
        increments = np.array([0.4, -0.9, 0.2])
        inventory = np.r_[47.5, 47.5 + np.cumsum(np.diff(times) * rates + increments)]
        result = budget_residual(times, inventory, rates, increments)
        np.testing.assert_allclose(result['corrected_residual'], 0., atol=1e-14)
        np.testing.assert_allclose(result['physical_only_residual'], increments, atol=1e-14)

    def test_dense_positive_proportional_weights(self):
        rng = np.random.default_rng(1931)
        transform = rng.normal(size=(7, 7))
        weight = transform.T @ transform + np.eye(7)
        result = inspect_metrics(2.7 * weight, weight)
        self.assertTrue(result['positive_proportional_metric'])
        np.testing.assert_allclose(result['generalized_eigenvalues'], 2.7, rtol=1e-12)

    def test_dense_metric_reversal_after_coordinate_change(self):
        rng = np.random.default_rng(1932)
        transform = rng.normal(size=(5, 5)) + 4 * np.eye(5)
        field = transform.T @ transform
        response = transform.T @ np.diag([.2, .8, 1.3, 2.1, 4.7]) @ transform
        witness = reversal_witness(response, field, amplitude=1e-3)
        a, b = np.asarray(witness['a']), np.asarray(witness['b'])
        self.assertLess(a @ field @ a, b @ field @ b)
        self.assertGreater(a @ response @ a, b @ response @ b)

    def test_global_metric_scaling_preserves_witness_order(self):
        response = np.array([[3., .4], [.4, 1.]])
        field = np.array([[1.2, -.1], [-.1, 2.]])
        for scale in (1e-8, 1., 1e8):
            with self.subTest(scale=scale):
                witness = reversal_witness(scale * response, scale * field, amplitude=.01)
                self.assertLess(witness['field_squared_a'], witness['field_squared_b'])
                self.assertGreater(witness['downstream_a'], witness['downstream_b'])


if __name__ == '__main__':
    unittest.main()
