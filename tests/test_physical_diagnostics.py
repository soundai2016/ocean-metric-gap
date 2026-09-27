"""Analytic checks of discrete energy and inventory accounting."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from physical_diagnostics import kinetic_increment, budget_residual

class EnergyTests(unittest.TestCase):
    def test_direct_energy_identity(self):
        rng = np.random.default_rng(702)
        u, du, m = rng.normal(size=(13, 3)), rng.normal(size=(13, 3)), rng.uniform(.5, 4, size=13)
        r = kinetic_increment(u, du, m, 2.)
        self.assertAlmostEqual(r['total_energy_j'], r['direct_energy_j'], places=12)
        self.assertAlmostEqual(r['effective_power_w'], r['total_energy_j']/2.)
    def test_zero_linear_work_can_inject_energy(self):
        r = kinetic_increment([[1.], [1.]], [[.1], [-.1]], [2., 2.], 10.)
        self.assertEqual(r['linear_energy_j'], 0.)
        self.assertAlmostEqual(r['quadratic_energy_j'], .02)
        self.assertAlmostEqual(r['effective_power_w'], .002)
    def test_zero_increment(self):
        self.assertEqual(kinetic_increment([[1.,2.]], [[0.,0.]], [3.], 1.)['total_energy_j'], 0.)
    def test_damping_has_negative_total_work(self):
        self.assertLess(kinetic_increment([[1.]], [[-.5]], [1.], 1.)['total_energy_j'], 0.)
    def test_invalid_interval(self):
        for v in (0.,-1.,np.nan,np.inf):
            with self.subTest(v=v), self.assertRaises(ValueError):
                kinetic_increment([[1.]], [[1.]], [1.], v)
    def test_invalid_masses(self):
        for mass in ([-1.],[0.],[np.nan],[1.,2.]):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                kinetic_increment([[1.]], [[1.]], mass, 1.)
    def test_invalid_velocity_shape(self):
        for u,du in (([1.],[1.]),([],[]),([[1.,2.]],[[1.]]),([[np.nan]],[[1.]])):
            with self.subTest(u=u), self.assertRaises(ValueError):
                kinetic_increment(u,du,[1.],1.)

class BudgetTests(unittest.TestCase):
    def test_physical_only_residual_not_hidden(self):
        r=budget_residual([0.,10.,20.],[100.,123.,142.],[2.,2.],[3.,-1.])
        np.testing.assert_allclose(r['corrected_residual'], [0.,0.])
        np.testing.assert_allclose(r['physical_only_residual'], [3.,-1.])
        self.assertEqual(r['cumulative_physical_only_residual'][-1],2.)
    def test_irregular_intervals(self):
        r=budget_residual([0.,2.,7.],[1.,7.,22.],[3.,3.])
        self.assertEqual(r['max_abs_corrected_residual'],0.)
    def test_sign_error_detected(self):
        r=budget_residual([0.,10.],[0.,20.],[-2.])
        self.assertEqual(r['corrected_residual'],[40.])
    def test_non_increasing_time(self):
        for t in ([0.,0.],[1.,0.],[0.,np.inf]):
            with self.subTest(t=t), self.assertRaises(ValueError):
                budget_residual(t,[0.,1.],[1.])
    def test_bad_shapes_and_nonfinite(self):
        cases=[([0.,1.],[0.],[1.],None),([0.,1.],[0.,1.],[1.,1.],None),
               ([0.,1.],[0.,1.],[1.],[1.,1.]),([0.,1.],[0.,np.nan],[1.],None)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                budget_residual(*args)
    def test_saved_examples_are_manufactured(self):
        r=json.loads((ROOT/'results/physical_diagnostics.json').read_text())
        self.assertFalse(r['new_ocean_simulations'])
        self.assertEqual(r['kind'],'manufactured_diagnostic_checks')
