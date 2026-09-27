"""Transport identities and independent analytical limits."""
import json, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tracer_example import grid, symbol, evolve, tendency, receiver, setup, analytic_gain
from error_metrics import conservative_projection

class TracerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((ROOT/'data/tracer_spec.json').read_text())
        cls.results=json.loads((ROOT/'results/tracer_results.json').read_text())

    def test_constant_state(self):
        np.testing.assert_allclose(evolve(np.ones(64),[0.,86400.],1e5,.1,100.),1.,atol=1e-14)

    def test_mass_conservation_for_independent_state(self):
        x=grid(128,1e5);q=1.+.2*np.cos(2*np.pi*x/1e5)+.1*np.sin(6*np.pi*x/1e5)
        out=evolve(q,[0.,1234.,864000.],1e5,.1,100.)
        np.testing.assert_allclose(out.mean(axis=1),q.mean(),atol=1e-14)

    def test_flux_telescope(self):
        rng=np.random.default_rng(1401);q=rng.random(64)
        self.assertAlmostEqual(float(tendency(q,1e5,.1,100.).sum()),0.,places=14)

    def test_discrete_variance_identity(self):
        x=grid(128,1e5);q=np.sin(2*np.pi*x/1e5)+.4*np.cos(6*np.pi*x/1e5);dx=1e5/len(x)
        dot=tendency(q,1e5,.1,100.)
        lhs=dx*np.dot(q-q.mean(),dot);rhs=-100./dx*np.sum((np.roll(q,-1)-q)**2)
        self.assertAlmostEqual(lhs,rhs,places=13)

    def test_pure_advection_preserves_variance(self):
        x=grid(128,1e5);q=np.cos(2*np.pi*x/1e5)
        out=evolve(q,[1e4,1e5],1e5,.1,0.)
        np.testing.assert_allclose(np.mean(out**2,axis=1),np.mean(q*q),atol=1e-14)

    def test_projection_applied_to_actual_error(self):
        s=setup(self.spec,64);p=conservative_projection(np.eye(64)/64,np.ones((1,64))/64)
        np.testing.assert_allclose(p@s['before_projection'],s['after_projection'],atol=1e-15)

    def test_analytic_reversal_and_projection_ratios(self):
        e=self.results['analytic_output_errors']
        self.assertAlmostEqual(e['A']/e['B'],4.)
        self.assertAlmostEqual((e['after_projection']/e['before_projection'])**2,4.)

    def test_grid_convergence_is_second_order(self):
        for row in self.results['convergence'][1:]:
            self.assertAlmostEqual(row['order_from_previous'],2.,delta=.001)
        self.assertLess(self.results['convergence'][-1]['max_output_error'],1e-6)

    def test_dense_exponential_check(self):
        self.assertLess(self.results['dense_exponential_max_abs_error'],5e-13)

    def test_no_negative_concentrations(self):
        self.assertGreater(self.results['minimum_concentration'],0.)

    def test_budget_residuals(self):
        self.assertLess(self.results['maximum_relative_mass_drift'],1e-13)
        self.assertLess(self.results['maximum_relative_variance_budget_residual'],1e-12)

    def test_receiver_normalization(self):
        weights=receiver(grid(64,1e5),1e5,62500.,10.)
        self.assertAlmostEqual(weights.sum(),1.)
        self.assertTrue(np.all(weights>0))

    def test_invalid_transport_inputs(self):
        for size in (True,4,64.5):
            with self.subTest(size=size),self.assertRaises(ValueError):grid(size,1e5)
        with self.assertRaises(ValueError):symbol(64,1e5,.1,-1.)
        with self.assertRaises(ValueError):evolve(np.ones(64),[-1.],1e5,.1,100.)

if __name__=='__main__':unittest.main()
