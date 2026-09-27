"""Analytic tests of restricted metrics, ranking bounds and projections."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from error_metrics import (restricted_metrics, conservative_projection,
                           projection_diagnostics, ranking_bound)
from quadratic_order import inspect_metrics, reversal_witness

class MetricTests(unittest.TestCase):
    def test_positive_proportional_order(self):
        r=ranking_bound(3*np.eye(2),np.eye(2),.8,1.)
        self.assertTrue(r['strictly_ordered_for_all_directions'])
        self.assertAlmostEqual(r['norm_ratio_threshold'],1.)

    def test_sharp_bound(self):
        q=np.diag([1.,4.]);w=np.eye(2)
        self.assertTrue(ranking_bound(q,w,.49,1.)['strictly_ordered_for_all_directions'])
        self.assertFalse(ranking_bound(q,w,.5,1.)['strictly_ordered_for_all_directions'])
        self.assertFalse(ranking_bound(q,w,.51,1.)['strictly_ordered_for_all_directions'])

    def test_null_direction_has_no_positive_threshold(self):
        r=ranking_bound(np.diag([0.,4.]),np.eye(2),.001,1.)
        self.assertEqual(r['norm_ratio_threshold'],0.)
        self.assertFalse(r['strictly_ordered_for_all_directions'])

    def test_zero_response_is_a_tie(self):
        r=ranking_bound(np.zeros((2,2)),np.eye(2),0.,1.)
        self.assertIsNone(r['norm_ratio_threshold'])
        self.assertFalse(r['strictly_ordered_for_all_directions'])

    def test_constraint_removes_anisotropy_in_a_special_case(self):
        n,q,w=restricted_metrics(np.diag([9.,2.,2.]),np.eye(3),[[1.,0.,0.]])
        self.assertTrue(inspect_metrics(q,w)['positive_proportional_metric'])
        np.testing.assert_allclose(np.array([[1.,0.,0.]])@n,0.,atol=1e-15)

    def test_constraint_does_not_generally_remove_anisotropy(self):
        n,q,w=restricted_metrics(np.diag([9.,2.,3.]),np.eye(3),[[1.,0.,0.]])
        self.assertTrue(inspect_metrics(q,w)['resolved_anisotropy'])
        pair=reversal_witness(q,w)
        self.assertGreater(pair['downstream_a'],pair['downstream_b'])

    def test_empty_constraints(self):
        n,q,w=restricted_metrics(np.eye(2),np.eye(2),np.empty((0,2)))
        np.testing.assert_allclose(n,np.eye(2))
        np.testing.assert_allclose(conservative_projection(w,np.empty((0,2))),np.eye(2))

    def test_dense_weighted_projection(self):
        w=np.array([[3.,.3,.1],[.3,2.,.2],[.1,.2,1.]])
        c=np.array([[1.,2.,1.]])
        p=conservative_projection(w,c)
        np.testing.assert_allclose(p@p,p,atol=1e-14)
        np.testing.assert_allclose(c@p,0.,atol=1e-14)
        np.testing.assert_allclose(p.T@w,w@p,atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(w-p.T@w@p)[0],-1e-12)

    def test_commuting_metric_is_safe(self):
        d=projection_diagnostics(np.diag([1.,4.,2.]),np.eye(3),[[1.,0.,0.]])
        self.assertTrue(d['nonincreasing_within_tolerance'])
        self.assertLess(d['commutator_relative_norm'],1e-14)

    def test_cross_block_coupling_produces_witness(self):
        q=np.array([[1.,1.],[1.,1.]])
        d=projection_diagnostics(q,np.eye(2),[[1.,0.]])
        self.assertFalse(d['nonincreasing_within_tolerance'])
        self.assertGreater(d['witness_downstream_after'],d['witness_downstream_before'])
        self.assertLess(d['witness_field_squared_after'],d['witness_field_squared_before'])

    def test_projection_coordinate_invariance(self):
        t=np.array([[1.,.3],[.1,2.]])
        q=np.array([[2.,1.],[1.,1.]])
        c=np.array([[1.,0.]])
        a=projection_diagnostics(q,np.eye(2),c)
        b=projection_diagnostics(t.T@q@t,t.T@t,c@t)
        self.assertAlmostEqual(a['maximum_loss_increase_per_unit_field_squared'],
                               b['maximum_loss_increase_per_unit_field_squared'],places=12)

    def test_invalid_constraints(self):
        for c in ([[1.,0.,0.]],[[1.,0.],[2.,0.]],[[np.nan,0.]],[[1+1j,0.]],[1.,0.]):
            with self.subTest(c=c), self.assertRaises(ValueError):
                restricted_metrics(np.eye(2),np.eye(2),c)

    def test_invalid_norms(self):
        for a,b in ((-1.,1.),(1.,0.),(np.nan,1.),(1.,np.inf)):
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):
                ranking_bound(np.eye(2),np.eye(2),a,b)

if __name__=='__main__':unittest.main()
