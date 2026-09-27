"""Receiver Jacobian, spectral bounds and nonlinear witness checks."""
import csv,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from acoustic_example import array_mismatch
from acoustic_metrics import cosine_basis

class AcousticMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=json.loads((ROOT/'results/acoustic_metric_results.json').read_text())
        with np.load(ROOT/'results/acoustic_metric_matrices.npz') as f:
            cls.matrices={k:f[k] for k in f.files}
        with (ROOT/'results/acoustic_metric_witnesses.csv').open() as f:cls.rows=list(csv.DictReader(f))

    def test_depth_metric(self):
        np.testing.assert_allclose(self.matrices['W'],np.eye(8),atol=2e-14)

    def test_receiver_metric_is_positive_semidefinite(self):
        for q in self.matrices['Q']:
            np.testing.assert_allclose(q,q.T,atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(q)[0],0.)

    def test_derivative_refinement(self):
        for row in self.result['checks']:
            self.assertLess(row['relative_Q_step_difference'],1e-6)
            self.assertLess(row['relative_lambda_min_step_difference'],5e-5)

    def test_modal_basis_refinement(self):
        for row in self.result['checks']:
            self.assertLess(row['relative_Q_basis_difference'],1e-6)
            self.assertLess(row['relative_lambda_min_basis_difference'],5e-5)

    def test_all_specified_witnesses_are_evaluated(self):
        self.assertEqual(len(self.rows),24)
        self.assertTrue(all(r['full_reversal']=='True' for r in self.rows))

    def test_small_amplitude_quadratic_limit(self):
        self.assertLess(self.result['small_amplitude_max_quadratic_relative_error'],.02)

    def test_sharp_quadratic_threshold(self):
        with (ROOT/'results/acoustic_ranking_bound.csv').open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            factor=float(row['threshold_factor'])
            self.assertAlmostEqual(float(row['quadratic_ratio_a_over_b']),factor**2,places=7)
            self.assertEqual(row['certified_quadratic_order']=='True',factor<1.)

    def test_projected_metric_matches_jacobian(self):
        for i,q in enumerate(self.matrices['Q']):
            p=self.matrices['p0'][i];g=self.matrices['G'][i]
            proj=np.eye(len(p))-np.outer(p,p.conj())/np.vdot(p,p).real
            direct=(g.conj().T@proj@g).real/np.vdot(p,p).real
            np.testing.assert_allclose(q,direct,atol=3e-12,rtol=1e-9)

    def test_nearly_parallel_fields_do_not_lose_small_mismatch(self):
        p=np.array([[1.+0j,0.+0j]]);r=np.array([[1.+0j,1e-8+0j]])
        self.assertAlmostEqual(float(array_mismatch(p,r)[0])/1e-16,1.,places=13)

    def test_nested_error_spaces_change_threshold(self):
        for distance in self.result['spec']['ranges_m']:
            rows=[r for r in self.result['dimensions'] if r['range_m']==distance]
            ordered=sorted(rows,key=lambda r:r['dimension'])
            self.assertTrue(all(a['norm_ratio_threshold']>=b['norm_ratio_threshold']
                                for a,b in zip(ordered[:-1],ordered[1:])))

if __name__=='__main__':unittest.main()
