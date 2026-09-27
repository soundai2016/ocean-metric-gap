"""Numerical regression and record-schema tests."""
import copy, csv, json, re, sys, unittest, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from acoustic_example import (range_grid, mismatch_curve, sampling_diagnostics,
    solve_modes, reference_speed, pressure, array_mismatch)
from evidence_records import load_cases, validate_case
from reporting_protocol import validate_record
class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions=json.loads((ROOT/'data/minimum_reporting_set.json').read_text())
        cls.record=json.loads((ROOT/'data/acoustic_reporting_record.json').read_text())
    def test_complete_example(self):
        report=validate_record(self.record,self.definitions,ROOT)
        self.assertTrue(report['valid']); self.assertIsNone(report['scientific_quality_score'])
        self.assertFalse(report['independent_source_verification'])
    def test_missing_or_duplicate_item(self):
        for items in (self.record['items'][:-1],self.record['items']+[self.record['items'][0]]):
            r=copy.deepcopy(self.record); r['items']=items
            with self.assertRaises(ValueError): validate_record(r,self.definitions,ROOT)
    def test_unknown_status(self):
        r=copy.deepcopy(self.record); r['items'][0]['status']='passed'
        with self.assertRaises(ValueError): validate_record(r,self.definitions,ROOT)
    def test_nonapplicable_needs_reason(self):
        r=copy.deepcopy(self.record); r['items'][2]['detail']=' '
        with self.assertRaises(ValueError): validate_record(r,self.definitions,ROOT)
    def test_invalid_tier(self):
        for tiers in ([],[4,4],[True],[0],['4'],[[4]],[{'tier':4}]):
            r=copy.deepcopy(self.record); r['tested_tiers']=tiers
            with self.subTest(tiers=tiers),self.assertRaises(ValueError):
                validate_record(r,self.definitions,ROOT)
    def test_no_external_or_missing_artifacts(self):
        for path in ('../outside.txt','/etc/passwd','not_a_file.txt'):
            r=copy.deepcopy(self.record); r['items'][0]['artifacts']=[path]
            with self.subTest(path=path),self.assertRaises(ValueError):
                validate_record(r,self.definitions,ROOT)

class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=json.loads((ROOT/'results/acoustic_sensitivity.json').read_text())
        cls.rows=cls.result['contrast']
    def test_selected_pair_is_declared_exploratory(self):
        self.assertIn('not preregistered',self.result['spec']['selection'])
        self.assertFalse(self.result['field_data']);self.assertFalse(self.result['trained_model'])
    def test_actual_ranking_reversal_and_change_of_order(self):
        u,l=self.rows
        self.assertLess(u['rms_mps'],l['rms_mps'])
        for key in ('range_10km_mismatch','range_30km_mismatch','range_mean_mismatch'):
            self.assertGreater(u[key],l[key])
        self.assertLess(u['range_20km_mismatch'],l['range_20km_mismatch'])
    def test_contrast_expected_values(self):
        np.testing.assert_allclose([r['range_10km_mismatch'] for r in self.rows],
                                   [.747821022145,.516058523730],atol=1e-8,rtol=0)
    def test_contrast_independent_convergence_limits(self):
        for row in self.rows:
            self.assertLess(row['basis_max_abs_error'],2e-8)
            self.assertLess(row['range_interpolation_max_abs_error'],5e-4)
            self.assertLess(abs(row['range_mean_mismatch']-row['fine_range_mean_mismatch']),3e-8)
    def test_sensitivity_all_configurations_saved(self):
        with (ROOT/'results/acoustic_operator_sensitivity.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),54)
        keys={(r['frequency_hz'],r['source_depth_m'],r['range_m'],r['perturbation']) for r in rows}
        self.assertEqual(len(keys),54)
        for row in rows:
            self.assertGreaterEqual(float(row['mismatch']),0.)
            self.assertLessEqual(float(row['mismatch']),1.)
            self.assertLess(float(row['basis_abs_error']),4e-7)
            self.assertEqual(row['reference_modes'],row['perturbed_modes'])
        self.assertEqual(self.result['operator_comparisons'],27)
        self.assertGreater(self.result['illustrative_field_task_reversals'],0)
        self.assertLess(self.result['illustrative_field_task_reversals'],27)
    def test_local_quadratic_operator_check(self):
        with (ROOT/'results/acoustic_local_metric.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),24)
        self.assertLess(max(float(row['relative_error']) for row in rows),.005)
        self.assertTrue(all(float(row['quadratic_mismatch']) >= 0 for row in rows))
    def test_full_amplitude_sweep_and_zero_error(self):
        with (ROOT/'results/acoustic_amplitude_sweep.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),150)
        for row in rows:
            if float(row['rms_mps'])==0.: self.assertLess(abs(float(row['mismatch'])),1e-12)
if __name__ == '__main__':
    unittest.main()
