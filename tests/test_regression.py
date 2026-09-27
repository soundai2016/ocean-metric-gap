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
class RangeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'results/acoustic_results.json').read_text())
        cls.rows = {row['range_step_m']: row for row in cls.result['range_convergence']}

    def test_endpoint_inclusive_range_grid(self):
        r = range_grid(500., 30000., 1.)
        self.assertEqual(len(r), 29501)
        self.assertEqual((r[0], r[-1]), (500., 30000.))
        np.testing.assert_array_equal(np.diff(r), 1.)

    def test_range_grid_rejects_invalid_bounds(self):
        for args in [(0, 10, 1), (10, 5, 1), (5, 5, 1), (1, 5, 0),
                     (1, 5, -1), (np.nan, 5, 1), (1, np.inf, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                range_grid(*args)

    def test_range_grid_rejects_truncated_endpoint(self):
        with self.assertRaises(ValueError):
            range_grid(1., 10., 2.)

    def test_chunking_preserves_field_calculation(self):
        z = np.linspace(0, 200, 401)
        c = reference_speed(z)
        a, b = solve_modes(z, c, 100.), solve_modes(z, c + .5, 100.)
        r, zr = np.linspace(500, 2000, 11), np.array([30., 50., 100., 150.])
        expected = array_mismatch(pressure(a, r, 50., zr), pressure(b, r, 50., zr))
        for chunk in (1, 4, 2048):
            with self.subTest(chunk=chunk):
                np.testing.assert_allclose(mismatch_curve(a, b, r, 50., zr, chunk),
                                           expected, atol=2e-14, rtol=2e-14)

    def test_chunking_rejects_bad_size_before_calculation(self):
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                mismatch_curve(None, None, np.array([1.]), 50., np.array([20.]), size)

    def test_chunking_rejects_bad_ranges(self):
        for r in ([], [0.], [np.nan], [[1.]], [-1.]):
            with self.subTest(r=r), self.assertRaises(ValueError):
                mismatch_curve(None, None, np.asarray(r), 50., np.array([20.]))

    def test_sampling_diagnostic_exact_linear_curve(self):
        r = np.arange(1., 6.)
        rows = sampling_diagnostics(r, {'linear': 2 * r + 1}, [2., 1.])
        for row in rows:
            self.assertAlmostEqual(row['linear_mean'], 7.)
            self.assertEqual(row['linear_interpolation_max_abs_error'], 0.)

    def test_sampling_diagnostic_detects_missed_peak(self):
        r = np.array([1., 2., 3.])
        row = sampling_diagnostics(r, {'peak': np.array([0., 1., 0.])}, [2.])[0]
        self.assertEqual(row['peak_max'], 0.)
        self.assertEqual(row['peak_interpolation_max_abs_error'], 1.)
        self.assertEqual(row['peak_max_abs_error'], 1.)

    def test_sampling_rejects_nonuniform_or_misaligned_inputs(self):
        for r, curves, steps in [([1, 2, 4], {'x': [0, 1, 0]}, [1]),
                                 ([1, 2, 3], {'x': [0, 1]}, [1]),
                                 ([1, 2, 3], {'x': [0, np.nan, 0]}, [1]),
                                 ([1, 2, 3], {'x': [0, 1, 0]}, [.5]),
                                 ([1, 2, 3], {}, [1]),
                                 ([1, 2, 3], {'x': [0, 1, 0]}, [])]:
            with self.subTest(r=r, steps=steps), self.assertRaises(ValueError):
                sampling_diagnostics(np.asarray(r), {k: np.asarray(v) for k,v in curves.items()}, steps)

    def test_main_range_step_resolves_reference_to_stated_tolerance(self):
        row = self.rows[self.result['spec']['report_range_step_m']]
        for name in ('uniform', 'localized'):
            self.assertLess(row[name + '_interpolation_max_abs_error'], 1e-3)
            self.assertLess(row[name + '_mean_abs_error'], 1e-7)

    def test_original_fifty_metre_sampling_is_distinguishable(self):
        for name in ('uniform', 'localized'):
            self.assertGreater(self.rows[50.][name + '_interpolation_max_abs_error'], .1)
            self.assertGreater(self.rows[50.][name + '_max_abs_error'], .01)

    def test_finest_reference_is_identical_to_itself(self):
        row = self.rows[self.result['spec']['range_reference_step_m']]
        for name in ('uniform', 'localized'):
            for suffix in ('_mean_abs_error', '_max_abs_error', '_interpolation_max_abs_error'):
                self.assertEqual(row[name + suffix], 0.)

    def test_twenty_km_reference_matches_reported_values(self):
        rows = {r['perturbation']: r for r in self.result['summaries']}
        self.assertAlmostEqual(rows['uniform']['range_20km_mismatch'], .8620389421, places=8)
        self.assertAlmostEqual(rows['localized']['range_20km_mismatch'], .7437081296, places=8)

    def test_basis_and_range_errors_are_separate_records(self):
        self.assertIn('max_array_mismatch_error_vs_finest', self.result['convergence'][0])
        self.assertIn('uniform_interpolation_max_abs_error', self.result['range_convergence'][0])
        self.assertFalse(self.result['field_data'])
        self.assertFalse(self.result['trained_model'])
if __name__ == '__main__':
    unittest.main()
