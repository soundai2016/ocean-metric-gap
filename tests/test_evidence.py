"""Validation of case metadata, evidence coding and literature transcriptions."""
from pathlib import Path
import csv
import copy
import json
import re
import sys
import unittest
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence_records import load_cases, validate_case, validate_source, flatten, source_index, ROOT

class EvidenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cases, cls.sources = load_cases()

    def test_all_released_cases_validate(self):
        for c in self.cases:
            validate_case(c, self.sources)

    def test_stable_ids_are_unique(self):
        self.assertEqual(len(self.cases), len({c['configuration_id'] for c in self.cases}))

    def test_multiple_configurations_per_source(self):
        counts = {s: sum((c['source_id'] == s for c in self.cases)) for s in self.sources}
        self.assertGreater(max(counts.values()), 2)

    def test_setting_does_not_assign_independence_or_next_test(self):
        c = copy.deepcopy(self.cases[0])
        old = flatten(c)
        for setting in 'DSHF':
            c['evaluation_setting'] = setting
            new = flatten(c)
            self.assertEqual(new['evidence_independence'], old['evidence_independence'])
            self.assertEqual(new['next_test'], old['next_test'])

    def test_missing_independence_rejected(self):
        c = copy.deepcopy(self.cases[0])
        del c['evidence_independence']
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_unknown_is_not_dev(self):
        self.assertTrue(all((c['evidence_independence'] == 'UNKNOWN' for c in self.cases)))
        self.assertTrue(all((c['independence_basis'] for c in self.cases)))

    def test_missing_locator_rejected(self):
        c = copy.deepcopy(self.cases[0])
        c['source_locator'] = ' '
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_source_version_mismatch_rejected(self):
        c = copy.deepcopy(self.cases[0])
        c['source_version'] = 'unspecified latest'
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_article_number_not_page(self):
        sources = copy.deepcopy(self.sources)
        c = self.cases[0]
        sources[c['source_id']]['article_number_is_page'] = True
        with self.assertRaises(ValueError):
            validate_case(c, sources)

    def test_baseline_exclusivity(self):
        for axis in 'CN':
            c = copy.deepcopy(self.cases[0])
            c['axes'][axis] = [axis + '0', axis + '1']
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    validate_case(c, self.sources)

    def test_unassessed_cannot_support_assignment(self):
        c = copy.deepcopy(self.cases[0])
        c['axis_evidence']['O']['status'] = 'unassessed'
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_n1_requires_actual_probability_assessment(self):
        c = copy.deepcopy(next((c for c in self.cases if 'N1' in c['axes']['N'])))
        c['probability_assessment'] = ''
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_proper_score_does_not_prove_calibration(self):
        c = copy.deepcopy(self.cases[0])
        c['calibration_quality'] = 'calibrated'
        c.pop('calibration_quality_basis', None)
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_n2_requires_reference_and_contrast(self):
        c = copy.deepcopy(next((c for c in self.cases if 'N2' in c['axes']['N'])))
        c['shift_contract'] = ''
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_unknown_n_not_replaced_by_n0(self):
        c = next((c for c in self.cases if not c['axes']['N']))
        self.assertEqual(flatten(c)['N_levels'], '')

    def test_domain_can_be_cross_classified(self):
        c = copy.deepcopy(self.cases[0])
        c['domain_tags'] = ['ocean_dynamics', 'underwater_acoustics']
        validate_case(c, self.sources)
        self.assertEqual(flatten(c)['domain_tags'], 'ocean_dynamics;underwater_acoustics')

    def test_undocumented_overclaim_rejected(self):
        c = copy.deepcopy(self.cases[0])
        c['author_overclaim'] = 'yes_because_no_field_test'
        with self.assertRaises(ValueError):
            validate_case(c, self.sources)

    def test_source_validation_scope_is_explicit(self):
        s = copy.deepcopy(next(iter(self.sources.values())))
        validate_source(s)
        self.assertEqual(s['validation_scope'], 'metadata_consistency_only')
        s['validation_scope'] = 'independent_source_verification'
        with self.assertRaises(ValueError):
            validate_source(s)

    def test_source_identity_fields_are_required(self):
        for field in ('source_id', 'citation_key', 'url', 'version', 'locator', 'validation_scope'):
            with self.subTest(field=field):
                s = copy.deepcopy(next(iter(self.sources.values())))
                del s[field]
                with self.assertRaises(ValueError):
                    validate_source(s)

    def test_source_identity_fields_are_nonempty_text(self):
        for field in ('source_id', 'citation_key', 'url', 'version', 'locator', 'validation_scope'):
            for value in (' ', None, 1):
                with self.subTest(field=field, value=value):
                    s = copy.deepcopy(next(iter(self.sources.values())))
                    s[field] = value
                    with self.assertRaises(ValueError):
                        validate_source(s)

    def test_published_transcription_is_complete(self):
        with (ROOT / 'data/closure_benchmark.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([int(r['grid_denominator']) for r in rows], [4, 5, 6, 7, 8])
        self.assertAlmostEqual(float(rows[0]['no_parameterization_rmse_m']), 0.278)
        self.assertAlmostEqual(float(rows[-1]['ke_tuned_rmse_m']), 0.0891)

    def test_transcription_preserves_source_metadata(self):
        with (ROOT / 'data' / 'closure_benchmark.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            with self.subTest(grid=row['grid_denominator']):
                source = self.sources[row['source_id']]
                self.assertIn(row['source_version'], source['version'])
                self.assertTrue(row['source_table'].strip())
                self.assertTrue(row['uncertainty'].strip())
                self.assertTrue(row['interpretation'].strip())

    def test_transcription_grid_spacing_matches_denominator(self):
        with (ROOT / 'data' / 'closure_benchmark.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            self.assertAlmostEqual(float(row['grid_degree']), 1.0 / int(row['grid_denominator']))

    def test_transcription_errors_are_finite_and_nonnegative(self):
        with (ROOT / 'data' / 'closure_benchmark.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ('no_parameterization_rmse_m', 'unscaled_rmse_m', 'ke_tuned_rmse_m'):
                with self.subTest(grid=row['grid_denominator'], column=key):
                    value = float(row[key])
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0.0)


if __name__ == '__main__':
    unittest.main()
