"""Check the numerical package's local output and input contracts."""
from pathlib import Path
import ast
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ExecutionContractTests(unittest.TestCase):
    def test_entry_point_is_available(self):
        ast.parse((ROOT/'run.py').read_text())
        self.assertTrue((ROOT/'VERSION').read_text().strip())

    def test_no_document_builder_modules(self):
        for name in ('build.py','build_configurations.py','document_blocks.py'):
            self.assertFalse((ROOT/'scripts'/name).exists())

    def test_no_document_sources_in_package(self):
        self.assertEqual(list(ROOT.rglob('*.tex')), [])
        self.assertEqual(list(ROOT.rglob('*.bib')), [])

    def test_figure_modules_use_local_output(self):
        import generate_figures
        import science_figures
        self.assertEqual(generate_figures.FIG, ROOT/'figures')
        self.assertEqual(science_figures.FIGURES, ROOT/'figures')

    def test_input_and_output_are_finite_json(self):
        def reject(token):
            raise ValueError(f'Nonfinite JSON value: {token}')
        for name in ('tracer_results.json','acoustic_metric_results.json',
                     'physical_diagnostics.json','quadratic_order_checks.json'):
            json.loads((ROOT/'results'/name).read_text(),parse_constant=reject)

    def test_all_numerical_source_files_parse(self):
        for path in (ROOT/'scripts').glob('*.py'):
            with self.subTest(path=path.name): ast.parse(path.read_text())


if __name__=='__main__':
    unittest.main()
