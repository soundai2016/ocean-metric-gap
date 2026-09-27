"""Reproduce the tracer and acoustic numerical experiments."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import unittest
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))

CORE_MODULES = (
    'tracer_example',
    'acoustic_example',
    'acoustic_sensitivity',
    'acoustic_metrics',
    'physical_diagnostics',
    'quadratic_order',
)
AUXILIARY_MODULES = ('evidence_records', 'evaluation_plan', 'reporting_protocol')


def run_module(name: str) -> dict:
    tick = time.perf_counter()
    result = importlib.import_module(name).run()
    print(f'{name:22s} {time.perf_counter() - tick:8.3f} s')
    return result


def print_numerical_summary(results: dict[str, dict]) -> None:
    tracer = results.get('tracer_example')
    if tracer:
        orders = [row['order_from_previous'] for row in tracer['convergence']
                  if row['order_from_previous'] is not None]
        print('\nTracer diagnostics')
        print(f"  Mass drift       : {tracer['maximum_relative_mass_drift']:.3e}")
        print(f"  Variance residual: {tracer['maximum_relative_variance_budget_residual']:.3e}")
        print(f"  Order rate       : {orders[-1]:.3f} (finest refinement)")
        print(f"  Dense exp. error : {tracer['dense_exponential_max_abs_error']:.3e}")

    acoustic = results.get('acoustic_example')
    if acoustic:
        report_basis = acoustic['spec']['report_basis_size']
        row = next(r for r in acoustic['convergence'] if r['basis_size'] == report_basis)
        range_row = next(r for r in acoustic['range_convergence']
                         if r['range_step_m'] == acoustic['spec']['report_range_step_m'])
        interpolation_errors = [v for k, v in range_row.items()
                                if k.endswith('_interpolation_max_abs_error')]
        print('\nAcoustic diagnostics')
        print(f"  Modes            : {acoustic['mode_count']}")
        print(f"  Analytic k error : {row['constant_max_relative_k_error']:.3e}")
        print(f"  Basis error      : {row['max_array_mismatch_error_vs_finest']:.3e}")
        print(f"  Range interp. err: {max(interpolation_errors):.3e}")

    sensitivity = results.get('acoustic_sensitivity')
    if sensitivity:
        print(f"  Local metric err : {sensitivity['local_metric_max_relative_error']:.3e}")
        print(f"  Basis sens. error: {sensitivity['sensitivity_max_basis_abs_error']:.3e}")

    metric = results.get('acoustic_metrics')
    if metric:
        print('\nQuadratic metric thresholds')
        for row in metric['checks']:
            print(f"  {row['range_m']/1000:5.1f} km  Threshold={row['norm_ratio_threshold']:.6f}  "
                  f"dQ(step)={row['relative_Q_step_difference']:.3e}  "
                  f"dQ(basis)={row['relative_Q_basis_difference']:.3e}")
        print(f"  Full reversals   : {metric['full_reversals']}/{metric['witness_rows']}")

    physical = results.get('physical_diagnostics')
    if physical:
        residual = physical['inventory_example']['max_abs_corrected_residual']
        print('\nBudget check')
        print(f'  Corrected residual: {residual:.3e}')


def run_tests() -> bool:
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f'\nTests: {result.testsRun} run, {len(result.failures)} failures, '
          f'{len(result.errors)} errors, {len(result.skipped)} skipped')
    return result.wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--tests-only', action='store_true', help='Run the unit tests only.')
    modes.add_argument('--figures-only', action='store_true', help='Regenerate figures from saved results, then test.')
    parser.add_argument('--no-figures', action='store_true', help='Run calculations and tests without plotting.')
    parser.add_argument('--auxiliary', action='store_true',
                        help='Also regenerate source-table and reporting-support files.')
    parser.add_argument('--figure-formats', nargs='+', choices=('pdf', 'png', 'svg'),
                        default=['pdf', 'png', 'svg'])
    args = parser.parse_args()
    if args.no_figures and (args.tests_only or args.figures_only):
        parser.error('--no-figures is only valid for a full numerical run')

    if args.tests_only:
        return 0 if run_tests() else 1

    try:
        results: dict[str, dict] = {}
        if not args.figures_only:
            print('Numerical calculations')
            for name in CORE_MODULES:
                results[name] = run_module(name)
            if args.auxiliary:
                print('\nAuxiliary tables')
                for name in AUXILIARY_MODULES:
                    run_module(name)
            print_numerical_summary(results)

        if not args.no_figures:
            formats = tuple(dict.fromkeys(args.figure_formats))
            print('\nFigures')
            tick = time.perf_counter()
            importlib.import_module('generate_figures').main(formats)
            importlib.import_module('science_figures').main(formats)
            print(f"  {'/'.join(formats)} written in {time.perf_counter() - tick:.3f} s")
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1

    return 0 if run_tests() else 1


if __name__ == '__main__':
    raise SystemExit(main())
