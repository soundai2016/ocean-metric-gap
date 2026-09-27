"""Run the normal-mode acoustic experiments and metric checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = str(ROOT / 'scripts')
if scripts not in sys.path:
    sys.path.insert(0, scripts)

import acoustic_example
import acoustic_metrics
import acoustic_sensitivity


def main() -> None:
    base = acoustic_example.run()
    sensitivity = acoustic_sensitivity.run()
    metric = acoustic_metrics.run()

    report_basis = base['spec']['report_basis_size']
    row = next(r for r in base['convergence'] if r['basis_size'] == report_basis)
    range_row = next(r for r in base['range_convergence']
                     if r['range_step_m'] == base['spec']['report_range_step_m'])
    interpolation_errors = [v for k, v in range_row.items()
                            if k.endswith('_interpolation_max_abs_error')]

    print('Normal-mode acoustics')
    print(f"Modes             : {base['mode_count']}")
    print(f"Analytic k error  : {row['constant_max_relative_k_error']:.3e}")
    print(f"Basis error       : {row['max_array_mismatch_error_vs_finest']:.3e}")
    print(f"Range interp. err : {max(interpolation_errors):.3e}")
    print(f"Local metric err  : {sensitivity['local_metric_max_relative_error']:.3e}")
    print(f"Basis sens. error : {sensitivity['sensitivity_max_basis_abs_error']:.3e}")
    print('\nMetric thresholds')
    for check in metric['checks']:
        print(f"{check['range_m']/1000:5.1f} km  Threshold={check['norm_ratio_threshold']:.6f}  "
              f"dQ(step)={check['relative_Q_step_difference']:.3e}  "
              f"dQ(basis)={check['relative_Q_basis_difference']:.3e}")


if __name__ == '__main__':
    main()
