"""Amplitude, frequency and source-depth sweeps for the prescribed waveguide."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from acoustic_example import (reference_speed, perturbation, solve_modes, pressure,
                              array_mismatch, range_grid, mismatch_curve, write_csv)
ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    base = json.loads((ROOT / 'data/acoustic_spec.json').read_text())
    spec = json.loads((ROOT / 'data/acoustic_sensitivity_spec.json').read_text())
    z = np.linspace(0, base['depth_m'], round(base['depth_m'] / base['depth_grid_m']) + 1)
    c = reference_speed(z)
    zr = np.arange(base['receiver_min_depth_m'], base['receiver_max_depth_m'] + .1,
                   base['receiver_spacing_m'])
    shapes = {'uniform': np.ones_like(z),
              'localized': perturbation(z, 45., 15., 1.)}
    points = np.array(spec['point_ranges_m'])
    grid = spec['amplitude_grid_mps']
    amplitudes = np.linspace(grid['start'], grid['stop'],
                            round((grid['stop'] - grid['start']) / grid['step']) + 1)
    ref = solve_modes(z, c, base['frequency_hz'], spec['report_basis_size'])
    p0 = pressure(ref, points, base['source_depth_m'], zr)
    amplitude_rows = []
    for shape, unit in shapes.items():
        for amp in amplitudes:
            other = solve_modes(z, c + amp * unit, base['frequency_hz'], spec['report_basis_size'])
            d = array_mismatch(p0, pressure(other, points, base['source_depth_m'], zr))
            for r, value in zip(points, d):
                amplitude_rows.append(dict(perturbation=shape, rms_mps=float(amp),
                    range_m=float(r), mismatch=float(value), modes=len(other.k)))
    contrast_rows, contrast_curves = [], {}
    r = range_grid(base['range_min_m'], base['range_max_m'], spec['range_step_m'])
    rf = range_grid(base['range_min_m'], base['range_max_m'], spec['range_reference_step_m'])
    check_ref = solve_modes(z, c, base['frequency_hz'], spec['check_basis_size'])
    for shape, unit in shapes.items():
        amp = spec['contrast'][shape + '_rms_mps']
        other = solve_modes(z, c + amp * unit, base['frequency_hz'], spec['report_basis_size'])
        fine = mismatch_curve(ref, other, rf, base['source_depth_m'], zr)
        stride = round(spec['range_step_m'] / spec['range_reference_step_m'])
        curve = fine[::stride]
        check_other = solve_modes(z, c + amp * unit, base['frequency_hz'], spec['check_basis_size'])
        check_curve = mismatch_curve(check_ref, check_other, r, base['source_depth_m'], zr)
        point_values = array_mismatch(p0, pressure(other, points, base['source_depth_m'], zr))
        contrast_curves[shape] = curve
        contrast_rows.append(dict(perturbation=shape, rms_mps=amp,
            range_10km_mismatch=float(point_values[0]), range_20km_mismatch=float(point_values[1]),
            range_30km_mismatch=float(point_values[2]),
            range_mean_mismatch=float(np.trapezoid(curve, r) / (r[-1] - r[0])),
            fine_range_mean_mismatch=float(np.trapezoid(fine, rf) / (rf[-1] - rf[0])),
            basis_max_abs_error=float(np.max(np.abs(curve - check_curve))),
            range_interpolation_max_abs_error=float(np.max(np.abs(np.interp(rf, r, curve) - fine)))))
    sensitivity_rows = []
    for frequency in spec['frequencies_hz']:
        modes = solve_modes(z, c, frequency, spec['report_basis_size'])
        checked = solve_modes(z, c, frequency, spec['check_basis_size'])
        for source_depth in spec['source_depths_m']:
            a0 = pressure(modes, points, source_depth, zr)
            ac = pressure(checked, points, source_depth, zr)
            for shape, unit in shapes.items():
                amp = spec['contrast'][shape + '_rms_mps']
                other = solve_modes(z, c + amp * unit, frequency, spec['report_basis_size'])
                check_other = solve_modes(z, c + amp * unit, frequency, spec['check_basis_size'])
                values = array_mismatch(a0, pressure(other, points, source_depth, zr))
                checks = array_mismatch(ac, pressure(check_other, points, source_depth, zr))
                for i, distance in enumerate(points):
                    sensitivity_rows.append(dict(frequency_hz=frequency, source_depth_m=source_depth,
                        range_m=float(distance), perturbation=shape, rms_mps=amp,
                        mismatch=float(values[i]), basis_abs_error=float(abs(values[i] - checks[i])),
                        reference_modes=len(modes.k), perturbed_modes=len(other.k)))
    # Local metric: centered derivative with common-gain projection.
    local_rows = []
    derivative_step = spec['local_metric_check']['derivative_step_mps']
    norm = np.sum(np.abs(p0) ** 2, axis=1)
    for shape, unit in shapes.items():
        plus = pressure(solve_modes(z, c + derivative_step * unit, base['frequency_hz'],
                         spec['report_basis_size']), points, base['source_depth_m'], zr)
        minus = pressure(solve_modes(z, c - derivative_step * unit, base['frequency_hz'],
                          spec['report_basis_size']), points, base['source_depth_m'], zr)
        derivative = (plus - minus) / (2 * derivative_step)
        perpendicular = derivative - p0 * (np.sum(p0.conj() * derivative, axis=1) / norm)[:, None]
        coefficient = np.sum(np.abs(perpendicular) ** 2, axis=1) / norm
        for amplitude in spec['local_metric_check']['amplitudes_mps']:
            estimate = pressure(solve_modes(z, c + amplitude * unit, base['frequency_hz'],
                                spec['report_basis_size']), points, base['source_depth_m'], zr)
            exact = array_mismatch(p0, estimate)
            quadratic = amplitude ** 2 * coefficient
            for i, distance in enumerate(points):
                local_rows.append(dict(perturbation=shape, rms_mps=amplitude, range_m=float(distance),
                    exact_mismatch=float(exact[i]), quadratic_mismatch=float(quadratic[i]),
                    relative_error=float(abs(exact[i] - quadratic[i]) / exact[i])))
    out = ROOT / 'results'
    write_csv(out / 'acoustic_local_metric.csv', local_rows)
    write_csv(out / 'acoustic_amplitude_sweep.csv', amplitude_rows)
    write_csv(out / 'acoustic_contrast.csv', contrast_rows)
    write_csv(out / 'acoustic_operator_sensitivity.csv', sensitivity_rows)
    write_csv(out / 'acoustic_contrast_curves.csv', [dict(range_m=float(v),
        **{name: float(curve[i]) for name, curve in contrast_curves.items()}) for i, v in enumerate(r)])
    index = {(row['frequency_hz'], row['source_depth_m'], row['range_m'], row['perturbation']):
             row['mismatch'] for row in sensitivity_rows}
    # Deterministic configuration count; not a probability estimate.
    reversals = sum(index[(f, s, float(d), 'uniform')] > index[(f, s, float(d), 'localized')]
                    for f in spec['frequencies_hz'] for s in spec['source_depths_m'] for d in points)
    result = {'kind': 'deterministic_idealised_operator_sensitivity', 'field_data': False,
        'trained_model': False, 'spec': spec, 'amplitude_rows': len(amplitude_rows),
        'local_metric_max_relative_error': max(row['relative_error'] for row in local_rows),
        'operator_comparisons': len(sensitivity_rows) // 2,
        'illustrative_field_task_reversals': int(reversals),
        'sensitivity_max_basis_abs_error': max(x['basis_abs_error'] for x in sensitivity_rows),
        'contrast': contrast_rows}
    (out / 'acoustic_sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
