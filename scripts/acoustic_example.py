"""Normal modes for a lossless, range-independent waveguide."""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal, eigh
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from functools import lru_cache
from scipy.special import hankel1
ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Modes:
    """Depth grid, horizontal wavenumbers, depth-normalized modes and frequency."""

    z: NDArray[np.float64]
    k: NDArray[np.float64]
    phi: NDArray[np.float64]
    frequency: float

def reference_speed(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prescribed reference profile c(z), with z in metres and c in m/s."""
    return 1490.0 + 12.0 * np.exp(-z / 35.0) + 0.025 * z

def depth_rms(values: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.trapezoid(values * values, z) / (z[-1] - z[0])))

def perturbation(z: NDArray[np.float64], centre: float, width: float, rms: float) -> NDArray[np.float64]:
    if width <= 0 or rms < 0:
        raise ValueError('width must be positive and rms nonnegative')
    shape = np.exp(-0.5 * ((z - centre) / width) ** 2)
    return shape * (rms / depth_rms(shape, z))

def solve_modes_fd(z: NDArray[np.float64], c: NDArray[np.float64], frequency: float) -> Modes:
    z, c = (np.asarray(z, dtype=float), np.asarray(c, dtype=float))
    if z.ndim != 1 or len(z) < 4 or c.shape != z.shape:
        raise ValueError('z and c must be equal-length one-dimensional arrays')
    if not (np.isfinite(z).all() and np.isfinite(c).all() and np.all(c > 0)):
        raise ValueError('finite coordinates and positive sound speeds required')
    dz = float(z[1] - z[0])
    if dz <= 0 or not np.allclose(np.diff(z), dz, rtol=1e-10, atol=1e-10):
        raise ValueError('a strictly increasing uniform depth grid is required')
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError('frequency must be finite and positive')
    w = 2 * np.pi * frequency
    diagonal = (w / c[1:-1]) ** 2 - 2 / dz ** 2
    off = np.full(len(diagonal) - 1, 1 / dz ** 2)
    lam, vec = eigh_tridiagonal(diagonal, off, select='v', select_range=(0.0, float((w / c.min()) ** 2)), lapack_driver='stebz', check_finite=True)
    order = np.argsort(lam)[::-1]
    k = np.sqrt(lam[order])
    phi = np.zeros((len(z), len(k)))
    phi[1:-1] = vec[:, order] / np.sqrt(dz)
    if len(k) == 0:
        raise ValueError('no propagating mode at the specified frequency')
    return Modes(z, k, phi, frequency)

@lru_cache(maxsize=16)
def quadrature(depth: float, n: int):
    x, weights = roots_legendre(n)
    return ((x + 1) * depth / 2, weights * depth / 2)

def solve_modes(z: NDArray[np.float64], c: NDArray[np.float64], frequency: float, basis_size: int=128) -> Modes:
    """Solve the sine-Galerkin eigenproblem by Gaussian quadrature."""
    z, c = (np.asarray(z, dtype=float), np.asarray(c, dtype=float))
    if z.ndim != 1 or len(z) < 4 or c.shape != z.shape:
        raise ValueError('z and c must be equal-length one-dimensional arrays')
    if not np.isfinite(z).all() or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError('finite arrays and positive sound speed required')
    if z[0] != 0 or np.any(np.diff(z) <= 0) or frequency <= 0 or (not np.isfinite(frequency)):
        raise ValueError('increasing depth from zero and positive frequency required')
    if basis_size < 4:
        raise ValueError('basis_size must be at least four')
    H, w = (float(z[-1]), 2 * np.pi * frequency)
    zq, weights = quadrature(H, max(512, 4 * basis_size))
    cq = CubicSpline(z, c)(zq)
    indices = np.arange(1, basis_size + 1)
    B = np.sqrt(2 / H) * np.sin(np.pi * zq[:, None] * indices[None, :] / H)
    matrix = B.T @ ((weights * (w / cq) ** 2)[:, None] * B)
    matrix -= np.diag((np.pi * indices / H) ** 2)
    lam, vec = eigh(matrix, subset_by_value=(0.0, (w / c.min()) ** 2), driver='evr')
    order = np.argsort(lam)[::-1]
    if len(order) == 0:
        raise ValueError('no propagating mode')
    if len(order) == basis_size:
        raise ValueError('basis too small to separate propagating and evanescent modes')
    eval_basis = np.sqrt(2 / H) * np.sin(np.pi * z[:, None] * indices[None, :] / H)
    phi = eval_basis @ vec[:, order]
    phi[0] = 0.0
    phi[-1] = 0.0
    return Modes(z, np.sqrt(lam[order]), phi, frequency)

def pressure(modes: Modes, ranges: NDArray[np.float64], source_depth: float, receiver_depths: NDArray[np.float64]) -> NDArray[np.complex128]:
    r = np.atleast_1d(np.asarray(ranges, dtype=float))
    zr = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('ranges must be finite and strictly positive')
    if not modes.z[0] < source_depth < modes.z[-1] or np.any(zr <= modes.z[0]) or np.any(zr >= modes.z[-1]) or (not np.isfinite(zr).all()):
        raise ValueError('source and receivers must be strictly within the water column')
    at_source = np.array([np.interp(source_depth, modes.z, modes.phi[:, m]) for m in range(len(modes.k))])
    at_receiver = np.array([np.interp(zr, modes.z, modes.phi[:, m]) for m in range(len(modes.k))])
    radial = hankel1(0, r[:, None] * modes.k[None, :])
    return radial * at_source[None, :] @ at_receiver

def array_mismatch(reference: NDArray[np.complex128], estimate: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Squared receiver-space residual, normalized by estimated field energy."""
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    if reference.shape != estimate.shape or reference.ndim != 2:
        raise ValueError('matching (range, receiver) arrays required')
    norm0 = np.sum(np.abs(reference) ** 2, axis=1)
    norm1 = np.sum(np.abs(estimate) ** 2, axis=1)
    if np.any(norm0 <= 0) or np.any(norm1 <= 0) or not np.isfinite(norm0).all() or not np.isfinite(norm1).all():
        raise ValueError('nonzero finite fields required')
    coefficient = np.sum(reference.conj() * estimate, axis=1) / norm0
    residual = estimate - coefficient[:, None] * reference
    return np.clip(np.sum(np.abs(residual) ** 2, axis=1) / norm1, 0., 1.)

def first_order_delta_k(modes: Modes, c: NDArray[np.float64], dc: NDArray[np.float64]) -> NDArray[np.float64]:
    kernel = modes.phi ** 2 * (dc / c ** 3)[:, None]
    return -(2 * np.pi * modes.frequency) ** 2 / modes.k * np.trapezoid(kernel, modes.z, axis=0)

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def range_grid(start: float, stop: float, step: float) -> NDArray[np.float64]:
    """Uniform, endpoint-inclusive grid; reject a silently shortened last interval."""
    if not np.isfinite([start, stop, step]).all() or start <= 0 or stop <= start or step <= 0:
        raise ValueError('finite positive ranges, positive step and stop > start required')
    intervals = (stop - start) / step
    if not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8):
        raise ValueError('range step must divide the interval exactly')
    return np.linspace(start, stop, round(intervals) + 1)


def mismatch_curve(reference: Modes, estimate: Modes, ranges: NDArray[np.float64],
                   source_depth: float, receivers: NDArray[np.float64],
                   chunk_size: int = 2048) -> NDArray[np.float64]:
    """Evaluate a curve without retaining a full range-by-receiver complex field."""
    ranges = np.asarray(ranges, dtype=float)
    if ranges.ndim != 1 or not len(ranges) or not np.isfinite(ranges).all() or np.any(ranges <= 0):
        raise ValueError('a nonempty one-dimensional array of positive finite ranges is required')
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    result = np.empty(len(ranges), dtype=float)
    for start in range(0, len(ranges), chunk_size):
        part = slice(start, start + chunk_size)
        p0 = pressure(reference, ranges[part], source_depth, receivers)
        p1 = pressure(estimate, ranges[part], source_depth, receivers)
        result[part] = array_mismatch(p0, p1)
    return result


def sampling_diagnostics(ranges: NDArray[np.float64], curves: dict[str, NDArray[np.float64]],
                         steps: list[float]) -> list[dict]:
    """Sampling error relative to the specified fine range grid."""
    r = np.asarray(ranges, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or np.any(np.diff(r) <= 0):
        raise ValueError('increasing finite reference ranges required')
    dr = float(r[1] - r[0])
    if not np.allclose(np.diff(r), dr, atol=1e-8, rtol=0):
        raise ValueError('uniform reference ranges required')
    if not curves or not steps:
        raise ValueError('nonempty curves and step list required')
    for curve in curves.values():
        if np.shape(curve) != r.shape or not np.isfinite(curve).all():
            raise ValueError('finite curves on the reference grid required')
    rows = []
    for step in steps:
        coarse_r = range_grid(float(r[0]), float(r[-1]), float(step))
        ratio = float(step) / dr
        if ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0):
            raise ValueError('each step must be an integer multiple of the reference step')
        stride = round(ratio)
        row = {'range_step_m': float(step), 'range_points': len(coarse_r)}
        for name, curve in curves.items():
            coarse = curve[::stride]
            mean = float(np.trapezoid(coarse, coarse_r) / (r[-1] - r[0]))
            ref_mean = float(np.trapezoid(curve, r) / (r[-1] - r[0]))
            row.update({name + '_mean': mean, name + '_max': float(np.max(coarse)),
                        name + '_mean_abs_error': abs(mean - ref_mean),
                        name + '_max_abs_error': abs(float(np.max(coarse) - np.max(curve))),
                        name + '_interpolation_max_abs_error':
                        float(np.max(np.abs(np.interp(r, coarse_r, coarse) - curve)))})
        rows.append(row)
    return rows


def run(spec_path: Path | None = None) -> dict:
    spec_path = spec_path or ROOT / 'data' / 'acoustic_spec.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    H, f = spec['depth_m'], spec['frequency_hz']
    r = range_grid(spec['range_min_m'], spec['range_max_m'], spec['report_range_step_m'])
    fine_r = range_grid(spec['range_min_m'], spec['range_max_m'], spec['range_reference_step_m'])
    zr = np.arange(spec['receiver_min_depth_m'], spec['receiver_max_depth_m'] + 0.1,
                   spec['receiver_spacing_m'])
    all_results, grids, mode_table, profiles, fine_curves = {}, [], [], [], {}
    for basis_size in spec['basis_sizes']:
        dz = spec['depth_grid_m']
        z = np.linspace(0.0, H, round(H / dz) + 1)
        c = reference_speed(z)
        modes = solve_modes(z, c, f, basis_size)
        curves = {}
        for shape in spec['perturbations']:
            dc = (np.full_like(z, spec['rms_mps']) if shape['name'] == 'uniform'
                  else perturbation(z, shape['centre_m'], shape['width_m'], spec['rms_mps']))
            perturbed = solve_modes(z, c + dc, f, basis_size)
            if len(modes.k) != len(perturbed.k):
                raise RuntimeError('mode cutoff crossed; linear matched-mode comparison invalid')
            curves[shape['name']] = mismatch_curve(modes, perturbed, r, spec['source_depth_m'], zr)
            if basis_size == spec['report_basis_size']:
                fine_curves[shape['name']] = mismatch_curve(
                    modes, perturbed, fine_r, spec['source_depth_m'], zr)
                for j in range(len(z)):
                    profiles.append(dict(perturbation=shape['name'], depth_m=z[j],
                                         reference_speed_mps=c[j], delta_speed_mps=dc[j]))
                dk = first_order_delta_k(modes, c, dc)
                for j in range(len(modes.k)):
                    mode_table.append(dict(perturbation=shape['name'], mode=j + 1,
                        k_reference_per_m=modes.k[j],
                        delta_k_exact_per_m=perturbed.k[j] - modes.k[j],
                        delta_k_linear_per_m=dk[j]))
        all_results[basis_size] = curves
        constant = solve_modes(z, np.full_like(z, 1500.0), f, basis_size)
        m = np.arange(1, len(constant.k) + 1)
        exact = np.sqrt((2 * np.pi * f / 1500.0) ** 2 - (m * np.pi / H) ** 2)
        grids.append(dict(basis_size=basis_size, depth_grid_m=dz, modes=len(modes.k),
                          constant_modes=len(exact), range_points=len(r),
                          range_step_m=spec['report_range_step_m'],
                          constant_max_relative_k_error=float(np.max(np.abs(constant.k - exact) / exact))))
    finest = max(spec['basis_sizes'])
    for record in grids:
        nb = record['basis_size']
        record['max_array_mismatch_error_vs_finest'] = float(max(
            np.max(np.abs(all_results[nb][s] - all_results[finest][s])) for s in all_results[nb]))
    report = all_results[spec['report_basis_size']]
    result_rows = [dict(range_m=float(x), **{name: float(curve[i]) for name, curve in report.items()})
                   for i, x in enumerate(r)]
    summaries = []
    for name, curve in report.items():
        summaries.append(dict(perturbation=name, rms_mps=spec['rms_mps'],
            range_10km_mismatch=float(np.interp(10000, r, curve)),
            range_20km_mismatch=float(np.interp(20000, r, curve)),
            range_30km_mismatch=float(np.interp(30000, r, curve)),
            range_mean_mismatch=float(np.trapezoid(curve, r) / (r[-1] - r[0])),
            max_mismatch=float(curve.max())))
    range_convergence = sampling_diagnostics(fine_r, fine_curves, spec['range_steps_m'])
    out = ROOT / 'results'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'acoustic_profiles.csv', profiles)
    write_csv(out / 'acoustic_range_curves.csv', result_rows)
    write_csv(out / 'acoustic_modal_sensitivity.csv', mode_table)
    write_csv(out / 'acoustic_convergence.csv', grids)
    write_csv(out / 'acoustic_summary.csv', summaries)
    write_csv(out / 'acoustic_range_convergence.csv', range_convergence)
    # Fine range grid for plotting and convergence checks.
    np.savez_compressed(out / 'acoustic_range_reference.npz', range_m=fine_r, **fine_curves)
    info = {'kind': 'deterministic_idealised_numerical_illustration', 'field_data': False,
            'trained_model': False, 'mode_count': len(modes.k), 'receiver_count': len(zr),
            'spec': spec, 'summaries': summaries, 'convergence': grids,
            'range_convergence': range_convergence,
            'reference_scope': '0.25 m numerical range reference at report basis; not exact or field truth'}
    (out / 'acoustic_results.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    return info


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('mode_count', 'receiver_count', 'summaries',
                                           'convergence', 'range_convergence')}, indent=2))
