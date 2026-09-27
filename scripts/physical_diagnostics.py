"""Energy increments and finite-volume inventory residuals."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike

ROOT = Path(__file__).resolve().parents[1]


def kinetic_increment(velocity: ArrayLike, increment: ArrayLike,
                      cell_mass: ArrayLike, interval_seconds: float) -> dict[str, float]:
    """Fixed-mass kinetic-energy increment; velocity in m/s, mass in kg."""
    u, du, mass = (np.asarray(a, dtype=float) for a in (velocity, increment, cell_mass))
    if u.ndim != 2 or not u.size or du.shape != u.shape or mass.shape != (u.shape[0],):
        raise ValueError('matching nonempty (cell, component) velocities and one mass per cell required')
    if not all(np.isfinite(a).all() for a in (u, du, mass)) or np.any(mass <= 0):
        raise ValueError('finite inputs and strictly positive cell masses required')
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError('finite positive update interval required')
    linear = float(np.sum(mass[:, None] * u * du))
    quadratic = float(0.5 * np.sum(mass[:, None] * du * du))
    direct = float(0.5 * np.sum(mass[:, None] * ((u + du) ** 2 - u ** 2)))
    return {'linear_energy_j': linear, 'quadratic_energy_j': quadratic,
            'total_energy_j': linear + quadratic, 'direct_energy_j': direct,
            'effective_power_w': (linear + quadratic) / interval_seconds}


def budget_residual(times_seconds: ArrayLike, inventory: ArrayLike,
                    inward_flux: ArrayLike, applied_increments: ArrayLike | None = None) -> dict:
    """Budget residuals for interval-mean inward flux and discrete increments."""
    t, x, f = (np.asarray(a, dtype=float) for a in (times_seconds, inventory, inward_flux))
    if t.ndim != 1 or len(t) < 2 or x.shape != t.shape or f.shape != (len(t) - 1,):
        raise ValueError('n times/inventories and n-1 interval-mean inward fluxes required')
    inc = np.zeros_like(f) if applied_increments is None else np.asarray(applied_increments, dtype=float)
    if inc.shape != f.shape or not all(np.isfinite(a).all() for a in (t, x, f, inc)):
        raise ValueError('matching finite arrays required')
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('times must increase strictly')
    physical = np.diff(x) - f * dt
    corrected = physical - inc
    return {'physical_only_residual': physical.tolist(),
            'corrected_residual': corrected.tolist(),
            'cumulative_physical_only_residual': np.cumsum(physical).tolist(),
            'cumulative_corrected_residual': np.cumsum(corrected).tolist(),
            'cumulative_applied_increment': np.cumsum(inc).tolist(),
            'max_abs_corrected_residual': float(np.max(np.abs(corrected)))}


def run() -> dict:
    # Opposite du cancels linear work, not |du|^2.
    kick = kinetic_increment([[1., 0.], [1., 0.]], [[0.1, 0.], [-0.1, 0.]], [2., 2.], 10.)
    # Source = 2 W; corrections = +3, -1 J.
    budget = budget_residual([0., 10., 20.], [100., 123., 142.], [2., 2.], [3., -1.])
    result = {'kind': 'manufactured_diagnostic_checks', 'new_ocean_simulations': False,
              'kinetic_increment_example': kick, 'inventory_example': budget,
              'assumptions': ['fixed cell masses for kinetic update',
                              'interval-average inward flux, not point samples',
                              'correction increments reported separately from physical sources']}
    path = ROOT / 'results' / 'physical_diagnostics.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
