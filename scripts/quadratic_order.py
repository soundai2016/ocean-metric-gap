"""Generalized-eigenvalue tests for quadratic error measures Q and W."""
from __future__ import annotations

import json
from pathlib import Path
from numbers import Real
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]


def _metrics(Q: ArrayLike, W: ArrayLike, rtol: float) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    if isinstance(rtol, bool) or not isinstance(rtol, Real) or not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('rtol must be a finite number strictly between zero and one')
    matrices = []
    for name, raw in (('Q', Q), ('W', W)):
        a = np.asarray(raw)
        if np.iscomplexobj(a):
            raise ValueError(f'{name} must be a real matrix')
        try:
            a = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must be numerical') from error
        if a.ndim != 2 or not a.shape[0] or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
            raise ValueError(f'{name} must be a nonempty finite square matrix')
        scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
        if np.max(np.abs(a - a.T)) > rtol * scale:
            raise ValueError(f'{name} must be symmetric')
        matrices.append((a + a.T) / 2)
    q, w = matrices
    if q.shape != w.shape:
        raise ValueError('Q and W must have matching dimensions')
    try:
        values, vectors = eigh(q, w, check_finite=True)
    except np.linalg.LinAlgError as error:
        raise ValueError('W must be positive definite') from error
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values[0] < -rtol * scale:
        raise ValueError('Q must be positive semidefinite')
    return q, w, values, vectors


def inspect_metrics(Q: ArrayLike, W: ArrayLike, *, rtol: float = 1e-10) -> dict[str, Any]:
    """Return the generalized spectrum and proportionality test at the given rtol."""
    _, _, values, _ = _metrics(Q, W, rtol)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    proportional = bool(values[-1] - values[0] <= rtol * scale)
    return {'generalized_eigenvalues': values.tolist(), 'relative_tolerance': rtol,
            'proportional_within_tolerance': proportional,
            'positive_proportional_metric': proportional and bool(values[0] > 0),
            'resolved_anisotropy': not proportional}


def reversal_witness(Q: ArrayLike, W: ArrayLike, *, amplitude: float = 1.,
                     rtol: float = 1e-10) -> dict[str, Any]:
    """Construct E(a)<E(b) but D(a)>D(b) from extreme eigendirections."""
    if isinstance(amplitude, bool) or not isinstance(amplitude, Real) or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('amplitude must be finite and positive')
    q, w, values, vectors = _metrics(Q, W, rtol)
    low, high = float(values[0]), float(values[-1])
    if high <= 0 or high - low <= rtol * high:
        raise ValueError('no anisotropy is resolved at the stated tolerance')
    ratio = np.sqrt((1. + max(low, 0.) / high) / 2.)
    a = amplitude * ratio * vectors[:, -1]
    b = amplitude * vectors[:, 0]
    e_a, e_b = float(a @ w @ a), float(b @ w @ b)
    d_a, d_b = float(a @ q @ a), float(b @ q @ b)
    if not e_a < e_b or not d_a > d_b:
        raise ValueError('floating-point arithmetic does not resolve the constructed strict inequalities')
    return {'a': a.tolist(), 'b': b.tolist(), 'field_squared_a': e_a, 'field_squared_b': e_b,
            'downstream_a': d_a, 'downstream_b': d_b, 'amplitude': float(amplitude),
            'relative_tolerance': rtol}


def run() -> dict[str, Any]:
    transform = np.array([[1., .3], [.2, 1.7]])
    cases = [
        ('isotropic', 3 * np.eye(2), np.eye(2)),
        ('anisotropic', np.diag([1., 4.]), np.eye(2)),
        ('semidefinite', np.diag([0., 4.]), np.eye(2)),
        ('zero_response', np.zeros((2, 2)), np.eye(2)),
        ('changed_coordinates', transform.T @ np.diag([1., 4.]) @ transform,
         transform.T @ transform),
    ]
    records = []
    for name, q, w in cases:
        record = {'case': name, 'Q': q.tolist(), 'W': w.tolist(), **inspect_metrics(q, w)}
        if record['resolved_anisotropy']:
            record['witnesses'] = [reversal_witness(q, w, amplitude=a) for a in (1., .01, .0001)]
        records.append(record)
    result = {'scope': 'manufactured_quadratic_metrics', 'ocean_forecast_fields_used': False,
              'records': records}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/quadratic_order_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
