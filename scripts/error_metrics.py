"""Error metrics on invariant subspaces and conservative projection tests."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh, null_space
from quadratic_order import _metrics


def constraint_matrix(raw: ArrayLike, dimension: int) -> NDArray[np.float64]:
    """Return a finite, real, full-row-rank constraint matrix C."""
    if np.iscomplexobj(raw):
        raise ValueError('constraints must be real')
    c = np.asarray(raw, dtype=float)
    if c.ndim != 2 or c.shape[1] != dimension or not np.isfinite(c).all():
        raise ValueError('C must have shape (constraints, state dimension)')
    if c.shape[0] >= dimension:
        raise ValueError('constraints must leave at least one admissible direction')
    if c.shape[0] and np.linalg.matrix_rank(c) != c.shape[0]:
        raise ValueError('constraint rows must be linearly independent')
    return c


def restricted_metrics(Q: ArrayLike, W: ArrayLike, C: ArrayLike,
                       *, rtol: float = 1e-10) -> tuple[NDArray, NDArray, NDArray]:
    """Return N, N.T Q N and N.T W N, where columns of N span ker(C)."""
    q, w, _, _ = _metrics(Q, W, rtol)
    c = constraint_matrix(C, len(w))
    n = null_space(c) if len(c) else np.eye(len(w))
    return n, n.T @ q @ n, n.T @ w @ n


def conservative_projection(W: ArrayLike, C: ArrayLike,
                            *, rtol: float = 1e-10) -> NDArray[np.float64]:
    """W-orthogonal projection onto zero-inventory errors C e = 0."""
    w0 = np.asarray(W)
    _, w, _, _ = _metrics(np.zeros_like(w0), w0, rtol)
    c = constraint_matrix(C, len(w))
    if not len(c):
        return np.eye(len(w))
    a = np.linalg.solve(w, c.T)
    return np.eye(len(w)) - a @ np.linalg.solve(c @ a, c)


def projection_diagnostics(Q: ArrayLike, W: ArrayLike, C: ArrayLike,
                           *, rtol: float = 1e-10) -> dict:
    """Check the sign of Pi A Pi - A in W-whitened coordinates."""
    q, w, _, _ = _metrics(Q, W, rtol)
    p = conservative_projection(w, C, rtol=rtol)
    d, v = eigh(w)
    root = (v * np.sqrt(d)) @ v.T
    invroot = (v / np.sqrt(d)) @ v.T
    a = invroot @ q @ invroot
    pi = root @ p @ invroot
    pi = (pi + pi.T) / 2
    delta = pi @ a @ pi - a
    values, vectors = eigh((delta + delta.T) / 2)
    scale = max(float(np.linalg.norm(a, 2)), np.finfo(float).tiny)
    maximum = float(values[-1])
    witness = invroot @ vectors[:, -1]
    return {
        'commutator_relative_norm': float(np.linalg.norm(pi @ a - a @ pi, 2) / scale),
        'maximum_loss_increase_per_unit_field_squared': maximum,
        'nonincreasing_within_tolerance': bool(maximum <= rtol * scale),
        'witness': witness.tolist(),
        'witness_field_squared_before': float(witness @ w @ witness),
        'witness_field_squared_after': float((p @ witness) @ w @ (p @ witness)),
        'witness_downstream_before': float(witness @ q @ witness),
        'witness_downstream_after': float((p @ witness) @ q @ (p @ witness)),
        'relative_tolerance': rtol,
    }


def ranking_bound(Q: ArrayLike, W: ArrayLike, norm_a: float,
                  norm_b: float, *, rtol: float = 1e-10) -> dict:
    """Sharp direction-independent quadratic ranking test at prescribed norms."""
    if not np.isfinite([norm_a, norm_b]).all() or norm_a < 0 or norm_b <= 0:
        raise ValueError('finite norm_a >= 0 and norm_b > 0 required')
    _, _, values, _ = _metrics(Q, W, rtol)
    low, high = max(0., float(values[0])), max(0., float(values[-1]))
    margin = low * norm_b ** 2 - high * norm_a ** 2
    scale = max(low * norm_b ** 2, high * norm_a ** 2, np.finfo(float).tiny)
    return {'lambda_min': low, 'lambda_max': high,
            'norm_ratio_threshold': float(np.sqrt(low / high)) if high > 0 else None,
            'upper_downstream_a': high * norm_a ** 2,
            'lower_downstream_b': low * norm_b ** 2,
            'strictly_ordered_for_all_directions': bool(margin > rtol * scale),
            'quadratic_margin': margin, 'relative_tolerance': rtol}
