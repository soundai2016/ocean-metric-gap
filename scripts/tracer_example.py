"""Periodic advection-diffusion with conservative finite-volume fluxes."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.special import i0e, i1e
from acoustic_example import write_csv
from error_metrics import conservative_projection, restricted_metrics, projection_diagnostics
from quadratic_order import inspect_metrics

ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_DAY = 86400.


def grid(size: int, length: float) -> NDArray[np.float64]:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 8:
        raise ValueError('grid size must be an integer >= 8')
    if not np.isfinite(length) or length <= 0:
        raise ValueError('domain length must be finite and positive')
    return (np.arange(size) + .5) * length / size


def symbol(size: int, length: float, velocity: float, diffusivity: float) -> NDArray:
    grid(size, length)
    if not np.isfinite([velocity, diffusivity]).all() or diffusivity < 0:
        raise ValueError('finite velocity and nonnegative diffusivity required')
    dx = length / size
    phase = 2 * np.pi * np.fft.fftfreq(size)
    return -1j * velocity * np.sin(phase) / dx - 4 * diffusivity * np.sin(phase / 2)**2 / dx**2


def evolve(values: ArrayLike, times: ArrayLike, length: float,
           velocity: float, diffusivity: float) -> NDArray[np.float64]:
    q = np.asarray(values, dtype=float)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if q.ndim != 1 or not np.isfinite(q).all() or t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError('finite one-dimensional state and nonnegative times required')
    lam = symbol(len(q), length, velocity, diffusivity)
    return np.fft.ifft(np.exp(t[:, None] * lam) * np.fft.fft(q)[None, :], axis=1).real


def tendency(values: ArrayLike, length: float, velocity: float,
             diffusivity: float) -> NDArray[np.float64]:
    q = np.asarray(values, dtype=float)
    symbol(len(q), length, velocity, diffusivity)
    dx = length / len(q)
    flux = velocity * (q + np.roll(q, -1)) / 2 - diffusivity * (np.roll(q, -1) - q) / dx
    return -(flux - np.roll(flux, 1)) / dx


def receiver(x: ArrayLike, length: float, centre: float, concentration: float) -> NDArray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite([length, centre, concentration]).all() or length <= 0 or concentration <= 0:
        raise ValueError('finite coordinates and positive length/concentration required')
    exponent = concentration * np.cos(2 * np.pi * (x - centre) / length)
    w = np.exp(exponent - exponent.max())
    return w / w.sum()


def analytic_gain(spec: dict) -> float:
    k = 2 * np.pi / spec['length_m']
    decay = np.exp(-spec['diffusivity_m2ps'] * k*k * spec['target_days'] * SECONDS_PER_DAY)
    concentration = spec['receiver_concentration']
    return float(np.sqrt(2) * decay * i1e(concentration) / i0e(concentration))


def setup(spec: dict, size: int) -> dict:
    length, velocity = spec['length_m'], spec['velocity_mps']
    target = spec['target_days'] * SECONDS_PER_DAY
    x = grid(size, length)
    k = 2 * np.pi / length
    phase = k * (x + velocity * target - spec['receiver_centre_m'])
    a = spec['amplitude_a'] * np.sqrt(2) * np.cos(phase)
    b = spec['amplitude_b'] * np.sqrt(2) * np.cos(phase + np.arccos(spec['phase_b_cosine']))
    bias = -spec['bias_cancellation_fraction'] * spec['amplitude_a'] * analytic_gain(spec)
    reference = spec['background_concentration'] + spec['reference_cosine_amplitude'] * np.cos(k*x)
    return {'x': x, 'reference': reference, 'A': a, 'B': b,
            'before_projection': a + bias, 'after_projection': a + bias - np.mean(a + bias),
            'bias': bias, 'weights': receiver(x, length, spec['receiver_centre_m'], spec['receiver_concentration'])}


def run() -> dict:
    spec = json.loads((ROOT / 'data/tracer_spec.json').read_text())
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    length, velocity, diffusion = (spec[k] for k in ('length_m', 'velocity_mps', 'diffusivity_m2ps'))
    target = spec['target_days'] * SECONDS_PER_DAY
    gain = analytic_gain(spec)
    exact = {'A': spec['amplitude_a'] * gain,
             'B': spec['amplitude_b'] * spec['phase_b_cosine'] * gain}
    exact['after_projection'] = exact['A']
    exact['before_projection'] = (1-spec['bias_cancellation_fraction']) * exact['A']
    convergence, rows, budget_rows, profile_rows, time_rows = [], [], [], [], []
    dense_error = None
    metric_data = {}
    lo, hi, step = spec['output_days']
    days = np.linspace(lo, hi, round((hi-lo)/step)+1)
    for size in spec['grid_sizes']:
        s = setup(spec, size)
        max_output_error = 0.
        for name in ('A','B','before_projection','after_projection'):
            e = s[name]
            forecast = s['reference'] + e
            evolved = evolve(forecast, days * SECONDS_PER_DAY, length, velocity, diffusion)
            evolved_error = evolve(e, [target], length, velocity, diffusion)[0]
            output = float(s['weights'] @ evolved_error)
            err = abs(output - exact[name]); max_output_error = max(max_output_error, err)
            mass = length * evolved.mean(axis=1)
            rel_mass = float(np.max(np.abs(mass - length*forecast.mean())) / abs(length*forecast.mean()))
            variance_residuals = []
            for state in evolved:
                dot = tendency(state, length, velocity, diffusion)
                dx = length / size
                rate = dx * np.dot(state-state.mean(), dot)
                dissipation = -diffusion / dx * np.sum((np.roll(state,-1)-state)**2)
                variance_residuals.append(abs(rate-dissipation) / max(abs(dissipation), np.finfo(float).tiny))
            budget_rows.append({'grid_size':size, 'case':name,
                'relative_mass_drift':rel_mass, 'relative_variance_budget_residual':max(variance_residuals),
                'minimum_concentration':float(evolved.min()), 'initial_mean_error':float(e.mean())})
            if size == spec['report_grid_size']:
                rows.append({'case':name, 'initial_field_rms':float(np.sqrt(np.mean(e*e))),
                    'initial_mean_error':float(e.mean()), 'output_error':output, 'squared_output_error':output*output,
                    'analytic_output_error':exact[name], 'output_discretization_error':err,
                    'relative_mass_drift':rel_mass})
                ev_error=evolve(e,days*SECONDS_PER_DAY,length,velocity,diffusion)
                for d, arr in zip(days,ev_error):
                    time_rows.append({'case':name,'lead_days':float(d),'output_error':float(s['weights']@arr),
                        'field_rms':float(np.sqrt(np.mean(arr*arr))),'mean_error':float(arr.mean())})
                for j in range(size):
                    profile_rows.append({'case':name,'x_m':float(s['x'][j]),'initial_error':float(e[j]),
                        'error_at_target':float(evolved_error[j]),'receiver_weight':float(s['weights'][j])})
        convergence.append({'grid_size':size,'grid_spacing_m':length/size,
            'max_output_error':max_output_error})
        if size == spec['dense_check_grid_size']:
            identity=np.eye(size)
            mat=np.column_stack([tendency(identity[:,j],length,velocity,diffusion) for j in range(size)])
            dense_error=float(np.max(np.abs(expm(target*mat) @ (s['reference']+s['A'])-
                evolve(s['reference']+s['A'],[target],length,velocity,diffusion)[0])))
        if size == spec['report_grid_size']:
            # First-harmonic basis: W = I.
            k=2*np.pi/length
            basis=np.column_stack([np.ones(size),np.sqrt(2)*np.cos(k*s['x']),np.sqrt(2)*np.sin(k*s['x'])])
            g=np.array([s['weights']@evolve(basis[:,j],[target],length,velocity,diffusion)[0] for j in range(3)])
            q=np.outer(g,g); w=basis.T@basis/size; c=np.array([[1.,0.,0.]])
            n, qn, wn=restricted_metrics(q,w,c)
            metric_data={'Q':q.tolist(),'W':w.tolist(),'C':c.tolist(),'N':n.tolist(),
                'restricted':inspect_metrics(qn,wn),'projection':projection_diagnostics(q,w,c)}
    for previous, current in zip(convergence[:-1],convergence[1:]):
        current['order_from_previous']=float(np.log(previous['max_output_error']/current['max_output_error'])/np.log(2))
    convergence[0]['order_from_previous']=None
    write_csv(out/'tracer_summary.csv',rows)
    write_csv(out/'tracer_convergence.csv',convergence)
    write_csv(out/'tracer_budgets.csv',budget_rows)
    write_csv(out/'tracer_profiles.csv',profile_rows)
    write_csv(out/'tracer_time_series.csv',time_rows)
    byname={r['case']:r for r in rows}
    result={'spec':spec,'analytic_gain':gain,'analytic_output_errors':exact,
        'summary':rows,'convergence':convergence,'metrics':metric_data,
        'dense_exponential_max_abs_error':dense_error,
        'maximum_relative_mass_drift':max(r['relative_mass_drift'] for r in budget_rows),
        'maximum_relative_variance_budget_residual':max(r['relative_variance_budget_residual'] for r in budget_rows),
        'minimum_concentration':min(r['minimum_concentration'] for r in budget_rows),
        'field_rms_ratio_A_over_B':byname['A']['initial_field_rms']/byname['B']['initial_field_rms'],
        'output_error_ratio_A_over_B':byname['A']['output_error']/byname['B']['output_error'],
        'projection_field_rms_ratio':byname['after_projection']['initial_field_rms']/byname['before_projection']['initial_field_rms'],
        'projection_squared_output_ratio':byname['after_projection']['squared_output_error']/byname['before_projection']['squared_output_error'],
        'origin':'new deterministic calculation; no trained predictor or forecast archive'}
    (out/'tracer_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
