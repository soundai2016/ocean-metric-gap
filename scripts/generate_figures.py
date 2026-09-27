"""Plot the published closure errors and computed acoustic diagnostics."""
from __future__ import annotations
import csv
from typing import Iterable
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.hashsalt'] = 'ocean-modelling'
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'

def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def finish(fig, name: str, formats: Iterable[str]) -> None:
    fig.tight_layout(pad=1.0)
    for extension in formats:
        if extension not in {'pdf', 'svg', 'png'}:
            raise ValueError(f'Unsupported figure format: {extension}')
        options = {'dpi': 300} if extension == 'png' else {}
        if extension == 'pdf':
            options['metadata'] = {'Creator': 'Matplotlib', 'CreationDate': None, 'ModDate': None}
        elif extension == 'svg':
            options['metadata'] = {'Date': None}
        fig.savefig(FIG / f'{name}.{extension}', **options)
    plt.close(fig)

def main(formats: tuple[str, ...]=('pdf',)) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = rows(ROOT / 'data' / 'closure_benchmark.csv')
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    x = np.arange(len(data))
    for key, label, marker, style in [('no_parameterization_rmse_m', 'No learned forcing', 'o', '-'), ('unscaled_rmse_m', 'Unscaled learned forcing', 's', '--'), ('ke_tuned_rmse_m', 'Kinetic-energy-tuned forcing', '^', '-.')]:
        ax.plot(x, [float(r[key]) for r in data], label=label, marker=marker, linestyle=style, lw=1.6, ms=5)
    ax.set_xticks(x, [f"1/{r['grid_denominator']}" for r in data])
    ax.set_xlabel('Horizontal grid configuration (degree)')
    ax.set_ylabel('Time-mean SSH RMSE (m)')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'closure_published_comparison', formats)
    profiles = rows(ROOT / 'results' / 'acoustic_profiles.csv')
    fig, ax = plt.subplots(figsize=(5.1, 4.2))
    for name, label, style in [('uniform', 'Uniform bias', '-'), ('localized', 'Localized upper-water error', '--')]:
        subset = [r for r in profiles if r['perturbation'] == name]
        ax.plot([float(r['delta_speed_mps']) for r in subset], [float(r['depth_m']) for r in subset], label=label, linestyle=style, lw=1.8)
    ax.invert_yaxis()
    ax.set_xlabel('Sound-speed perturbation (m/s)')
    ax.set_ylabel('Depth (m)')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_perturbations', formats)
    data = rows(ROOT / 'results' / 'acoustic_range_curves.csv')
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for name, label, style in [('uniform', 'Uniform bias', '-'), ('localized', 'Localized upper-water error', '--')]:
        ax.plot([float(r['range_m']) / 1000 for r in data], [float(r[name]) for r in data], label=label, linestyle=style, lw=0.7)
    ax.set(xlabel='Source-receiver range (km)', ylabel='Array mismatch $1-\\gamma^2$', ylim=(0, 1), xlim=(0.5, 30))
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_array_mismatch', formats)
    data = rows(ROOT / 'results' / 'acoustic_modal_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for name, label in [('uniform', 'Uniform'), ('localized', 'Localized')]:
        part = [r for r in data if r['perturbation'] == name]
        ax.plot([int(r['mode']) for r in part], [20000 * float(r['delta_k_exact_per_m']) for r in part], lw=1.4, label=label + ' / recomputed')
        ax.plot([int(r['mode']) for r in part], [20000 * float(r['delta_k_linear_per_m']) for r in part], linestyle='none', marker='x', ms=4, label=label + ' / first order')
    ax.set(xlabel='Mode index', ylabel='Modal phase change at 20 km (rad)')
    ax.legend(frameon=False, fontsize=8.5)
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_modal_phase', formats)
    data = rows(ROOT / 'results' / 'acoustic_convergence.csv')[:-1]
    fig, ax = plt.subplots(figsize=(6.1, 3.6))
    ax.loglog([int(r['basis_size']) for r in data], [float(r['max_array_mismatch_error_vs_finest']) for r in data], marker='o', lw=1.5)
    ax.set(xlabel='Number of sine basis functions', ylabel='Maximum mismatch difference\nfrom the 192-function calculation')
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_basis_convergence', formats)
    fine = np.load(ROOT / 'results' / 'acoustic_range_reference.npz')
    r = fine['range_m']
    # Plot a short interval at native resolution.
    window = (r >= 19500) & (r <= 20000)
    stride = int(round(50.0 / float(r[1] - r[0])))
    coarse = ((np.arange(len(r)) % stride) == 0) & window
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for name, label, style, marker in [('uniform', 'Uniform', '-', 'o'),
                                      ('localized', 'Localized', '--', 's')]:
        ax.plot(r[window] / 1000, fine[name][window], linestyle=style, lw=1.2,
                label=label + ' / 0.25 m reference')
        ax.plot(r[coarse] / 1000, fine[name][coarse], linestyle=':', marker=marker,
                fillstyle='none', ms=4, lw=0.8, label=label + ' / 50 m samples')
    ax.set(xlabel='Source-receiver range (km)', ylabel='Array mismatch $1-\\gamma^2$',
           xlim=(19.5, 20), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower left')
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_range_detail', formats)
    data = rows(ROOT / 'results' / 'acoustic_range_convergence.csv')[:-1]
    fig, ax = plt.subplots(figsize=(6.1, 3.7))
    for name, label, marker in [('uniform', 'Uniform', 'o'), ('localized', 'Localized', 's')]:
        ax.loglog([float(x['range_step_m']) for x in data],
                  [float(x[name + '_interpolation_max_abs_error']) for x in data],
                  marker=marker, lw=1.3, label=label)
    ax.set(xlabel='Range sampling interval (m)',
           ylabel='Maximum linear-interpolation error\nrelative to the 0.25 m reference')
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_range_convergence', formats)
    data = rows(ROOT / 'results' / 'acoustic_amplitude_sweep.csv')
    fig, ax = plt.subplots(figsize=(6.7, 4.0))
    styles = [('uniform', 'Uniform bias', '-', 'o', 0.4),
              ('localized', 'Localized upper-water error', '--', 's', 0.5)]
    for shape, label, linestyle, marker, marked_rms in styles:
        part = [d for d in data
                if d['perturbation'] == shape and float(d['range_m']) == 10000.]
        amplitudes = [float(d['rms_mps']) for d in part]
        marked_index = int(np.argmin(np.abs(np.asarray(amplitudes) - marked_rms)))
        ax.plot(amplitudes, [float(d['mismatch']) for d in part],
                label=label, linestyle=linestyle, lw=1.8,
                marker=marker, markevery=[marked_index], ms=6)
    contrast = rows(ROOT / 'results' / 'acoustic_contrast.csv')
    for row, offset, name in zip(contrast, [(-108, 26), (-86, -42)], ['A', 'B']):
        x, y = float(row['rms_mps']), float(row['range_10km_mismatch'])
        ax.annotate(f"{name}: {x:.1f} m/s; D = {y:.3f}", (x, y), xytext=offset,
                    textcoords='offset points', fontsize=9,
                    arrowprops={'arrowstyle': '-', 'lw': 0.8})
    ax.set(xlabel='Depth-RMS sound-speed error (m/s)',
           ylabel='Normalized array mismatch $D$', xlim=(0, 0.6), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.tick_params(direction='in', top=True, right=True)
    finish(fig, 'acoustic_rank_reversal', formats)
    print('Figures written to sourcecode/figures (' + ', '.join(formats) + ').')
if __name__ == '__main__':
    main()
