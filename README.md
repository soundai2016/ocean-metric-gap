# Ocean error metrics: numerical experiments

Reference implementation for the tracer-transport and normal-mode acoustic experiments used in the accompanying study. The repository contains deterministic numerical solvers, error-metric calculations, regression tests, and plotting scripts. No model training, external forecast download, or GPU is required.

## Requirements

- Python 3.11+
- NumPy
- SciPy
- Matplotlib

The archived results were generated on CPU. Install the pinned environment with

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce the calculations

Run the complete numerical workflow:

```bash
python run.py
```

Run the two physical examples directly:

```bash
python run_tracer.py
python run_acoustics.py
```

Useful variants:

```bash
python run.py --no-figures
python run.py --tests-only
python run.py --figures-only
python run.py --figure-formats pdf png
```

`run.py` prints the main numerical diagnostics directly, including tracer mass drift, variance-budget residual, grid-convergence order, acoustic discretization errors, and the quadratic-metric thresholds. A non-zero exit code indicates a failed calculation or unit test.

## Main outputs

| Experiment | Configuration | Main outputs |
|---|---|---|
| Periodic tracer transport | `data/tracer_spec.json` | `results/tracer_summary.csv`, `tracer_convergence.csv`, `tracer_budgets.csv` |
| Normal-mode acoustics | `data/acoustic_spec.json` | `results/acoustic_summary.csv`, `acoustic_convergence.csv`, `acoustic_range_convergence.csv` |
| Acoustic sensitivity | `data/acoustic_sensitivity_spec.json` | `results/acoustic_contrast.csv`, `acoustic_amplitude_sweep.csv`, `acoustic_operator_sensitivity.csv` |
| Local acoustic metric | `data/acoustic_metric_spec.json` | `results/acoustic_metric_spectrum.csv`, `acoustic_metric_refinement.csv`, `acoustic_metric_witnesses.csv` |
| Physical budget checks | prescribed manufactured cases | `results/physical_diagnostics.json` |
| Quadratic ordering checks | prescribed matrices | `results/quadratic_order_checks.json` |

Figures are written to `figures/` in PDF, PNG, and SVG by default.

## Numerical methods

The tracer example uses a centered finite-volume advection-diffusion operator on a periodic grid and integrates the semidiscrete system exactly in time through its Fourier eigenvalues. Grid refinement is checked against the analytic first-harmonic response, and a dense matrix exponential provides an independent small-grid check.

The acoustic example solves a pressure-release, range-independent waveguide with a sine-Galerkin discretization and Gaussian quadrature. A finite-difference eigenproblem is retained as an independent discretization check. Array mismatch is evaluated from the receiver-space orthogonal residual, and derivative and basis refinements are reported for the local quadratic metric.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover conservation identities, analytic mode limits, discretization refinement, coordinate invariance, quadratic-order bounds, and regression values used by the figures.

## Repository layout

- `scripts/` — solvers, metrics, diagnostics, and plotting code
- `data/` — numerical experiment specifications
- `tests/` — unit and regression tests
- `results/` — numerical tables and compact result files
- `figures/` — generated figures

Auxiliary source-table and reporting-support files are not part of the default numerical workflow. They can be regenerated, when needed, with `python run.py --auxiliary`.

## License

Apache License 2.0. See `LICENSE`.
