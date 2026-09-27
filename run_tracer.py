"""Run the periodic conservative tracer experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys_path = str(ROOT / 'scripts')
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from tracer_example import run


def main() -> None:
    result = run()
    orders = [row['order_from_previous'] for row in result['convergence']
              if row['order_from_previous'] is not None]
    print('Periodic tracer')
    print(f"Mass drift        : {result['maximum_relative_mass_drift']:.3e}")
    print(f"Variance residual : {result['maximum_relative_variance_budget_residual']:.3e}")
    print(f"Order rate        : {orders[-1]:.3f}")
    print(f"Dense exp. error  : {result['dense_exponential_max_abs_error']:.3e}")


if __name__ == '__main__':
    main()
