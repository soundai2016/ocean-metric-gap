"""Build measurement plans from declared evaluation requests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TIER_CHECKS = {
    1: ('offline', 5, ('M02', 'M03', 'M07'),
        'Evaluate held-out target error, amplitude, and resolved scales.'),
    2: ('initialized', 6, ('M01', 'M02', 'M07', 'M08'),
        'Compare initialized predictions by lead, depth, and region.'),
    3: ('coupled', 7, ('M04', 'M05', 'M07'),
        'Run coupled controls; measure work, inventories, drift, and forced response.'),
    4: ('downstream', 8, ('M02', 'M07', 'M10'),
        'Fix the physical operator and geometry; propagate each environmental estimate.'),
}


def make_plan(request: dict[str, Any]) -> dict[str, Any]:
    """Validate a request and return its measurement records."""
    if not isinstance(request, dict):
        raise ValueError('request must be an object')
    fields = {'record_id', 'claimed_tests', 'comparison_target',
              'prognostic_correction', 'probabilistic_claim', 'transfer_condition'}
    if set(request) != fields:
        raise ValueError('request fields must be exactly: ' + ', '.join(sorted(fields)))
    if not isinstance(request['record_id'], str) or not request['record_id'].strip():
        raise ValueError('record_id must be a nonempty string')
    tests = request['claimed_tests']
    if not isinstance(tests, list) or not tests:
        raise ValueError('claimed_tests must be a nonempty list')
    if any(type(t) is not int or t not in TIER_CHECKS for t in tests):
        raise ValueError('claimed tests must be integers 1 through 4')
    if len(set(tests)) != len(tests):
        raise ValueError('claimed tests must be unique')
    comparison = request['comparison_target']
    if not isinstance(comparison, str) or comparison not in ('complete_system', 'matched_information'):
        raise ValueError('comparison_target must be complete_system or matched_information')
    for key in ('prognostic_correction', 'probabilistic_claim'):
        if type(request[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    transfer = request['transfer_condition']
    if transfer is not None and (not isinstance(transfer, str) or not transfer.strip()):
        raise ValueError('transfer_condition must be null or a named, nonempty condition')
    checks: list[dict[str, Any]] = []

    def add(identifier: str, line: int, items: tuple[str, ...], action: str) -> None:
        checks.append({'id': identifier, 'algorithm_line': line, 'reporting_items': list(items),
                       'action': action, 'status': 'pending', 'criterion': None,
                       'result': None, 'source_locator': None})

    add('specification', 1, ('M01', 'M02', 'M03', 'M04'),
        'Record issue-time inputs, state support, learned target, and receiving equations.')
    comparator_action = ('Hold admissible inputs fixed and identify remaining numerical differences.'
        if comparison == 'matched_information' else
        'Record each complete system and its input pipeline; compare their practical performance.')
    add('comparator', 2, ('M06',), comparator_action)
    add('verification_design', 3, ('M07', 'M08'),
        'Fix support, diagnostic, tolerance, and independent comparison unit before testing.')
    for tier in sorted(tests):
        identifier, line, items, action = TIER_CHECKS[tier]
        add(identifier, line, items, action)
    if request['prognostic_correction']:
        add('correction_budget', 9, ('M04', 'M05'),
            'Record before/after inventories, physical fluxes, increments, and residuals.')
    if request['probabilistic_claim']:
        add('probabilistic_verification', 10, ('M07', 'M08'),
            'Compute proper scores and conditional reliability using declared replication units.')
    if transfer is not None:
        add('transfer_frozen', 11, ('M06', 'M09'),
            f'Test frozen parameters after changing: {transfer.strip()}.')
        add('transfer_retuned', 11, ('M06', 'M09'),
            f'Report any retuned result separately for: {transfer.strip()}.')
    add('assessment', 12, ('M07', 'M08', 'M09'),
        'Compare measurements with the declared criterion; leave missing tests unresolved.')
    add('release', 13, ('M10',),
        'Report measurements, missing tests and the versioned M01-M10 record.')
    return {'record_id': request['record_id'], 'claimed_tests': sorted(tests),
            'comparison_target': comparison, 'checks': checks,
            'reporting_items': [f'M{i:02}' for i in range(1, 11)],
            'scope': 'measurement_plan_only', 'experiment_results_evaluated': False}


def run() -> dict[str, Any]:
    requests = json.loads((ROOT / 'data/evaluation_requests.json').read_text(encoding='utf-8'))
    if not isinstance(requests, list) or not requests:
        raise ValueError('evaluation_requests.json must contain a nonempty array')
    plans = [make_plan(request) for request in requests]
    if len({plan['record_id'] for plan in plans}) != len(plans):
        raise ValueError('example record identifiers must be unique')
    result = {'algorithm': 1, 'scope': 'measurement_plan_only', 'plans': plans}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/evaluation_plans.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
