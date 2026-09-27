"""Validate M01--M10 fields and file paths in an experiment record."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_record(record: dict, definitions: dict, project_root: Path) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get('items'), list):
        raise ValueError('record must contain an item list')
    expected = {entry['id'] for entry in definitions['items']}
    items = record['items']
    if any(not isinstance(item, dict) for item in items):
        raise ValueError('each item must be an object')
    if {item.get('id') for item in items} != expected or len(items) != len(expected):
        raise ValueError('every reporting item must occur exactly once')
    for field in ('record_id', 'claim', 'scope'):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f'nonempty {field} required')
    tiers = record.get('tested_tiers')
    if not isinstance(tiers, list) or not tiers:
        raise ValueError('a nonempty, unique tested_tiers list is required')
    if any(isinstance(t, bool) or not isinstance(t, int) or t not in (1, 2, 3, 4) for t in tiers):
        raise ValueError('tested tiers must be integers 1 through 4')
    if len(set(tiers)) != len(tiers):
        raise ValueError('tested tiers must be unique')
    base = project_root.resolve()
    counts = {status: 0 for status in ('reported', 'unreported', 'not_applicable')}
    artifacts = []
    for item in items:
        status = item.get('status')
        if status not in counts:
            raise ValueError('invalid reporting status')
        for field in ('detail', 'artifacts'):
            if field not in item:
                raise ValueError(f'{field} required for every item')
        if not isinstance(item['detail'], str) or not item['detail'].strip():
            raise ValueError('a description, missing-information note, or non-applicability reason is required')
        if not isinstance(item['artifacts'], list) or any(not isinstance(x, str) for x in item['artifacts']):
            raise ValueError('artifact paths must be a list of strings')
        for name in item['artifacts']:
            raw = Path(name)
            if raw.is_absolute():
                raise ValueError('artifact paths must be relative')
            resolved = (base / raw).resolve()
            if not resolved.is_relative_to(base) or not resolved.is_file():
                raise ValueError(f'missing or out-of-package artifact: {name}')
            artifacts.append(name)
        counts[status] += 1
    return {'record_id': record['record_id'], 'valid': True, 'status_counts': counts,
            'checked_artifacts': sorted(set(artifacts)),
            'scope': 'record_completeness_and_local_paths_only',
            'scientific_quality_score': None, 'independent_source_verification': False}


def run() -> dict:
    definitions = json.loads((ROOT / 'data/minimum_reporting_set.json').read_text())
    names = ('acoustic_reporting_record.json', 'mom6_reporting_record.json')
    records = [validate_record(json.loads((ROOT / 'data' / name).read_text()),
                               definitions, ROOT) for name in names]
    result = {'valid': all(record['valid'] for record in records), 'records': records,
              'scope': 'record_completeness_and_local_paths_only',
              'scientific_quality_score': None, 'independent_source_verification': False}
    (ROOT / 'results/reporting_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
