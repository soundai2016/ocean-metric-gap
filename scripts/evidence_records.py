"""Validate and export source-derived configuration records."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


STATUSES = {'supported', 'tested_not_supported', 'not_reported', 'not_applicable', 'unassessed'}

ROLES = {'learning_method', 'classical_baseline', 'data_product', 'infrastructure', 'review', 'context'}

def validate_source(source: dict) -> None:
    for field in ['source_id', 'citation_key', 'url', 'version', 'locator', 'validation_scope']:
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f'Source missing {field}')
    if source['validation_scope'] != 'metadata_consistency_only':
        raise ValueError('Source grouping validates metadata consistency only')
    if source.get('article_number_is_page'):
        raise ValueError('Article number cannot be used as a pinpoint page')

def validate_case(case: dict, sources: dict) -> None:
    required = ['configuration_id', 'source_id', 'citation_key', 'configuration_label', 'record_role', 'domain_tags', 'task', 'evaluation_setting', 'axes', 'axis_evidence', 'source_locator', 'source_url', 'source_version', 'evidence_independence', 'independence_basis', 'reported_result', 'comparator', 'result_locator', 'review_interpretation', 'next_test', 'extension_type', 'author_overclaim']
    missing = [name for name in required if name not in case]
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    for name in ('reported_result', 'comparator', 'result_locator', 'review_interpretation'):
        if not isinstance(case[name], str) or not case[name].strip():
            raise ValueError(f'{name} must be nonempty text')
    if not re.fullmatch('[A-Z][A-Z0-9-]+', case['configuration_id']):
        raise ValueError('Stable explicit configuration_id required')
    source = sources.get(case['source_id'])
    if not source or case['citation_key'] != source['citation_key']:
        raise ValueError('Source and citation key mismatch')
    if case['source_version'] != source['version'] or case['source_url'] != source['url']:
        raise ValueError('Case source version/URL must match the explicit source record')
    if not case['source_locator'].strip() or not source['locator'].strip():
        raise ValueError('A specific section, equation, table or PDF locator is required')
    if case['record_role'] not in ROLES:
        raise ValueError('Unknown record role')
    if not case['domain_tags'] or len(set(case['domain_tags'])) != len(case['domain_tags']):
        raise ValueError('Nonempty unique domain tags required; roles and domains are separate')
    if case['evaluation_setting'] not in {'D', 'S', 'H', 'F'}:
        raise ValueError('Unknown setting')
    if case['evidence_independence'] not in {'UNKNOWN', 'DEV', 'EXT_REPLICATION', 'EXT_AUDIT'}:
        raise ValueError('Unknown evidence-independence value')
    if not case['independence_basis'].strip():
        raise ValueError('Independence needs a reason even when UNKNOWN')
    axes = case['axes']
    for axis in 'OEA':
        if axes[axis] not in [f'{axis}{i}' for i in range(4)] + ['U']:
            raise ValueError(f'Invalid single-valued {axis} code')
    for axis in 'CN':
        values = axes[axis]
        if not isinstance(values, list) or len(set(values)) != len(values):
            raise ValueError('Facet codes must be unique lists')
        if any((value not in [f'{axis}{i}' for i in range(4)] for value in values)):
            raise ValueError(f'Invalid {axis} code')
        if f'{axis}0' in values and len(values) > 1:
            raise ValueError(f'{axis}0 is exclusive')
    for axis in 'OCEAN':
        evidence = case['axis_evidence'][axis]
        if evidence['status'] not in STATUSES or not evidence['detail'].strip():
            raise ValueError('Every axis requires an explicit evidence status and explanation')
        assigned = axes[axis] not in ([], 'U')
        if assigned and evidence['status'] != 'supported':
            raise ValueError('An assigned code requires a source-supported rationale')
    if 'N1' in axes['N'] and (not case.get('probability_assessment', '').strip()):
        raise ValueError('N1 requires the actual probability evaluation, not a model name')
    if 'N2' in axes['N'] and (not case.get('shift_contract', '').strip()):
        raise ValueError('N2 requires a declared reference/contrast mechanism')
    if case.get('calibration_quality') == 'calibrated' and (not case.get('calibration_quality_basis')):
        raise ValueError('A proper score alone cannot establish calibration')
    if case['extension_type'] not in {'reviewer_proposed_test', 'source_stated_limitation'}:
        raise ValueError('Next test must distinguish our proposal from a source statement')
    if case['author_overclaim'] not in {'not_assessed', 'supported_by_quote_and_comparison'}:
        raise ValueError('Overclaim status cannot be inferred from a missing endpoint')
    if case['author_overclaim'] == 'supported_by_quote_and_comparison' and (not all((case.get(key, '').strip() for key in ['claim_quote', 'claim_quote_locator', 'evidence_comparison']))):
        raise ValueError('A stated overclaim needs the source claim, its locator and an evidence comparison')
    if source.get('article_number_is_page'):
        raise ValueError('Article numbers are not pinpoint page references')

def flatten(case: dict) -> dict:
    row = {k: v for k, v in case.items() if k not in {'axes', 'axis_evidence'}}
    row['domain_tags'] = ';'.join(row['domain_tags'])
    for axis in 'OCEAN':
        value = case['axes'][axis]
        row[axis + '_levels'] = ';'.join(value) if isinstance(value, list) else value
        row[axis + '_evidence_status'] = case['axis_evidence'][axis]['status']
        row[axis + '_evidence_summary'] = case['axis_evidence'][axis]['detail']
    return row

def source_index(cases: list[dict]) -> dict[str, dict]:
    """Group source metadata and reject contradictory identities."""
    sources = {}
    for case in cases:
        source_id = case['source_id']
        identity = {
            'citation_key': case['citation_key'],
            'url': case['source_url'],
            'version': case['source_version'],
        }
        if source_id in sources:
            source = sources[source_id]
            if any(source[field] != value for field, value in identity.items()):
                raise ValueError(f'Conflicting source metadata for {source_id}')
            if case['source_locator'] not in source['locators']:
                source['locators'].append(case['source_locator'])
        else:
            sources[source_id] = {
                'source_id': source_id,
                **identity,
                'locators': [case['source_locator']],
                'validation_scope': 'metadata_consistency_only',
            }
    for source in sources.values():
        source['locator'] = '; '.join(source.pop('locators'))
        validate_source(source)
    return sources

def load_cases(path: Path | None = None) -> tuple[list[dict], dict[str, dict]]:
    path = path or ROOT / 'data' / 'case_configurations.json'
    cases = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Case data must be a nonempty list')
    identifiers = [case['configuration_id'] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate configuration IDs')
    sources = source_index(cases)
    for case in cases:
        validate_case(case, sources)
    return cases, sources

def export_evidence(cases: list[dict], sources: dict) -> None:
    """CSV exports and arithmetic from published values; no model reruns implied."""
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    fields = ['configuration_id', 'citation_key', 'configuration_label', 'comparator',
              'reported_result', 'result_locator', 'review_interpretation', 'next_test',
              'source_url', 'source_version']
    with (out / 'review_evidence.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cases)
    (out / 'source_index.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')
    with (ROOT / 'data' / 'closure_benchmark.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    calculated = []
    for row in rows:
        baseline = float(row['no_parameterization_rmse_m'])
        corrected = float(row['ke_tuned_rmse_m'])
        calculated.append(dict(configuration='grid_1/' + row['grid_denominator'],
            source_id='zhang', source_locator='Table 2', baseline_rmse_m=baseline,
            learned_rmse_m=corrected, percent_reduction=100 * (baseline - corrected) / baseline))
    with (ROOT / 'data' / 'closure_geometry.csv').open(encoding='utf-8') as handle:
        row = next(csv.DictReader(handle))
    baseline, corrected = float(row['baseline_rmse_m']), float(row['learned_rmse_m'])
    calculated.append(dict(configuration='introduced_obstacle', source_id='zhang',
        source_locator=row['source_locator'], baseline_rmse_m=baseline,
        learned_rmse_m=corrected, percent_reduction=100 * (baseline - corrected) / baseline))
    with (out / 'closure_relative_changes.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calculated[0]))
        writer.writeheader()
        writer.writerows(calculated)

def run() -> dict:
    cases, sources = load_cases()
    export_evidence(cases, sources)
    return {'configurations': len(cases), 'sources': len(sources),
            'validation_scope': 'metadata_consistency_only',
            'external_models_rerun': False}

if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
