from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

import experiment_versions as experiments
from composition_versions import (
    PreparationChangedError, initialise_composition_schema, preparation_digest,
)
from profile_versions import (
    ProfileNotFoundError, ProfileValidationError, capture_default, create_revision, get_version, initialise_schema,
)

TARGET = 'flux1-dev-native-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare(conn, refs, target):
    assert conn.in_transaction  # New profile rows must be visible to this same connection.
    nodes = []
    for ref in refs:
        version = get_version(conn, ref['stable_id'], ref['profile_version_id'])
        labels = [slot['label'] for slot in version['binding']['slots']]
        nodes.append({**ref, **version['settings'], 'loader_export': {
            'status': 'ready', 'target_contract_id': target,
            'architecture_slot_values': version['values'], 'architecture_slot_labels': labels,
            'slot_values': version['values'], 'slot_labels': labels, 'loader_slot_count': len(labels),
            'numeric_csv': ','.join(str(value) for value in version['values']),
        }})
    ids = [ref['stable_id'] for ref in refs]
    result = {'target_contract_id': target, 'compatible': True, 'requested_loras': ids,
              'included_loras': ids, 'excluded_loras': [], 'node_payloads': nodes}
    result['preparation_digest'] = preparation_digest(result)
    return result


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'experiments.db'
    conn = sqlite3.connect(path)
    initialise_schema(conn)
    initialise_composition_schema(conn)
    experiments.initialise_experiment_schema(conn)
    roots = []
    for sid in ('person', 'coat'):
        roots.append(capture_default(conn, stable_id=sid, binding={
            'architecture': 'fixture', 'source_identity': {'basis': 'header_stat', 'header_sha256': 'a'*64, 'size_bytes': 100, 'mtime_ns': 10},
            'engine_version': 'baseline', 'policy_version': 'structural_baseline_unvalidated',
            'target_contract': {'id': TARGET, 'sha256': 'b'*64},
            'loader_adapter': {'id': 'fixture-adapter', 'source_sha256': {'loader.py': 'c'*64}},
            'slots': [{'group': 'base', 'label': 'BASE'}, {'group': 'double', 'label': 'DOUBLE 0'}],
        }, values=[1, -0.8765432109876543], settings={'role': sid, 'strength_model': 0.875,
                                                     'strength_clip': None, 'affect_clip': False}))
    metrics = {'status': 'complete', 'engine_version': 'effective_native_lora_v1', 'target_contract_id': TARGET,
               'sources': [{'path': 'private/local/source.safetensors', 'identity': roots[0]['binding']['source_identity']}],
               'image_quality_verified': False}
    plan = {'job_id': 'job-1', 'engine_version': metrics['engine_version'], 'policy_version': 'gentle_positive_alignment_v1',
            'target_contract_id': TARGET, 'metrics_receipt': metrics, 'metrics_receipt_sha256': digest(metrics),
            'input_preparation_digest': 'd'*64,
            'entries': [{'stable_id': root['stable_id'], 'parent_version_id': root['version_id'],
                         'values': [1, -0.75], 'settings': root['settings'], 'ab': {}, 'priority': index}
                        for index, root in enumerate(roots)]}
    update_preview(plan, roots)
    yield conn, path, roots, plan
    conn.close()


def update_preview(plan, roots):
    plan['policy_preview'] = {'policy_version': plan['policy_version'], 'status': 'experimental_preview', 'calibrated': False,
                              'constants': {'max_reduction': 0.2, 'pressure_threshold': 0.5}, 'changes': [], 'blocks': [],
                              'entries': [{'stable_id': entry['stable_id'], 'profile_version_id': entry['parent_version_id'],
                                           'priority': entry['priority'], 'values': entry['values'], 'before_values': root['values']}
                                          for entry, root in zip(plan['entries'], roots)]}
    plan['proposal_digest'] = experiments.experiment_plan_digest(plan)


def save(state, **overrides):
    conn, _, _, plan = state
    args = {'job_id': plan['job_id'], 'expected_proposal_digest': plan['proposal_digest'],
            'name': 'Softer combination', 'idempotency_key': 'request-1',
            'experiment_resolver': lambda conn, job: deepcopy(plan), 'preparation_resolver': prepare}
    args.update(overrides)
    return experiments.save_experiment(conn, **args)


def counts(conn):
    return tuple(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in
                 ('lora_profile_versions', 'lora_composition_versions', 'lora_experiment_versions'))


def test_complete_atomic_save_exact_provenance_and_defaults_unchanged(state):
    conn, path, roots, plan = state
    result = save(state)
    assert counts(conn) == (4, 1, 1)
    assert result['status'] == 'saved'
    assert result['requires_revalidation'] is True
    assert 'node_payloads' not in result
    assert result['receipt']['plan'] == plan
    assert result['receipt']['plan']['metrics_receipt_sha256'] == digest(plan['metrics_receipt'])
    assert [get_version(conn, root['stable_id'], root['version_id']) for root in roots] == roots
    for root, ref in zip(roots, result['composition']['entries']):
        version = get_version(conn, ref['stable_id'], ref['profile_version_id'])
        assert version['values'] == [1, -0.75]
        assert version['parent_id'] == root['version_id']
        assert version['default_id'] == root['version_id']
        assert version['binding'] == root['binding']
        assert version['provenance']['method'] == 'gentle_balance_experiment'
        assert version['provenance']['experiment_id'] == result['experiment_id']
        assert version['provenance']['policy_version'] == plan['policy_version']
    reopened = sqlite3.connect(path)
    try:
        assert experiments.get_experiment(reopened, result['experiment_id']) == result
        for operation in ('UPDATE lora_experiment_versions SET job_id="rewrite"', 'DELETE FROM lora_experiment_versions'):
            with pytest.raises(sqlite3.IntegrityError, match='immutable'):
                reopened.execute(operation)
            reopened.rollback()
    finally:
        reopened.close()


def test_unchanged_entries_stay_pinned_and_no_changes_adds_no_history(state):
    conn, _, roots, plan = state
    plan['entries'][0]['values'] = roots[0]['values']
    update_preview(plan, roots)
    result = save(state)
    assert result['composition']['entries'][0]['profile_version_id'] == roots[0]['version_id']
    assert counts(conn) == (3, 1, 1)
    plan['entries'][1]['values'] = roots[1]['values']
    update_preview(plan, roots)
    before = counts(conn)
    result = save(state, idempotency_key='no-change', preparation_resolver=lambda *args: pytest.fail('No-change must not prepare new history'))
    assert result['status'] == 'no_changes'
    assert counts(conn) == before


@pytest.mark.parametrize('failure_at', ['second_child', 'prepare', 'composition', 'receipt'])
def test_any_failure_rolls_back_every_child_recipe_and_receipt(state, monkeypatch, failure_at):
    conn, _, _, _ = state
    before = counts(conn)
    if failure_at == 'second_child':
        original = experiments._insert
        calls = []
        def insert(*args, **kwargs):
            calls.append(kwargs['stable_id'])
            if len(calls) == 2:
                raise RuntimeError('Injected second child failure')
            return original(*args, **kwargs)
        monkeypatch.setattr(experiments, '_insert', insert)
    elif failure_at == 'composition':
        conn.execute("CREATE TRIGGER fail_recipe BEFORE INSERT ON lora_composition_versions BEGIN SELECT RAISE(ABORT, 'Injected recipe failure'); END")
    elif failure_at == 'receipt':
        conn.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON lora_experiment_versions BEGIN SELECT RAISE(ABORT, 'Injected receipt failure'); END")
    def fail_prepare(*args):
        assert counts(conn)[0] == 4
        raise RuntimeError('Source changed after preview')
    with pytest.raises((RuntimeError, sqlite3.IntegrityError)):
        save(state, **({'preparation_resolver': fail_prepare} if failure_at == 'prepare' else {}))
    assert counts(conn) == before
    assert not conn.in_transaction


def test_exact_retry_returns_original_without_reresolving_stale_job(state):
    result = save(state)
    assert save(state, experiment_resolver=lambda *args: pytest.fail('A replay is historical, not new Copy authority')) == result
    assert counts(state[0]) == (4, 1, 1)
    for overrides in ({'name': 'Different'}, {'job_id': 'another'}, {'expected_proposal_digest': 'a'*64}, {'parent_version_id': result['composition']['version_id']}):
        with pytest.raises(PreparationChangedError, match='save identifier'):
            save(state, **overrides)


@pytest.mark.parametrize('mutation', ['digest', 'metrics_hash', 'job', 'parent', 'duplicate', 'values', 'settings', 'target'])
def test_bad_or_stale_server_plan_adds_nothing(state, mutation):
    conn, _, roots, plan = state
    if mutation == 'digest':
        plan['entries'][0]['values'][1] = -0.72
    elif mutation == 'metrics_hash':
        plan['metrics_receipt']['sources'][0]['identity']['mtime_ns'] = 20
    elif mutation == 'job':
        plan['job_id'] = 'wrong-job'
    elif mutation == 'parent':
        plan['entries'][0]['parent_version_id'] = roots[1]['version_id']
    elif mutation == 'duplicate':
        plan['entries'][1]['stable_id'] = plan['entries'][0]['stable_id']
    elif mutation == 'values':
        plan['entries'][0]['values'].append(1)
    elif mutation == 'settings':
        plan['entries'][0]['settings'] = dict(plan['entries'][0]['settings'], strength_model=9)
    else:
        plan['target_contract_id'] = 'other-target'
    if mutation != 'digest':
        if mutation in ('parent', 'duplicate', 'values'):
            update_preview(plan, roots)
        plan['proposal_digest'] = experiments.experiment_plan_digest(plan)
    with pytest.raises((ProfileValidationError, ProfileNotFoundError)):
        save(state, job_id='job-1')
    assert counts(conn) == (2, 0, 0)


@pytest.mark.parametrize('field', ['policy_version', 'values', 'before_values', 'priority', 'input_digest'])
def test_full_policy_preview_alignment_is_required_even_with_matching_plan_hash(state, field):
    conn, _, _, plan = state
    if field == 'policy_version':
        plan['policy_preview']['policy_version'] = 'different'
    elif field == 'input_digest':
        plan['input_preparation_digest'] = 'not-a-digest'
    else:
        plan['policy_preview']['entries'][0][field] = 2 if field == 'priority' else [1, 0.4]
    plan['proposal_digest'] = experiments.experiment_plan_digest(plan)
    with pytest.raises(ProfileValidationError):
        save(state)
    assert counts(conn) == (2, 0, 0)


def test_parent_recipe_links_preserved_and_prior_ab_recorded_without_mutation(state):
    conn, _, roots, plan = state
    root = roots[0]
    manual = create_revision(conn, stable_id=root['stable_id'], default_id=root['version_id'], parent_id=root['version_id'],
                             name='Manual trial', binding=root['binding'], values=root['values'], settings=root['settings'],
                             ab={'A': {'slot_labels': ['DOUBLE 0'], 'min': -1, 'max': -0.5,
                                       'value': root['values'][1], 'basis': 'Personal trial'}})
    plan['entries'][0]['parent_version_id'] = manual['version_id']
    update_preview(plan, [manual, roots[1]])
    first = save(state)
    child_id = first['composition']['entries'][0]['profile_version_id']
    child = get_version(conn, 'person', child_id)
    assert child['ab'] == {}
    assert child['provenance']['parent_ab'] == manual['ab']
    assert child['provenance']['ab_changed'] is True
    assert get_version(conn, 'person', manual['version_id']) == manual
    second = save(state, idempotency_key='another-experiment', parent_version_id=first['composition']['version_id'])
    assert second['composition']['parent_version_id'] == first['composition']['version_id']
    assert second['composition']['composition_id'] == first['composition']['composition_id']


def test_dropping_only_ab_cannot_create_a_spurious_experiment(state):
    conn, _, roots, plan = state
    root = roots[0]
    manual = create_revision(conn, stable_id=root['stable_id'], default_id=root['version_id'], parent_id=root['version_id'],
                             name='Manual trial', binding=root['binding'], values=root['values'], settings=root['settings'],
                             ab={'A': {'slot_labels': ['DOUBLE 0'], 'min': -1, 'max': -0.5,
                                       'value': root['values'][1], 'basis': 'Personal trial'}})
    plan['entries'][0]['parent_version_id'] = manual['version_id']
    for entry, parent in zip(plan['entries'], [manual, roots[1]]):
        entry['values'] = parent['values']
    update_preview(plan, [manual, roots[1]])
    with pytest.raises(ProfileValidationError, match='A/B'):
        save(state)
    assert counts(conn) == (3, 0, 0)


def test_two_simultaneous_identical_saves_produce_only_one_history_batch(state):
    _, path, _, plan = state
    def worker(_):
        conn = sqlite3.connect(path, timeout=10)
        try:
            return save((conn, path, [], plan))
        finally:
            conn.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, range(2)))
    assert results[0] == results[1]
    assert counts(state[0]) == (4, 1, 1)
