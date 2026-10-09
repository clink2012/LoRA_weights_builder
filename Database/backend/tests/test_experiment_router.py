from copy import deepcopy
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from analysis_job_service import JobError
from composition_versions import initialise_composition_schema, preparation_digest
from experiment_router import create_experiment_router
from experiment_versions import initialise_experiment_schema
from inspire_export import ARCHITECTURE_LABELS
from profile_versions import capture_default, get_version, initialise_schema

TARGET = 'flux1-dev-native-v1'


def prepare(conn, entries, target):
    nodes = []
    for entry in entries:
        version = get_version(conn, entry['stable_id'], entry['profile_version_id'])
        values = version['values']
        nodes.append({**entry, **version['settings'], 'profile_default_binding': version['binding'],
                      'loader_export': {'status': 'ready', 'target_contract_id': target,
                                        'architecture_slot_values': values, 'architecture_slot_labels': list(ARCHITECTURE_LABELS),
                                        'slot_values': values, 'slot_labels': list(ARCHITECTURE_LABELS), 'loader_slot_count': 58,
                                        'numeric_csv': ','.join(str(v) for v in values)}})
    ids = [entry['stable_id'] for entry in entries]
    result = {'target_contract_id': target, 'compatible': True, 'requested_loras': ids,
              'included_loras': ids, 'excluded_loras': [], 'node_payloads': nodes}
    result['preparation_digest'] = preparation_digest(result)
    return result


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'experiment-api.db'
    conn = sqlite3.connect(path)
    initialise_schema(conn)
    initialise_composition_schema(conn)
    initialise_experiment_schema(conn)
    roots = [capture_default(conn, stable_id=sid, binding={
        'architecture': 'flux.1', 'source_identity': {'basis': 'file_sha256', 'sha256': 'a'*64},
        'engine_version': 'baseline', 'policy_version': 'structural_baseline_unvalidated',
        'target_contract': {'id': TARGET, 'sha256': 'b'*64},
        'loader_adapter': {'id': 'test', 'source_sha256': {'test.py': 'c'*64}},
        'slots': [{'group': label.split(' ')[0].lower(), 'label': label} for label in ARCHITECTURE_LABELS],
    }, values=[1]*58, settings={'role': sid, 'strength_model': 1, 'strength_clip': 0, 'affect_clip': False})
        for sid in ('person', 'clothing')]
    refs = [{'stable_id': root['stable_id'], 'profile_version_id': root['version_id']} for root in roots]
    prepared = prepare(conn, refs, TARGET)
    metrics = {'status': 'complete', 'engine_version': 'effective_native_lora_v1', 'target_contract_id': TARGET,
               'slot_labels': list(ARCHITECTURE_LABELS), 'outer_model_strength_applied': False, 'block_weights_applied': False,
               'sources': [{'source_index': i, 'block_squared_norms': [1]*58} for i in range(2)],
               'pairs': [{'left_index': 0, 'right_index': 1, 'block_inner_products': [1]*58}]}
    class Service:
        stale = False
        def revalidate(self, job_id):
            if self.stale or job_id != 'known-job':
                raise JobError('analysis_not_current', 'Current measurements required', 409)
            return {'job': {'metrics': deepcopy(metrics)}, 'preparation': deepcopy(prepared)}
    service = Service()
    app = FastAPI()
    app.include_router(create_experiment_router(lambda: sqlite3.connect(path), service, prepare))
    payload = {'job_id': 'known-job', 'priorities': {'person': 2, 'clothing': 0},
               'expected_preparation_digest': prepared['preparation_digest']}
    yield TestClient(app), conn, roots, service, payload
    conn.close()


def save_body(payload, preview):
    return {**payload, 'expected_proposal_digest': preview['proposal_digest'],
            'name': 'Gentle trial', 'idempotency_key': 'one-save'}


def counts(conn):
    return [conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in
            ('lora_profile_versions', 'lora_composition_versions', 'lora_experiment_versions')]


def test_preview_atomic_save_replay_and_full_numeric_recipe(state):
    client, conn, roots, _, payload = state
    preview = client.post('/api/experiments/preview', json=payload)
    assert preview.status_code == 200
    data = preview.json()
    assert data['can_save'] is True and len(data['policy_preview']['changes']) == 57
    assert counts(conn) == [2, 0, 0]
    saved = client.post('/api/experiments/save', json=save_body(payload, data))
    assert saved.status_code == 200, saved.text
    result = saved.json()
    assert result['status'] == 'saved' and result['requires_revalidation'] is True
    assert counts(conn) == [3, 1, 1]
    assert result['composition']['entries'][0]['profile_version_id'] == roots[0]['version_id']
    assert result['receipt']['plan']['policy_preview'] == data['policy_preview']
    assert result['composition']['historical_snapshot']['node_payloads'][1]['loader_export']['slot_values'] == [1] + [0.8]*57
    assert client.post('/api/experiments/save', json=save_body(payload, data)).json() == result
    assert client.get('/api/experiments/' + result['experiment_id']).json() == result
    assert counts(conn) == [3, 1, 1]


def test_priority_change_after_preview_or_changed_file_rejects_save(state):
    client, conn, _, service, payload = state
    preview = client.post('/api/experiments/preview', json=payload).json()
    body = save_body(payload, preview)
    body['priorities'] = {'person': 0, 'clothing': 2}
    assert client.post('/api/experiments/save', json=body).status_code == 409
    service.stale = True
    assert client.post('/api/experiments/save', json=save_body(payload, preview)).status_code == 409
    assert counts(conn) == [2, 0, 0]


def test_equal_priorities_return_no_changes_without_creating_history(state):
    client, conn, _, _, payload = state
    payload['priorities'] = {'person': 1, 'clothing': 1}
    preview = client.post('/api/experiments/preview', json=payload).json()
    assert preview['can_save'] is False and preview['policy_preview']['changes'] == []
    result = client.post('/api/experiments/save', json=save_body(payload, preview))
    assert result.status_code == 200 and result.json()['status'] == 'no_changes'
    assert counts(conn) == [2, 0, 0]


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(values=[0]*58), lambda p: p.update(metrics={'status': 'complete'}),
    lambda p: p['priorities'].update(person=True), lambda p: p['priorities'].pop('clothing'),
])
def test_browser_cannot_supply_values_metrics_or_invalid_priorities(state, mutation):
    client, conn, _, _, payload = state
    mutation(payload)
    assert client.post('/api/experiments/preview', json=payload).status_code == 422
    assert counts(conn) == [2, 0, 0]


def test_stale_displayed_preparation_rejects_preview(state):
    client, _, _, _, payload = state
    payload['expected_preparation_digest'] = '0'*64
    assert client.post('/api/experiments/preview', json=payload).status_code == 409


def test_role_start_uses_pinned_saved_roles_and_keeps_protected_pair_unresolved(state):
    client, conn, _, _, payload = state
    payload.update(policy_kind='role_start', priorities={})
    response = client.post('/api/experiments/preview', json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['policy_preview']['policy_version'] == 'role_measured_start_v1'
    assert [rule['priority'] for rule in result['policy_preview']['role_rules']] == [2, 2]
    assert result['can_save'] is False
    assert counts(conn) == [2, 0, 0]


def test_role_start_override_saves_graph_receipt_atomically_and_detects_policy_drift(state):
    client, conn, _, _, payload = state
    payload.update(policy_kind='role_start', priorities={'clothing': 0})
    response = client.post('/api/experiments/preview', json=payload)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview['policy_preview']['contribution_graphs'][1]['proposed_norms'][1] == .8
    body = save_body(payload, preview)
    saved = client.post('/api/experiments/save', json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()['receipt']['plan']['policy_preview'] == preview['policy_preview']
    assert counts(conn) == [3, 1, 1]
    body.update(policy_kind='gentle', priorities={'person': 2, 'clothing': 0}, idempotency_key='different')
    assert client.post('/api/experiments/save', json=body).status_code == 409


def test_unknown_policy_is_rejected_without_writes(state):
    client, conn, _, _, payload = state
    payload['policy_kind'] = 'invented'
    assert client.post('/api/experiments/preview', json=payload).status_code == 422
    assert counts(conn) == [2, 0, 0]
