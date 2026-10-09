from copy import deepcopy
import os
import sqlite3
import pytest

import lora_api_server as api
from composition_versions import preparation_digest
from experiment_versions import initialise_experiment_schema
from inspire_export import ARCHITECTURE_LABELS
from profile_versions import get_version
from test_profile_export_api import profile_client, capture


@pytest.fixture
def starting_state(profile_client, monkeypatch):
    client, file, database = profile_client
    native = api.prepare_native_flux_node
    def with_roles(row):
        node, coverage = native(row)
        node['role'] = 'character' if row['stable_id'] == 'first' else 'clothing'
        return node, coverage
    monkeypatch.setattr(api, 'prepare_native_flux_node', with_roles)
    roots = [capture(client, sid) for sid in ('first', 'second')]
    entries = [{'stable_id': root['stable_id'], 'profile_version_id': root['version_id']} for root in roots]
    conn = sqlite3.connect(database)
    initialise_experiment_schema(conn)
    source = api.resolve_composition_preparation(conn, entries, 'flux1-dev-native-v1')
    conn.close()
    norms = [0, 1] + [0]*56
    metrics = {'status': 'complete', 'engine_version': 'effective_native_lora_v1',
        'target_contract_id': 'flux1-dev-native-v1', 'measurement_basis': 'effective_native_parameter_update',
        'slot_labels': list(ARCHITECTURE_LABELS), 'outer_model_strength_applied': False, 'block_weights_applied': False,
        'sources': [{'source_index': index, 'block_squared_norms': norms, 'block_norms': norms, 'total_squared_norm': 1} for index in range(2)],
        'pairs': [{'left_index': 0, 'right_index': 1, 'block_inner_products': norms}]}
    def revalidate(job_id):
        assert job_id == 'current-job'
        with sqlite3.connect(database) as conn:
            current = api.resolve_composition_preparation(conn, entries, 'flux1-dev-native-v1')
        return {'job': {'metrics': deepcopy(metrics)}, 'preparation': current}
    monkeypatch.setattr(api.analysis_jobs, 'revalidate', revalidate)
    body = {'job_id': 'current-job', 'priorities': {}, 'expected_preparation_digest': source['preparation_digest']}
    return client, file, database, roots, source, body


def test_main_proposal_has_fresh_sparse_loader_csv_and_retains_defaults_without_personal_saves(starting_state):
    client, _, database, roots, source, body = starting_state
    response = client.post('/api/experiments/prepare', json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['preparation_digest'] == preparation_digest(result)
    assert result['source_preparation'] == source
    assert [node['loader_export']['slot_values'][1] for node in result['node_payloads']] == [.9, .65]
    assert all(node['profile_version_id'] is None for node in result['node_payloads'])
    assert all(node['loader_export']['loader_slot_count'] == 12 for node in result['node_payloads'])
    for node, proposal in zip(result['node_payloads'], result['starting_proposal']['policy_preview']['entries']):
        assert node['loader_export']['architecture_slot_values'] == proposal['values']
        assert list(map(float, node['loader_export']['numeric_csv'].split(','))) == node['loader_export']['slot_values']
    again = client.post('/api/experiments/prepare', json=body).json()
    assert again['starting_proposal']['computed_baseline']['reused']
    forced = client.post('/api/experiments/prepare', json={**body, 'force_recompute': True}).json()
    assert forced['starting_proposal']['computed_baseline']['baseline_id'] != again['starting_proposal']['computed_baseline']['baseline_id']
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM lora_profile_versions WHERE kind='personal'").fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM lora_composition_versions').fetchone()[0] == 0
        assert [get_version(conn, root['stable_id'], root['version_id'])['values'] for root in roots] == [root['values'] for root in roots]


def test_explicit_proposal_save_replays_and_reexports_identical_values_in_recipe(starting_state):
    client, _, database, roots, _, body = starting_state
    prepared = client.post('/api/experiments/prepare', json=body).json()
    proposal = prepared['starting_proposal']
    save = {**body, 'expected_proposal_digest': proposal['proposal_digest'], 'policy_kind': 'managed_start', 'name': 'Starting pair', 'idempotency_key': 'explicit-save'}
    response = client.post('/api/experiments/save', json=save)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == 'saved'
    assert [node['loader_export']['numeric_csv'] for node in result['composition']['historical_snapshot']['node_payloads']] == [node['loader_export']['numeric_csv'] for node in prepared['node_payloads']]
    replay = client.post('/api/experiments/save', json=save).json()
    assert replay['composition']['version_id'] == result['composition']['version_id']
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora_composition_versions').fetchone()[0] == 1
        assert all(get_version(conn, root['stable_id'], root['version_id'])['kind'] == 'default' for root in roots)


def test_changed_source_or_browser_vectors_cannot_produce_a_copyable_proposal(starting_state):
    client, file, _, _, _, body = starting_state
    assert client.post('/api/experiments/prepare', json={**body, 'values': [0]*58}).status_code == 422
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
    assert client.post('/api/experiments/prepare', json=body).status_code == 409


def test_header_change_between_calculation_and_export_is_rejected(starting_state, monkeypatch):
    client, file, _, _, _, body = starting_state
    resolve = api.resolve_composition_preparation
    def move_after_preparation(conn, entries, target):
        prepared = resolve(conn, entries, target)
        stat = file.stat()
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
        return prepared
    monkeypatch.setattr(api, 'resolve_composition_preparation', move_after_preparation)
    assert client.post('/api/experiments/prepare', json=body).status_code == 409
