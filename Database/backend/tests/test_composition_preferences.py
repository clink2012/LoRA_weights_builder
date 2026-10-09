from copy import deepcopy
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from composition_preferences import (
    choose_preference, initialise_preference_schema, resolve_preference, restore_originals,
)
from composition_preference_router import create_composition_preference_router
from composition_versions import (
    PreparationChangedError, get_composition, initialise_composition_schema,
    list_compositions, preparation_digest, save_composition,
)
from profile_versions import create_revision, get_version, initialise_schema, ProfileValidationError
from test_composition_versions import add_default, resolve as base_prepare, TARGET


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'preferences.sqlite'
    conn = sqlite3.connect(path)
    initialise_schema(conn)
    initialise_composition_schema(conn)
    initialise_preference_schema(conn)
    roots = [add_default(conn, 'A', 'person'), add_default(conn, 'B', 'clothing')]
    current = {root['stable_id']: {key: deepcopy(root[key]) for key in ('binding', 'values', 'settings', 'ab')} for root in roots}
    entries = [{'stable_id': root['stable_id'], 'profile_version_id': root['version_id']} for root in roots]
    calls = []

    def default(c, sid):
        calls.append(sid)
        return deepcopy(current[sid])

    def prepare(c, refs, target):
        prepared = base_prepare(c, refs, target)
        for node in prepared['node_payloads']:
            node['loader_export']['numeric_csv'] = ','.join(str(value) for value in node['loader_export']['slot_values'])
        prepared['preparation_digest'] = preparation_digest(prepared)
        return prepared

    def save(refs=entries, name='Personal pair', parent=None):
        return save_composition(conn, name=name, entries=refs, target_contract_id=TARGET,
            parent_version_id=parent, expected_preparation_digest=prepare(conn, refs, TARGET)['preparation_digest'], preparation_resolver=prepare)

    def choose(recipe):
        return choose_preference(conn, version_id=recipe['version_id'],
            expected_preparation_digest=prepare(conn, recipe['entries'], TARGET)['preparation_digest'],
            default_resolver=default, preparation_resolver=prepare)

    def recall(ids=('A', 'B'), target=TARGET, c=None):
        return resolve_preference(c or conn, stable_ids=list(ids), target_contract_id=target,
                                  default_resolver=default, preparation_resolver=prepare)

    yield {'conn': conn, 'path': path, 'roots': roots, 'entries': entries, 'current': current,
           'default': default, 'prepare': prepare, 'save': save, 'choose': choose, 'recall': recall, 'calls': calls}
    conn.close()


def test_exact_personal_versions_recalled_after_reopen_without_export_authority(state):
    s = state
    root = s['roots'][0]
    child = create_revision(s['conn'], stable_id='A', default_id=root['version_id'], parent_id=root['version_id'],
        binding=root['binding'], values=[-.2345678901234567, .4567890123456789],
        settings={**root['settings'], 'role': 'clothing', 'strength_model': .7123456789012345}, ab={}, name='Personal identity')
    refs = [{**s['entries'][0], 'profile_version_id': child['version_id']}, s['entries'][1]]
    recipe = s['save'](refs)
    s['choose'](recipe)
    reopened = sqlite3.connect(s['path'])
    try:
        result = s['recall'](c=reopened)
        assert result['status'] == 'preferred'
        assert result['recipe'] == recipe
        assert result['recipe']['entries'] == refs
        assert result['recipe']['requires_revalidation'] is True
        assert 'node_payloads' not in result and 'preparation_digest' not in result
        assert get_version(reopened, 'A', child['version_id'])['values'] == child['values']
    finally:
        reopened.close()


def test_preferences_isolated_by_exact_order_target_and_membership(state):
    s = state
    recipe = s['save']()
    s['choose'](recipe)
    assert s['recall'](['B', 'A'])['status'] == 'none'
    assert s['recall'](['A'])['status'] == 'none'
    assert s['recall'](target='another-target')['status'] == 'none'
    assert s['recall']()['recipe']['version_id'] == recipe['version_id']


def test_choosing_another_named_version_and_original_reset_preserve_all_history(state):
    s = state
    first = s['save'](name='First option')
    second = s['save'](name='Second option', parent=first['version_id'])
    s['choose'](first)
    s['choose'](second)
    assert s['recall']()['recipe']['version_id'] == second['version_id']
    originals = restore_originals(s['conn'], stable_ids=['A', 'B'], target_contract_id=TARGET, default_resolver=s['default'])
    assert originals['entries'] == s['entries']
    assert s['recall']()['status'] == 'none'
    assert len(list_compositions(s['conn'])) == 2
    assert get_composition(s['conn'], first['version_id']) == first
    s['choose'](first)
    assert s['recall']()['recipe']['version_id'] == first['version_id']
    for operation in ('UPDATE lora_composition_preferences SET version_id=NULL', 'DELETE FROM lora_composition_preferences'):
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            s['conn'].execute(operation)
        s['conn'].rollback()


@pytest.mark.parametrize('part', ['source', 'loader', 'target_hash', 'slots', 'engine'])
def test_source_or_loader_drift_retains_preference_for_review_without_recall(state, part):
    s = state
    recipe = s['save']()
    s['choose'](recipe)
    binding = s['current']['A']['binding']
    if part == 'source':
        binding['source_identity']['sha256'] = 'd' * 64
    elif part == 'loader':
        binding['loader_adapter']['source_sha256']['loader.py'] = 'd' * 64
    elif part == 'target_hash':
        binding['target_contract']['sha256'] = 'd' * 64
    elif part == 'slots':
        binding['slots'][1]['label'] = 'DOUBLE 1'
    else:
        binding['engine_version'] = 'next-engine'
    result = s['recall']()
    assert result['status'] == 'needs_review' and result['recipe'] is None
    assert result['retained_version_id'] == recipe['version_id']
    assert get_composition(s['conn'], recipe['version_id']) == recipe


def test_recall_requires_current_preparation_even_when_bindings_still_match(state):
    s = state
    s['choose'](s['save']())
    def blocked(c, e, t):
        return {**s['prepare'](c, e, t), 'compatible': False}
    with pytest.raises(ProfileValidationError, match='freshly prepared'):
        resolve_preference(s['conn'], stable_ids=['A', 'B'], target_contract_id=TARGET,
                           default_resolver=s['default'], preparation_resolver=blocked)


def test_changed_displayed_preparation_does_not_update_preference(state):
    s = state
    recipe = s['save']()
    with pytest.raises(PreparationChangedError, match='changed'):
        choose_preference(s['conn'], version_id=recipe['version_id'], expected_preparation_digest='0' * 64,
                          default_resolver=s['default'], preparation_resolver=s['prepare'])
    assert s['recall']()['status'] == 'none'


def test_no_preference_read_does_not_capture_defaults_or_create_records(state):
    s = state
    assert s['recall']()['status'] == 'none'
    assert s['calls'] == []
    assert s['conn'].execute('SELECT COUNT(*) FROM lora_composition_preferences').fetchone()[0] == 0
    assert s['conn'].execute('SELECT COUNT(*) FROM lora_profile_versions').fetchone()[0] == 2


def test_failed_original_capture_keeps_preference(state):
    s = state
    recipe = s['save']()
    s['choose'](recipe)
    s['current']['B']['values'] = [1, .9]  # Same binding cannot overwrite Default.
    with pytest.raises(ProfileValidationError, match='immutable Default'):
        restore_originals(s['conn'], stable_ids=['A', 'B'], target_contract_id=TARGET, default_resolver=s['default'])
    assert s['recall']()['recipe']['version_id'] == recipe['version_id']


def test_original_reset_after_source_change_captures_a_new_default_without_deleting_old_lineage(state):
    s = state
    recipe = s['save']()
    s['choose'](recipe)
    s['current']['A']['binding']['source_identity']['sha256'] = 'd' * 64
    assert s['recall']()['status'] == 'needs_review'
    result = restore_originals(s['conn'], stable_ids=['A', 'B'], target_contract_id=TARGET, default_resolver=s['default'])
    new_id = result['entries'][0]['profile_version_id']
    assert new_id != s['roots'][0]['version_id']
    assert get_version(s['conn'], 'A', new_id)['binding'] == s['current']['A']['binding']
    assert get_version(s['conn'], 'A', s['roots'][0]['version_id']) == s['roots'][0]
    assert get_composition(s['conn'], recipe['version_id']) == recipe
    assert s['recall']()['status'] == 'none'


def test_router_rejects_client_values_and_allows_checked_preference_and_reset(state):
    s = state
    app = FastAPI()
    app.include_router(create_composition_preference_router(lambda: sqlite3.connect(s['path']), s['default'], s['prepare']))
    recipe = s['save']()
    body = {'version_id': recipe['version_id'], 'expected_preparation_digest': s['prepare'](s['conn'], recipe['entries'], TARGET)['preparation_digest']}
    ids = {'stable_ids': ['A', 'B'], 'target_contract_id': TARGET}
    with TestClient(app) as client:
        base = '/api/composition-preferences'
        assert client.post(base+'/choose', json={**body, 'values': [1, 2]}).status_code == 422
        assert client.post(base+'/resolve', json={**ids, 'context_key': 'forged'}).status_code == 422
        assert client.post(base+'/resolve', json={**ids, 'stable_ids': ['A', 'A']}).status_code == 422
        assert client.post(base+'/choose', json={**body, 'expected_preparation_digest': '0'*64}).status_code == 409
        assert client.post(base+'/choose', json=body).json()['status'] == 'preferred'
        assert client.post(base+'/resolve', json=ids).json()['recipe']['version_id'] == recipe['version_id']
        assert client.post(base+'/originals', json=ids).json()['entries'] == s['entries']
        assert client.post(base+'/resolve', json=ids).json()['status'] == 'none'
