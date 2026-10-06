from copy import deepcopy
import hashlib
import json
import sqlite3
import struct
import zlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from composition_versions import initialise_composition_schema, save_composition
from profile_versions import initialise_schema
from render_trial_router import create_render_trial_router
from render_trials import initialise_render_trial_schema, inspect_png
from test_composition_versions import add_default, resolve, TARGET

GENERATION = {'checkpoint': 'local-model.safetensors', 'checkpoint_sha256': None,
              'positive_prompt': 'An adult wearing dark polka-dot tights.\n Preserve this whitespace. ',
              'negative_prompt': '', 'seed': '18446744073709551615', 'width': 896, 'height': 1152,
              'steps': 20, 'sampler': 'dpmpp_2m', 'scheduler': 'sgm_uniform', 'guidance': 3.5,
              'denoise': 1, 'stage': 'first_pass'}
ASSESSMENT = {'identity': 'not_assessed', 'effect': 'absent', 'colour': 'absent',
              'outcome': 'no_improvement', 'notes': 'No clear dots.', 'regressions': ''}


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def png(metadata=None):
    text = b'' if metadata is None else chunk(b'tEXt', b'prompt\0' + json.dumps(metadata).encode())
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + text + chunk(b'IDAT', zlib.compress(b'\0\x01\x02\x03')) + chunk(b'IEND', b''))


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'render-trials.sqlite'
    conn = sqlite3.connect(path)
    initialise_schema(conn)
    initialise_composition_schema(conn)
    initialise_render_trial_schema(conn)
    roots = [add_default(conn, sid, sid) for sid in ('person', 'clothing')]
    entries = [{'stable_id': v['stable_id'], 'profile_version_id': v['version_id']} for v in roots]
    prepared = resolve(conn, entries, TARGET)
    recipe = save_composition(conn, name='Identity and tights', entries=entries, target_contract_id=TARGET,
                              expected_preparation_digest=prepared['preparation_digest'], preparation_resolver=resolve)
    app = FastAPI()
    app.include_router(create_render_trial_router(lambda: sqlite3.connect(path)))
    body = {'name': 'Tights strength trial', 'composition_version_id': recipe['version_id'],
            'generation': deepcopy(GENERATION), 'criteria': 'Preserve face, pattern and dark tights.',
            'baseline_trial_id': None, 'idempotency_key': 'trial-key'}
    yield TestClient(app), conn, path, recipe, body
    conn.close()


def trial(client, body):
    response = client.post('/api/render-trials', json=body)
    assert response.status_code == 200, response.text
    return response.json()


def attach(client, record, data=None, filename='result.png'):
    return client.post(f"/api/render-trials/{record['trial_id']}/evidence", params={'filename': filename},
                       content=png() if data is None else data, headers={'Content-Type': 'image/png'})


def assess(client, record, **changes):
    body = {'assessment': deepcopy(ASSESSMENT), 'expected_assessment_id': None, 'idempotency_key': 'assessment-key'}
    body.update(changes)
    return client.post(f"/api/render-trials/{record['trial_id']}/assessments", json=body)


def test_exact_recipe_large_seed_evidence_negative_outcome_and_correction_survive_restore(state, tmp_path):
    client, conn, path, recipe, body = state
    record = trial(client, body)
    assert record['receipt']['composition'] == recipe
    assert record['receipt']['declared_generation'] == GENERATION
    assert record['requires_revalidation'] and not record['generation_verified']
    assert trial(client, body)['trial_id'] == record['trial_id']
    data = png({'untrusted': '<script>do not execute</script>'})
    attached = attach(client, record, data).json()
    assert attached['evidence'][0]['sha256'] == hashlib.sha256(data).hexdigest()
    assert attached['evidence'][0]['metadata']['embedded_untrusted_metadata']['prompt']['untrusted'].startswith('<script>')
    assert not attached['evidence'][0]['metadata']['image_recipe_match_verified']
    assert len(attach(client, record, data).json()['evidence']) == 1
    first = assess(client, record).json()
    correction = {'assessment': {**ASSESSMENT, 'effect': 'partial', 'outcome': 'partial'},
                  'expected_assessment_id': first['assessments'][0]['assessment_id'], 'idempotency_key': 'correction'}
    corrected = assess(client, record, **correction).json()
    assert len(corrected['assessments']) == 2 and corrected['assessments'][0]['assessment'] == ASSESSMENT
    assert assess(client, record, **correction).json() == corrected
    assert assess(client, record, idempotency_key='stale-new-answer').status_code == 409
    route = f"/api/render-trials/{record['trial_id']}/evidence/{attached['evidence'][0]['evidence_id']}"
    response = client.get(route)
    assert response.content == data and response.headers['x-content-type-options'] == 'nosniff'
    # Same SQLite backup used by the app recovers exact image bytes and immutable answers.
    backup = sqlite3.connect(tmp_path / 'restored.sqlite')
    conn.backup(backup)
    assert backup.execute('SELECT png FROM lora_render_evidence').fetchone()[0] == data
    assert backup.execute('SELECT COUNT(*) FROM lora_render_assessments').fetchone()[0] == 2
    backup.close()
    for table in ('lora_render_trials', 'lora_render_evidence', 'lora_render_assessments'):
        for statement in (f'DELETE FROM {table}', f'UPDATE {table} SET trial_id="changed"'):
            with pytest.raises(sqlite3.IntegrityError, match='immutable'):
                conn.execute(statement)
            conn.rollback()


def test_controlled_baseline_rejects_changed_generation_and_accepts_unchanged_trial(state):
    client, conn, _, _, body = state
    baseline = trial(client, body)
    comparison = {**body, 'idempotency_key': 'comparison', 'name': 'Second variant', 'baseline_trial_id': baseline['trial_id']}
    comparison['generation'] = {**GENERATION, 'seed': '12'}
    assert client.post('/api/render-trials', json=comparison).status_code == 422
    assert conn.execute('SELECT COUNT(*) FROM lora_render_trials').fetchone()[0] == 1
    comparison['generation'] = GENERATION
    assert trial(client, comparison)['baseline_trial_id'] == baseline['trial_id']


@pytest.mark.parametrize('change', [
    {'generation': {**GENERATION, 'seed': 18446744073709551615}},
    {'generation': {**GENERATION, 'seed': '-1'}},
    {'generation': {**GENERATION, 'seed': '18446744073709551616'}},
    {'generation': {**GENERATION, 'width': True}},
    {'generation': {**GENERATION, 'width': 8192, 'height': 8192}},
    {'generation': {**GENERATION, 'denoise': 2}}, {'receipt': {'trusted': True}},
])
def test_invalid_inputs_or_browser_provenance_are_rejected_atomically(state, change):
    client, conn, _, _, body = state
    assert client.post('/api/render-trials', json={**body, **change}).status_code == 422
    assert conn.execute('SELECT COUNT(*) FROM lora_render_trials').fetchone()[0] == 0


def test_conflicting_idempotency_missing_evidence_and_false_acceptance(state):
    client, conn, _, _, body = state
    record = trial(client, body)
    assert client.post('/api/render-trials', json={**body, 'name': 'different'}).status_code == 409
    assert assess(client, record).status_code == 422
    attach(client, record)
    assert assess(client, record, assessment={**ASSESSMENT, 'outcome': 'accepted'}).status_code == 422
    assert conn.execute('SELECT COUNT(*) FROM lora_render_assessments').fetchone()[0] == 0
    assert assess(client, record).status_code == 200
    assert assess(client, record, assessment={**ASSESSMENT, 'notes': 'changed'}).status_code == 409


@pytest.mark.parametrize('data', [b'not a PNG', png()[:-3], png() + b'trailing', png().replace(b'IDAT', b'XDAT')])
def test_malformed_png_does_not_create_evidence(state, data):
    client, conn, _, _, body = state
    record = trial(client, body)
    assert attach(client, record, data).status_code == 422
    assert conn.execute('SELECT COUNT(*) FROM lora_render_evidence').fetchone()[0] == 0


def test_upload_paths_cross_trial_reads_and_limits(state, monkeypatch):
    client, _, _, _, body = state
    record = trial(client, body)
    assert attach(client, record, filename='../outside.png').status_code == 422
    evidence = attach(client, record).json()['evidence'][0]
    assert client.get(f"/api/render-trials/other/evidence/{evidence['evidence_id']}").status_code == 404
    monkeypatch.setattr('render_trial_router.MAX_PNG', 10)
    assert attach(client, record).status_code == 413
    assert client.post('/api/render-trials/missing/evidence?filename=a.png', content=b'x').status_code == 415


def test_plain_metadata_rejects_duplicate_or_nonfinite_json():
    original = png({'safe': True})
    extra = chunk(b'tEXt', b'prompt\0{"bad":NaN}')
    with pytest.raises(ValueError):
        inspect_png(original[:-12] + extra + original[-12:])
