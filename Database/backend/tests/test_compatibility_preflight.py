"""Current-library preflight uses complete filtered counts and no write authority."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import struct

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import compatibility_preflight as preflight
from compatibility_router import create_compatibility_router
import flux_header_coverage as coverage
from inspire_export import build_inspire_flux1_export, LOADER_SHA256


def adapter(root, name, *, valid=True):
    prefix = 'lora_unet_double_blocks_0_img_attn_proj'
    header = {prefix + '.lora_up.weight': {'dtype': 'F16', 'shape': [3072, 4], 'data_offsets': [0, 24576]},
              prefix + '.lora_down.weight': {'dtype': 'F16', 'shape': [4, 3072], 'data_offsets': [24576, 49152]}}
    if not valid:
        header = {key.replace('lora_up', 'lora_B').replace('lora_down', 'lora_A'): value for key, value in header.items()}
    raw = json.dumps(header).encode()
    path = root / name
    with path.open('wb') as stream:
        stream.write(struct.pack('<Q', len(raw)) + raw)
        stream.truncate(8 + len(raw) + 49152)
    return path


@pytest.fixture
def state(tmp_path):
    root = tmp_path / 'loras'
    root.mkdir()
    database = tmp_path / 'library.db'
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute('CREATE TABLE lora(id INTEGER PRIMARY KEY,stable_id TEXT,filename TEXT,file_path TEXT,base_model_code TEXT,category_code TEXT)')
        conn.execute('CREATE TABLE lora_catalogue_presence(lora_id INTEGER,presence TEXT,observed_metadata_json TEXT)')
        conn.execute('CREATE TABLE lora_catalogue_scans(scan_id TEXT,completed_at TEXT)')
        conn.execute("INSERT INTO lora_catalogue_scans VALUES('scan-1','2026-10-03')")
    calls = []

    def evaluate(row):
        calls.append(row['stable_id'])
        tensors, identity = coverage.read_header(Path(row['file_path']))
        resolved = coverage.resolve_native_header(tensors)
        export = build_inspire_flux1_export(resolved['block_presence_baseline'], base_model_code='FLX',
                                           block_layout='flux_transformer_57', base_weight=1,
                                           resolved_patch_keys=resolved['resolved_patch_keys'], base_patch_keys=resolved['base_patch_keys'],
                                           coverage_complete=True, coverage_source=resolved['coverage_source'], loader_source_sha256=LOADER_SHA256)
        return {'loader_export': export}, {**resolved, 'file_identity': identity}

    service = preflight.CompatibilityService(database, root, evaluate, source_verifier=lambda: {'loader': 'pinned'})
    app = FastAPI()
    app.include_router(create_compatibility_router(service))
    return service, calls, TestClient(app)


def add(state, sid, name, *, valid=True, presence='current', base='FLX'):
    service, _, _ = state
    path = adapter(service.root, name, valid=valid)
    with closing(sqlite3.connect(service.database)) as conn, conn:
        row = conn.execute('INSERT INTO lora(stable_id,filename,file_path,base_model_code,category_code) VALUES(?,?,?,?,?)',
                           (sid, name, str(path), base, 'PPL'))
        conn.execute('INSERT INTO lora_catalogue_presence VALUES(?,?,?)', (row.lastrowid, presence, '{}'))
    return path


def query(state, **kwargs):
    return state[0].query(reference_stable_id='ref', target_contract_id=coverage.CONTRACT_ID, **kwargs)


def test_structural_filter_runs_before_pagination_and_excludes_missing_inventory(state):
    add(state, 'ref', 'z-reference.safetensors')
    add(state, 'bad', 'a-unsupported.safetensors', valid=False)
    add(state, 'good', 'b-good.safetensors')
    add(state, 'missing', 'c-old.safetensors', presence='missing')
    add(state, 'other', 'd-other.safetensors', base='LT5')
    result = query(state, limit=1)
    assert result['total'] == 2 and result['results'][0]['stable_id'] == 'good'
    assert result['counts'] == {'eligible': 2, 'excluded': 2, 'unknown': 0}
    assert query(state, limit=1, offset=1)['results'][0]['stable_id'] == 'ref'
    excluded = query(state, view='excluded')
    assert {row['stable_id'] for row in excluded['results']} == {'bad', 'other'}
    assert excluded['results'][0]['compatibility']['reason_code'] == 'unsupported_adapter'
    assert result['scope'] == 'structural_loader_checks_only'
    assert all('loader_export' not in row for row in result['results'])


def test_cache_invalidation_for_stat_and_new_scan_without_writes(state):
    service, calls, _ = state
    add(state, 'ref', 'ref.safetensors')
    path = add(state, 'other', 'other.safetensors')
    before = service.database.read_bytes()
    query(state)
    assert sorted(calls) == ['other', 'ref']
    calls.clear()
    result = query(state)
    assert calls == [] and result['freshness']['header_bytes_read'] == 32  # Initial/final reference plus two candidate readability checks.
    assert service.database.read_bytes() == before
    # Different length forces fresh validation and an explicit unknown status.
    with path.open('ab') as stream: stream.write(b'changed')
    result = query(state, view='all')
    assert calls == ['other']
    assert next(row for row in result['results'] if row['stable_id'] == 'other')['compatibility']['status'] == 'unknown'
    calls.clear()
    with closing(sqlite3.connect(service.database)) as conn, conn:
        conn.execute("INSERT INTO lora_catalogue_scans VALUES('scan-2','2026-10-04')")
    assert query(state)['freshness']['scan_id'] == 'scan-2'
    assert sorted(calls) == ['other', 'ref']


def test_loader_source_failure_prevents_cached_passes(state):
    add(state, 'ref', 'ref.safetensors')
    query(state)

    def changed():
        raise coverage.CoverageError('runtime_source_changed', 'Loader changed')

    state[0].source_verifier = changed
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'runtime_source_changed' and error.value.status == 503


def test_bad_reference_is_actionable_error_not_empty_compatible_list(state):
    add(state, 'ref', 'ref.safetensors', valid=False)
    add(state, 'good', 'good.safetensors')
    response = state[2].post('/api/catalogue/compatible', json={'reference_stable_id': 'ref', 'target_contract_id': coverage.CONTRACT_ID})
    assert response.status_code == 409
    detail = response.json()['detail']
    assert detail['reason_code'] == 'reference_not_supported'
    assert detail['reference']['reason_code'] == 'unsupported_adapter'
    assert 'results' not in response.json()


@pytest.mark.parametrize('budget', ['bytes', 'time', 'candidates'])
def test_exceeded_budget_never_returns_a_partial_list(state, monkeypatch, budget):
    add(state, 'ref', 'ref.safetensors')
    add(state, 'second', 'second.safetensors')
    monkeypatch.setattr(preflight, {'bytes': 'MAX_TOTAL_HEADER_BYTES', 'time': 'MAX_SECONDS', 'candidates': 'MAX_CANDIDATES'}[budget], 0)
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'preflight_budget'


def test_scan_change_during_check_rejects_whole_list(state):
    service, _, _ = state
    add(state, 'ref', 'ref.safetensors')
    evaluate = service.evaluate

    def changed(row):
        result = evaluate(row)
        with closing(sqlite3.connect(service.database)) as conn, conn:
            conn.execute("INSERT INTO lora_catalogue_scans VALUES('scan-new','2026-10-04')")
        return result

    service.evaluate = changed
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'catalogue_changed'


def test_changed_reference_during_later_candidate_check_rejects_whole_list(state):
    service, _, _ = state
    ref = add(state, 'ref', 'a-ref.safetensors')
    add(state, 'later', 'z-later.safetensors')
    evaluate = service.evaluate

    def changed(row):
        result = evaluate(row)
        if row['stable_id'] == 'later':
            with ref.open('ab') as stream: stream.write(b'changed')
        return result

    service.evaluate = changed
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'files_changed'


def test_noncurrent_reference_and_unknown_target_fail_closed(state):
    add(state, 'ref', 'ref.safetensors', presence='missing')
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'reference_not_current'
    with pytest.raises(preflight.PreflightError) as error:
        state[0].query(reference_stable_id='ref', target_contract_id='ltx-2.5')
    assert error.value.code == 'unsupported_target'


def test_http_paths_and_invalid_values_are_rejected(state):
    add(state, 'ref', 'ref.safetensors')
    body = {'reference_stable_id': 'ref', 'target_contract_id': coverage.CONTRACT_ID}
    for addition in ({'path': 'C:/'}, {'limit': True}, {'view': 'probably'}, {'offset': -1}):
        assert state[2].post('/api/catalogue/compatible', json={**body, **addition}).status_code == 422
    assert state[2].post('/api/catalogue/compatible', json=body).status_code == 200


def test_cache_is_bounded_and_file_outside_root_cannot_pass(state, monkeypatch):
    monkeypatch.setattr(preflight, 'MAX_CACHE_ENTRIES', 1)
    add(state, 'ref', 'ref.safetensors')
    add(state, 'other', 'other.safetensors')
    add(state, 'outside', 'outside.safetensors')
    with closing(sqlite3.connect(state[0].database)) as conn, conn:
        conn.execute("UPDATE lora SET file_path=? WHERE stable_id='outside'", (str(state[0].database),))
    result = query(state, view='all')
    assert len(state[0].cache) == 1
    assert next(row for row in result['results'] if row['stable_id'] == 'outside')['compatibility']['reason_code'] == 'outside_library'


def test_one_unreadable_candidate_is_unknown_and_retries_without_stat_change(state, monkeypatch):
    add(state, 'ref', 'ref.safetensors')
    denied = add(state, 'denied', 'denied.safetensors')
    query(state)  # Prove permission failure also invalidates an existing cached pass.
    before = denied.stat()
    original = Path.open

    def blocked(path, *args, **kwargs):
        if path == denied:
            raise PermissionError('synthetic per-file denial')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', blocked)
    result = query(state, view='all')
    assert result['counts'] == {'eligible': 1, 'excluded': 0, 'unknown': 1}
    assert next(row for row in result['results'] if row['stable_id'] == 'denied')['compatibility']['reason_code'] == 'unreadable_file'
    assert query(state)['total'] == 1
    monkeypatch.setattr(Path, 'open', original)
    assert query(state)['total'] == 2
    assert denied.stat().st_mtime_ns == before.st_mtime_ns


def test_header_reader_wrapped_io_failure_is_not_cached_as_invalid_format(state):
    service, calls, _ = state
    add(state, 'ref', 'ref.safetensors')
    add(state, 'denied', 'denied.safetensors')
    original = service.evaluate

    def failing(row):
        if row['stable_id'] == 'denied':
            try:
                raise PermissionError('Header read access changed')
            except PermissionError as cause:
                raise coverage.CoverageError('invalid_header', 'Header could not be read safely') from cause
        return original(row)

    service.evaluate = failing
    assert query(state, view='all')['counts']['unknown'] == 1
    service.evaluate = original
    calls.clear()
    assert query(state)['total'] == 2
    assert 'denied' in calls


def test_unavailable_library_root_remains_global_failure(state):
    add(state, 'ref', 'ref.safetensors')
    state[0].root = state[0].root / 'not-mounted'
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'preflight_unavailable' and error.value.status == 503


def test_per_file_stat_denial_rechecks_after_access_returns(state, monkeypatch):
    add(state, 'ref', 'ref.safetensors')
    denied = add(state, 'denied', 'denied.safetensors')
    query(state)
    original = preflight._stat_identity

    def blocked(path):
        if path == denied:
            raise PermissionError('Synthetic candidate stat denial')
        return original(path)

    monkeypatch.setattr(preflight, '_stat_identity', blocked)
    assert query(state, view='all')['counts'] == {'eligible': 1, 'excluded': 0, 'unknown': 1}
    monkeypatch.setattr(preflight, '_stat_identity', original)
    assert query(state)['total'] == 2


def test_changed_source_pins_invalidate_cached_outcomes(state):
    add(state, 'ref', 'ref.safetensors')
    service, calls, _ = state
    first = query(state)
    calls.clear()
    service.source_verifier = lambda: {'loader': 'new-approved-pin'}
    second = query(state)
    assert calls == ['ref']
    assert first['freshness']['contract_fingerprint'] != second['freshness']['contract_fingerprint']


@pytest.mark.parametrize('filter_reference_out', [False, True])
def test_reference_becoming_unreadable_during_candidates_rejects_list(state, monkeypatch, filter_reference_out):
    service, _, _ = state
    reference = add(state, 'ref', 'z-reference.safetensors')
    add(state, 'other', 'a-other.safetensors')
    original_open, original_evaluate = Path.open, service.evaluate
    denied = False

    def read(path, *args, **kwargs):
        if path == reference and denied:
            raise PermissionError('Reference access changed')
        return original_open(path, *args, **kwargs)

    def evaluate(row):
        nonlocal denied
        result = original_evaluate(row)
        if row['stable_id'] == 'other':
            denied = True
        return result

    monkeypatch.setattr(Path, 'open', read)
    service.evaluate = evaluate
    with pytest.raises(preflight.PreflightError) as error:
        query(state, **({'search': 'other'} if filter_reference_out else {}))
    assert error.value.code == 'reference_not_supported'
    assert error.value.context['reference']['reason_code'] == 'unreadable_file'


def test_replaced_root_link_cannot_grant_eligibility(state):
    service = state[0]
    add(state, 'ref', 'ref.safetensors')
    moved = service.root.with_name('moved-root')
    service.root.rename(moved)
    try:
        service.root.symlink_to(moved, target_is_directory=True)
    except OSError:
        moved.rename(service.root)
        pytest.skip('Directory symlinks are unavailable on this host')
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'preflight_unavailable'


def test_intermediate_directory_link_is_not_followed(state):
    service = state[0]
    add(state, 'ref', 'ref.safetensors')
    add(state, 'other', 'other.safetensors')
    target = service.root.parent / 'outside'
    target.mkdir()
    adapter(target, 'other.safetensors')
    link = service.root / 'linked'
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('Directory symlinks are unavailable on this host')
    with closing(sqlite3.connect(service.database)) as conn, conn:
        conn.execute("UPDATE lora SET file_path=? WHERE stable_id='other'", (str(link / 'other.safetensors'),))
    result = query(state, view='all')
    assert result['counts'] == {'eligible': 1, 'excluded': 0, 'unknown': 1}
    assert next(row for row in result['results'] if row['stable_id'] == 'other')['compatibility']['reason_code'] == 'unsupported_file_path'


def test_directory_identity_replacement_during_query_rejects_whole_list(state, monkeypatch):
    service = state[0]
    add(state, 'ref', 'ref.safetensors')
    add(state, 'other', 'other.safetensors')
    original_stat, original_evaluate = Path.lstat, service.evaluate
    changed = False

    def replacement(path):
        info = original_stat(path)
        if path == service.root and changed:
            from types import SimpleNamespace
            return SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino + 1, st_mode=info.st_mode,
                                   st_file_attributes=getattr(info, 'st_file_attributes', 0))
        return info

    def evaluate(row):
        nonlocal changed
        result = original_evaluate(row)
        if row['stable_id'] == 'other':
            changed = True
        return result

    monkeypatch.setattr(Path, 'lstat', replacement)
    service.evaluate = evaluate
    with pytest.raises(preflight.PreflightError) as error:
        query(state)
    assert error.value.code == 'files_changed'
