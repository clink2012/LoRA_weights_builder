"""Background startup scans stay bounded and cannot turn observations into support."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import struct
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from adapter_identity import identify_file
from catalogue_refresh import CatalogueError, CatalogueService, initialise_catalogue_schema
from library_scan_router import create_library_scan_router, install_library_scan
from library_scan_service import LibraryScanService, initialise_library_scan_schema
from test_catalogue_refresh import SCHEMA


def write_adapter(root, relative='FLUX/People/example.safetensors', module='double_blocks.0.img_attn.proj', metadata=None):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {module + '.lora_A.weight': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [0, 8]},
              module + '.lora_B.weight': {'dtype': 'F32', 'shape': [2, 1], 'data_offsets': [8, 16]}}
    if metadata:
        header['__metadata__'] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(16))
    return path


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'loras'
    root.mkdir()
    database = tmp_path / 'catalogue.db'
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute(SCHEMA)
        conn.execute('CREATE TABLE lora_user_profiles (stable_id TEXT,value TEXT)')
        conn.execute("INSERT INTO lora_user_profiles VALUES ('FLX-PPL-900','original profile')")
        conn.commit()
        initialise_catalogue_schema(conn)
        initialise_library_scan_schema(conn)
    catalogue = CatalogueService(database, root)
    service = LibraryScanService(catalogue)
    yield service, catalogue, root, database
    service.shutdown()


def wait(service):
    service._thread.join(timeout=5)
    assert not service._thread.is_alive()
    return service.status()


def test_startup_once_and_manual_active_request_coalesces(setup):
    service, catalogue, root, db = setup
    write_adapter(root)
    entered, release = threading.Event(), threading.Event()
    original = service.inspector

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    service.inspector = held
    first = service.startup()
    assert entered.wait(2)
    assert service.startup()['job_id'] == first['job_id']
    assert service.start()['job_id'] == first['job_id']
    release.set()
    status = wait(service)
    assert status['status'] == 'complete'
    assert status['scanned'] == status['total'] == 1
    assert status['remaining'] == 0
    assert service.startup()['job_id'] == first['job_id']
    assert service.start()['job_id'] != first['job_id']
    wait(service)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora_catalogue_scans').fetchone()[0] == 2
        assert conn.execute('SELECT value FROM lora_user_profiles').fetchone()[0] == 'original profile'


def test_disagreement_persisted_but_file_and_folder_never_changed(setup):
    service, _, root, db = setup
    path = write_adapter(root, 'LTXV2/People/misfiled.safetensors', metadata={'ss_base_model_version': 'flux2_klein_9b'})
    before = path.read_bytes()
    service.startup()
    status = wait(service)
    assert status['issue_count'] == 1
    item = service.issues()['results'][0]
    assert item['source_status'] == 'current'
    assert item['observation']['issues'][0]['code'] == 'family_disagreement'
    assert not item['observation']['architecture_verified'] and not item['observation']['export_verified']
    assert path.read_bytes() == before
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT file_path FROM lora').fetchone()[0] == str(path)


def test_unknown_family_is_not_reported_as_corrupt_or_compatible(setup):
    service, _, root, _ = setup
    write_adapter(root, 'SDXL/People/ordinary.safetensors', 'lora_unet_input_blocks_0_proj')
    service.start()
    status = wait(service)
    assert status['issue_count'] == 0
    assert status['unchecked_family_count'] == 1
    assert not status['architecture_verified']
    assert service.issues()['total'] == 0


def test_cleaned_library_needs_no_retired_folders_and_preserves_missing_history(setup):
    service, _, root, db = setup
    # Old rows must survive removing entire model folders, with no scan failure.
    with sqlite3.connect(db) as conn:
        for code, folder in [('PNY', 'PONY'), ('SDX', 'SDXL'), ('ILL', 'Illustrious')]:
            conn.execute("INSERT INTO lora (stable_id, filename, file_path, base_model_code, last_modified, created_at, updated_at) VALUES (?, ?, ?, ?, 0, 'old', 'old')",
                         (code + '-PPL-001', 'old.safetensors', str(root / folder / 'People/old.safetensors'), code))
    write_adapter(root)
    service.startup()
    status = wait(service)
    assert status['status'] == 'complete'
    assert status['total'] == 1
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT l.base_model_code,p.presence FROM lora l JOIN lora_catalogue_presence p ON p.lora_id=l.id WHERE l.base_model_code IN ('PNY','SDX','ILL')").fetchall()
        assert sorted(rows) == [('ILL', 'missing'), ('PNY', 'missing'), ('SDX', 'missing')]
        assert conn.execute('SELECT value FROM lora_user_profiles').fetchone()[0] == 'original profile'


def test_partial_batch_can_resume_without_repeating_checked_files(setup):
    service, catalogue, root, db = setup
    for number in range(3):
        write_adapter(root, f'FLUX/People/{number}.safetensors')
    service.max_files = 1
    service.start()
    first = wait(service)
    assert (first['status'], first['scanned'], first['remaining']) == ('partial', 1, 2)
    service.start(resume=True)
    second = wait(service)
    assert (second['status'], second['scanned'], second['remaining']) == ('partial', 2, 1)
    service.start(resume=True)
    third = wait(service)
    assert (third['status'], third['scanned'], third['remaining']) == ('complete', 3, 0)
    assert first['catalogue_scan_id'] == third['catalogue_scan_id']
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora_catalogue_scans').fetchone()[0] == 1


def test_byte_budget_is_reserved_before_header_inspector_reads(setup):
    service, _, root, _ = setup
    write_adapter(root)
    service.max_bytes = 20
    service.inspector = lambda *_args, **_kwargs: pytest.fail('No header budget available')
    service.start()
    status = wait(service)
    assert (status['status'], status['scanned'], status['remaining']) == ('partial', 0, 1)
    assert status['reason_code'] == 'audit_budget'


def test_unreadable_and_unsupported_headers_remain_visible_without_corruption_claim(setup):
    service, _, root, _ = setup
    path = write_adapter(root)
    path.write_bytes(struct.pack('<Q', 0))
    huge = root / 'FLUX/People/huge.safetensors'
    huge.write_bytes(struct.pack('<Q', 17 * 1024 * 1024))
    service.start()
    status = wait(service)
    assert status['issue_count'] == 2
    items = service.issues()['results']
    codes = {item['observation']['issues'][0]['code'] for item in items}
    assert codes == {'header_budget', 'invalid_header'}
    assert next(item for item in items if item['filename'] == 'huge.safetensors')['observation']['status'] == 'not_checked'


def test_nonfloating_adapter_is_not_checked_not_called_corrupt(setup):
    service, _, root, _ = setup
    path = write_adapter(root)
    raw = path.read_bytes()
    length = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+length])
    for value in header.values():
        value['dtype'] = 'I8'
        value['shape'] = [8]
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + bytes(16))
    service.start()
    wait(service)
    observation = service.issues()['results'][0]['observation']
    assert observation['status'] == 'not_checked'
    assert observation['issues'][0]['code'] == 'unsupported_tensor'
    assert observation['issues'][0]['severity'] == 'not_checked'


def test_source_replaced_during_audit_cannot_gain_current_observation(setup):
    service, _, root, _ = setup
    path = write_adapter(root)

    def replace_after_read(*args, **kwargs):
        result = identify_file(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b'changed')
        return result

    service.inspector = replace_after_read
    service.start()
    wait(service)
    item = service.issues()['results'][0]
    assert item['source_status'] == 'stale'
    assert item['requires_revalidation']
    assert item['observation']['issues'][0]['code'] == 'file_changed'
    assert 'identification' not in item['observation']


def test_post_read_file_change_still_consumes_full_reserved_header_budget(setup):
    service, _, root, _ = setup
    first = write_adapter(root, 'FLUX/People/first.safetensors')
    write_adapter(root, 'FLUX/People/second.safetensors')
    length = struct.unpack('<Q', first.read_bytes()[:8])[0]
    service.max_bytes = 16 + length + 8
    inspected = []

    def replaced(path, **kwargs):
        inspected.append(path)
        result = identify_file(path, **kwargs)
        path.write_bytes(path.read_bytes() + b'changed')
        return result

    service.inspector = replaced
    service.start()
    result = wait(service)
    assert result['status'] == 'partial'
    assert result['scanned'] == 1 and result['remaining'] == 1
    assert len(inspected) == 1
    assert result['header_bytes_accounted'] == 16 + length


def test_new_catalogue_scan_prevents_old_audit_overwrite(setup):
    service, catalogue, root, db = setup
    write_adapter(root)
    replacement = {}

    def supersede(*args, **kwargs):
        result = identify_file(*args, **kwargs)
        replacement.update(catalogue.refresh())
        return result

    service.inspector = supersede
    service.start()
    status = wait(service)
    assert status['reason_code'] == 'scan_superseded'
    assert service.issues()['catalogue_scan_id'] == replacement['scan_id']
    assert service.issues()['total'] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora_library_observations').fetchone()[0] == 0


def test_cancellation_during_header_read_saves_no_late_observation(setup):
    service, _, root, db = setup
    write_adapter(root)

    def cancel_after_read(*args, **kwargs):
        result = identify_file(*args, **kwargs)
        service.cancel()
        return result

    service.inspector = cancel_after_read
    service.start()
    status = wait(service)
    assert status['status'] == 'cancelled' and status['remaining'] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora_library_observations').fetchone()[0] == 0


def test_cancellation_after_inventory_sql_rolls_back_catalogue(setup, monkeypatch):
    service, catalogue, root, db = setup
    write_adapter(root)
    original = catalogue._apply

    def cancel_after_apply(*args, **kwargs):
        result = original(*args, **kwargs)
        service.cancel()
        return result

    monkeypatch.setattr(catalogue, '_apply', cancel_after_apply)
    service.start()
    status = wait(service)
    assert status['status'] == 'cancelled'
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM lora_catalogue_scans').fetchone()[0] == 0


def test_router_rejects_paths_and_uses_late_provider(setup):
    service, _, root, _ = setup
    app = FastAPI()
    calls = []
    app.include_router(create_library_scan_router(lambda: calls.append(True) or service))
    with TestClient(app) as client:
        assert client.post('/api/library-scan', json={'root': 'elsewhere'}).status_code == 422
        assert not calls
        assert client.get('/api/library-scan').json()['status'] == 'idle'
        assert calls


def test_disabled_startup_never_resolves_database_factory(monkeypatch):
    monkeypatch.setenv('LORA_DISABLE_STARTUP_SCAN', '1')
    app = FastAPI()
    install_library_scan(app, lambda: pytest.fail('Disabled startup must not access database'))
    with TestClient(app) as client:
        assert client.get('/api/library-scan').status_code == 503


def test_enabled_startup_resolves_selected_service_late(setup, monkeypatch):
    service, _, root, _ = setup
    monkeypatch.delenv('LORA_DISABLE_STARTUP_SCAN', raising=False)
    write_adapter(root)
    app = FastAPI()
    selected = [None]
    install_library_scan(app, lambda: selected[0])
    selected[0] = service
    with TestClient(app):
        assert wait(service)['status'] == 'complete'


def test_cross_process_lease_prevents_duplicate_jobs(setup):
    service, catalogue, root, _ = setup
    write_adapter(root)
    second = LibraryScanService(catalogue)
    entered, release = threading.Event(), threading.Event()

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return identify_file(*args, **kwargs)

    service.inspector = held
    service.start()
    assert entered.wait(2)
    try:
        with pytest.raises(CatalogueError, match='Another app process'):
            second.start()
    finally:
        release.set()
    assert wait(service)['status'] == 'complete'
    second.shutdown()


def test_startup_busy_is_visible_nonfatal_and_manual_retry_succeeds(setup):
    service, catalogue, root, _ = setup
    write_adapter(root)
    second = LibraryScanService(catalogue)
    entered, release = threading.Event(), threading.Event()

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return identify_file(*args, **kwargs)

    service.inspector = held
    service.start()
    assert entered.wait(2)
    try:
        status = second.startup()
        assert status['status'] == 'failed' and status['reason_code'] == 'scan_busy'
        assert second._thread is None
        assert second.startup()['reason_code'] == 'scan_busy'
    finally:
        release.set()
    wait(service)
    try:
        second.start()
        assert wait(second)['status'] == 'complete'
    finally:
        second.shutdown()


def test_startup_storage_failure_is_not_swallowed(setup, monkeypatch):
    service, _, _, _ = setup

    def failed():
        raise sqlite3.OperationalError('Database unavailable')

    monkeypatch.setattr(service, 'connection', failed)
    with pytest.raises(sqlite3.OperationalError, match='unavailable'):
        service.startup()


def test_elapsed_budget_does_not_claim_clean_library(setup):
    service, _, root, _ = setup
    write_adapter(root)
    service.max_seconds = 0
    service.start()
    result = wait(service)
    assert result['status'] == 'partial'
    assert result['scanned'] == 0 and result['remaining'] == 1


def test_resume_checks_stat_before_reusing_previous_observation(setup):
    service, _, root, _ = setup
    path = write_adapter(root)
    service.start()
    assert wait(service)['status'] == 'complete'
    path.write_bytes(path.read_bytes() + b'changed')
    service.start(resume=True)
    assert wait(service)['issue_count'] == 1
    assert service.issues()['results'][0]['source_status'] == 'stale'


def test_empty_inventory_cannot_complete_after_superseding_scan(setup, monkeypatch):
    service, catalogue, _, _ = setup
    original = service._current_rows

    def replace(scan_id):
        result = original(scan_id)
        catalogue.refresh()
        return result

    monkeypatch.setattr(service, '_current_rows', replace)
    service.start()
    assert wait(service)['reason_code'] == 'scan_superseded'


def test_old_issue_records_remain_historical_when_new_inventory_replaces_them(setup):
    service, catalogue, root, db = setup
    path = write_adapter(root, 'LTXV2/People/misfiled.safetensors')
    service.start()
    wait(service)
    old_id = service.status()['catalogue_scan_id']
    path.unlink()
    service.start()
    status = wait(service)
    assert status['catalogue']['counts']['missing'] == 1
    assert service.issues()['results'] == []
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM lora_library_observations WHERE scan_id=?', (old_id,)).fetchone()[0] == 1
