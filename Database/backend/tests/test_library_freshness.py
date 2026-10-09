"""Opening a browser can detect library drift without rewriting owner history."""
import json
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import catalogue_refresh as catalogue
from library_scan_router import create_library_scan_router
from library_scan_service import LibraryScanService
from test_catalogue_refresh import setup, add_file, all_rows


def test_missing_initial_inventory_is_not_current_even_for_empty_root(setup):
    service, _, db, _ = setup
    before = db.read_bytes()
    result = service.freshness()
    assert result['status'] == 'not_scanned'
    assert result['catalogue_scan_id'] is None
    assert result['counts']['discovered'] == 0
    assert db.read_bytes() == before


def test_detects_added_removed_returned_and_modified_without_any_writes(setup, monkeypatch):
    service, root, db, _ = setup
    changed = add_file(root, 'FLUX/People/changed.safetensors')
    removed = add_file(root, 'FLUX/People/removed.safetensors')
    returned = add_file(root, 'FLUX/People/returned.safetensors')
    service.refresh()
    returned.unlink()
    receipt = service.refresh()
    assert service.freshness()['status'] == 'current'  # retained missing history is not new drift
    changed.write_bytes(b'new source size and content')
    removed.unlink()
    returned.write_bytes(b'returned source')
    add_file(root, 'FLUX/People/new.safetensors')
    # Any model file open (including its header) would fail this check.
    monkeypatch.setattr(catalogue.Path, 'open', lambda *args, **kwargs: pytest.fail('No model payload/header open permitted'))
    with open(db, 'rb') as stream:
        before = stream.read()
    profiles = all_rows(db, 'lora_user_profiles')
    result = service.freshness()
    assert result['status'] == 'outdated'
    assert result['catalogue_scan_id'] == receipt['scan_id']
    assert result['counts'] == {'discovered': 3, 'added': 1, 'removed': 1, 'returned': 1, 'changed': 1}
    assert not result['tensor_payload_read'] and not result['export_verified']
    with open(db, 'rb') as stream:
        assert stream.read() == before
    assert all_rows(db, 'lora_user_profiles') == profiles


def test_refresh_makes_drift_current_and_keeps_saved_identity(setup):
    service, root, db, _ = setup
    path = add_file(root)
    service.refresh()
    sid = service.search()['results'][0]['stable_id']
    path.write_bytes(b'edited file')
    assert service.freshness()['counts']['changed'] == 1
    service.refresh()
    assert service.freshness()['status'] == 'current'
    assert service.search()['results'][0]['stable_id'] == sid
    assert len(all_rows(db, 'lora')) == 1


def test_different_configured_root_is_outdated_without_relinking(setup, tmp_path):
    service, root, db, _ = setup
    add_file(root)
    service.refresh()
    replacement = tmp_path / 'moved'
    replacement.mkdir()
    result = catalogue.CatalogueService(db, replacement).freshness()
    assert result['status'] == 'outdated' and result['root_changed']
    assert result['counts']['removed'] == 1
    assert len(all_rows(db, 'lora')) == 1


def test_unreadable_or_changing_tree_cannot_report_current(setup, monkeypatch):
    service, root, db, _ = setup
    service.refresh()
    original = catalogue.discover
    calls = 0
    def changing(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls == 2:
            result['excluded'].append('new link')
        return result
    monkeypatch.setattr(catalogue, 'discover', changing)
    with pytest.raises(catalogue.CatalogueError, match='changed'):
        service.freshness()
    monkeypatch.setattr(catalogue, 'discover', original)
    root.rmdir()
    with pytest.raises(catalogue.CatalogueError) as error:
        service.freshness()
    assert error.value.code == 'freshness_unavailable'


def test_newer_inventory_cannot_receive_old_comparison(setup, monkeypatch):
    service, _, db, _ = setup
    service.refresh()
    connection = service.connection
    calls = 0
    def replaced():
        nonlocal calls
        calls += 1
        if calls == 2:
            with sqlite3.connect(db) as conn:
                conn.execute('INSERT INTO lora_catalogue_scans VALUES(?,?,?,?)', ('newer', 'later', str(service.root), json.dumps({})))
        return connection()
    monkeypatch.setattr(service, 'connection', replaced)
    with pytest.raises(catalogue.CatalogueError) as error:
        service.freshness()
    assert error.value.code == 'scan_superseded'


def test_freshness_endpoint_read_only_and_reports_budgets(setup, monkeypatch):
    catalogue_service, root, db, _ = setup
    service = LibraryScanService(catalogue_service)
    app = FastAPI()
    app.include_router(create_library_scan_router(service))
    client = TestClient(app)
    assert client.get('/api/library-scan/freshness').json()['status'] == 'not_scanned'
    monkeypatch.setattr(catalogue, 'MAX_FILES', 0)
    add_file(root)
    result = client.get('/api/library-scan/freshness')
    assert result.status_code == 409
    assert result.json()['detail']['reason_code'] == 'inventory_budget'
    assert all_rows(db, 'lora_catalogue_scans') == []
