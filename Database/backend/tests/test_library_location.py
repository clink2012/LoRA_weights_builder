"""Library selection persists per database and takes effect only on startup."""
from contextlib import closing
from pathlib import Path
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from catalogue_refresh import CatalogueError
from library_location import initialise_location_schema, selected_location
from library_scan_router import create_library_scan_router, install_library_scan
from test_library_scan import setup, wait, write_adapter


def test_setting_reopens_with_database_and_never_changes_current_session(setup, tmp_path):
    service, catalogue, root, db = setup
    new = tmp_path / 'new-root'
    new.mkdir()
    with closing(service.connection()) as conn:
        initialise_location_schema(conn)
    assert service.location()['active_root'] == str(root)
    saved = service.choose_location(str(new))
    assert saved['selected_root'] == str(new) and saved['restart_required']
    assert catalogue.root == root
    with closing(service.connection()) as conn:
        assert selected_location(conn, root) == new
        assert conn.execute('SELECT value FROM lora_user_profiles').fetchone()[0] == 'original profile'
        assert conn.execute('SELECT COUNT(*) FROM lora_catalogue_scans').fetchone()[0] == 0
    app = FastAPI()
    applied = []
    install_library_scan(app, lambda: service, lambda current: applied.append(current.root))
    with TestClient(app):
        wait(service)
        assert catalogue.root == new and applied == [new]
        assert not service.location()['restart_required']


@pytest.mark.parametrize('bad', ['', 'relative/path', '\0', '/missing-library-root-for-test'])
def test_invalid_root_cannot_overwrite_saved_setting(setup, bad):
    service, _, root, _ = setup
    with closing(service.connection()) as conn:
        initialise_location_schema(conn)
    service.choose_location(str(root))
    with pytest.raises(CatalogueError):
        service.choose_location(bad)
    assert service.location()['selected_root'] == str(root)


def test_file_and_symbolic_link_are_rejected(setup, tmp_path):
    service, _, root, _ = setup
    file = tmp_path / 'file'
    file.write_text('not a folder')
    with pytest.raises(CatalogueError, match='folder'):
        service.choose_location(str(file))
    link = tmp_path / 'link'
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip('Symbolic links unavailable for this Windows account')
    with pytest.raises(CatalogueError, match='links'):
        service.choose_location(str(link))


def test_strict_setting_route_preserves_no_path_scan_contract(setup, tmp_path):
    service, _, root, _ = setup
    with closing(service.connection()) as conn:
        initialise_location_schema(conn)
    app = FastAPI()
    app.include_router(create_library_scan_router(service))
    client = TestClient(app)
    assert client.get('/api/library-scan/location').json()['active_root'] == str(root)
    assert client.put('/api/library-scan/location', json={'root': str(root), 'database': 'other'}).status_code == 422
    assert client.put('/api/library-scan/location', json={'root': str(root)}).status_code == 200
    assert client.post('/api/library-scan', json={'root': str(root)}).status_code == 422


def test_api_startup_binds_services_together_without_opening_models(tmp_path, monkeypatch):
    import lora_api_server as api
    from catalogue_refresh import CatalogueService
    db, root = tmp_path / 'db', tmp_path / 'root'
    monkeypatch.setenv('LORA_ROOT', 'before')
    monkeypatch.setattr(api.catalogue_service, 'database', api.catalogue_service.database)
    monkeypatch.setattr(api.catalogue_service, 'root', api.catalogue_service.root)
    monkeypatch.setattr(api.compatibility_service, 'database', api.compatibility_service.database)
    monkeypatch.setattr(api.compatibility_service, 'root', api.compatibility_service.root)
    api.apply_selected_library_location(CatalogueService(db, root))
    assert Path(api.os.environ['LORA_ROOT']) == root
    assert api.catalogue_service.database == api.compatibility_service.database == db
    assert api.catalogue_service.root == api.compatibility_service.root == root
    assert not db.exists() and not root.exists()
