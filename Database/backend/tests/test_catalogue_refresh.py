"""File inventory changes catalogue presence without changing analysis/history."""
from pathlib import Path
import sqlite3
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalogue_refresh as catalogue
from catalogue_router import create_catalogue_router


SCHEMA = '''CREATE TABLE lora (
 id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT UNIQUE NOT NULL, filename TEXT NOT NULL,
 stable_id TEXT, base_model_name TEXT,base_model_code TEXT,category_name TEXT,category_code TEXT,
 model_family TEXT,lora_type TEXT,rank INTEGER,has_block_weights INTEGER DEFAULT 0,
 block_layout TEXT,clip_contributor INTEGER DEFAULT 0,clip_tensor_count INTEGER DEFAULT -1,
 last_modified REAL NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)'''


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'loras'
    root.mkdir()
    database = tmp_path / 'catalogue.db'
    with sqlite3.connect(database) as conn:
        conn.execute(SCHEMA)
        conn.execute('CREATE TABLE lora_user_profiles (stable_id TEXT, saved_values TEXT)')
        conn.execute("INSERT INTO lora_user_profiles VALUES ('FLX-PPL-1500','original saved values')")
        conn.commit()
        catalogue.initialise_catalogue_schema(conn)
    service = catalogue.CatalogueService(database, root)
    app = FastAPI()
    app.include_router(create_catalogue_router(service))
    return service, root, database, TestClient(app)


def add_file(root, relative='FLUX/01 - People/current.safetensors', data=b'No tensor header needed'):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def existing(database, path, sid='FLX-PPL-999', **values):
    fields = {'file_path': str(path), 'filename': path.name, 'stable_id': sid,
              'last_modified': 123, 'created_at': 'old', 'updated_at': 'old',
              'base_model_code': 'FLX', 'category_code': 'PPL', 'has_block_weights': 1, 'block_layout': 'legacy-layout'}
    fields.update(values)
    with sqlite3.connect(database) as conn:
        conn.execute(f"INSERT INTO lora ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", tuple(fields.values()))


def all_rows(database, table):
    with sqlite3.connect(database) as conn:
        return conn.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()


def test_refresh_preserves_old_rows_and_history_adds_new_and_tracks_missing(setup):
    service, root, db, _ = setup
    present = add_file(root)
    existing(db, present)
    missing = root / 'FLUX/01 - People/removed.safetensors'
    existing(db, missing, 'FLX-PPL-1000')
    add_file(root, 'FLUX/01 - People/new.safetensors')
    before_rows, before_profiles = all_rows(db, 'lora'), all_rows(db, 'lora_user_profiles')
    receipt = service.refresh()
    assert receipt['counts'] == {'discovered': 2, 'added': 1, 'assigned_existing_ids': 0,
                                 'present': 2, 'missing': 1, 'out_of_scope': 0}
    assert receipt['architecture_verified'] is False and receipt['tensor_payload_read'] is False
    assert all_rows(db, 'lora')[:2] == before_rows
    assert all_rows(db, 'lora_user_profiles') == before_profiles
    new = next(row for row in service.search()['results'] if row['filename'] == 'new.safetensors')
    assert new['stable_id'] == 'FLX-PPL-1501'  # Reserve missing rows AND history-only IDs.
    assert new['has_block_weights'] == 0 and new['block_layout'] is None
    assert new['role'] == 'character' and new['role_source'] == 'folder_hint'
    assert new['metadata_provenance']['architecture_verified'] is False
    assert service.search(presence='missing')['results'][0]['stable_id'] == 'FLX-PPL-1000'
    second = service.refresh()
    assert second['counts']['added'] == 0
    assert second['inventory_sha256'] == receipt['inventory_sha256']
    assert len(all_rows(db, 'lora_catalogue_scans')) == 2


def test_removed_and_returned_file_reuses_same_identity(setup):
    service, root, db, _ = setup
    path = add_file(root)
    service.refresh()
    sid = service.search()['results'][0]['stable_id']
    path.unlink()
    service.refresh()
    assert service.search()['results'] == []
    assert service.search(presence='missing')['results'][0]['stable_id'] == sid
    path.write_bytes(b'Changed file contents; new preparation must inspect it')
    service.refresh()
    assert service.search()['results'][0]['stable_id'] == sid
    assert len(all_rows(db, 'lora')) == 1


def test_filename_relocation_is_never_automatically_relinked(setup):
    service, root, db, _ = setup
    path = add_file(root)
    existing(db, path)
    path.rename(root / 'relocated.safetensors')
    receipt = service.refresh()
    assert receipt['counts']['missing'] == 1 and receipt['counts']['added'] == 1
    assert service.search()['results'][0]['stable_id'].startswith('UNK-UNK-')
    assert service.search(presence='missing')['results'][0]['stable_id'] == 'FLX-PPL-999'


@pytest.mark.parametrize('folder,code', [('LTXV2','LTX'), ('LTXV2_5','LT5'), ('MiniMax-H3','MH3'), ('Flux.2-Klein','F2K')])
def test_new_families_are_folder_metadata_only(setup, folder, code):
    service, root, _, _ = setup
    add_file(root, folder + '/08 - Clothing/item.safetensors')
    service.refresh()
    result = service.search(base=code)['results'][0]
    assert result['base_model_code'] == code
    assert result['stable_id'] == code + '-CLT-001'
    assert result['has_block_weights'] == 0 and result['block_layout'] is None
    assert result['metadata_provenance']['basis'] == 'folder_hint'


def test_blank_id_assignment_leaves_existing_metadata_untouched(setup):
    service, root, db, _ = setup
    path = add_file(root, 'WAN2.2/T2V/unclassified/item.safetensors')
    existing(db, path, None, base_model_code=None, category_code=None)
    receipt = service.refresh()
    row = service.search()['results'][0]
    assert receipt['counts']['assigned_existing_ids'] == 1
    assert row['stable_id'] == 'W22-UNK-001'
    assert row['base_model_code'] is None and row['category_code'] is None
    assert row['metadata_provenance']['base_model_code'] == 'W22'


@pytest.mark.parametrize('change', ['file', 'directory'])
def test_change_before_commit_rolls_back_new_rows_ids_and_presence(setup, monkeypatch, change):
    service, root, db, _ = setup
    path = add_file(root)
    existing(db, path, None)
    before = all_rows(db, 'lora')
    verify = catalogue.verify_inventory
    def changed(inventory, started):
        if change == 'file':
            path.write_bytes(b'changed')
        else:
            add_file(root, 'newfolder/unseen.safetensors')
        verify(inventory, started)
    monkeypatch.setattr(catalogue, 'verify_inventory', changed)
    with pytest.raises(catalogue.CatalogueError, match='changed'):
        service.refresh()
    assert all_rows(db, 'lora') == before
    assert all_rows(db, 'lora_catalogue_presence') == []
    assert all_rows(db, 'lora_catalogue_scans') == []


def test_http_rejects_paths_and_invalid_filters_and_handles_missing_root(setup):
    service, root, _, client = setup
    assert client.get('/api/catalogue').json()['catalogue_status'] == 'not_refreshed'
    assert client.post('/api/catalogue/refresh', json={'root': 'C:/'}).status_code == 422
    assert client.get('/api/catalogue?presence=deleted').status_code == 422
    assert client.get('/api/catalogue?limit=0').status_code == 422
    root.rmdir()
    response = client.post('/api/catalogue/refresh', json={})
    assert response.status_code == 503
    assert response.json()['detail']['reason_code'] == 'refresh_unavailable'


def test_current_missing_all_pagination_and_filename_search(setup):
    service, root, db, client = setup
    add_file(root, 'FLUX/01 - People/alpha.safetensors')
    add_file(root, 'FLUX/01 - People/beta.safetensors')
    existing(db, root / 'FLUX/01 - People/old.safetensors')
    assert client.post('/api/catalogue/refresh', json={}).status_code == 200
    assert service.search(limit=1, offset=1)['results'][0]['filename'] == 'beta.safetensors'
    assert service.search(presence='all')['total'] == 3
    assert service.search(search='ALP', category='PPL')['total'] == 1


def test_retired_global_reindex_cannot_invoke_legacy_tensor_engine(monkeypatch):
    import lora_api_server as api
    monkeypatch.setattr(api, 'index_all_loras', lambda: pytest.fail('Legacy indexer must never run'))
    monkeypatch.setattr(api, 'assign_stable_ids', lambda: pytest.fail('Legacy ID assigner must never run'))
    response = TestClient(api.app).post('/api/lora/reindex_all')
    assert response.status_code == 410
    assert response.json()['detail']['reason_code'] == 'legacy_reindex_retired'
