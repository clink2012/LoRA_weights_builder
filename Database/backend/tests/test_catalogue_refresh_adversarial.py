"""Independent refresh checks: disk inventory must never erase saved identity."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import threading

import pytest

import catalogue_refresh as catalogue


@pytest.fixture
def state(tmp_path):
    root = tmp_path / 'library'
    root.mkdir()
    database = tmp_path / 'catalogue.db'
    with closing(sqlite3.connect(database)) as conn:
        conn.execute('''CREATE TABLE lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL, base_model_name TEXT, base_model_code TEXT,
            category_name TEXT, category_code TEXT, model_family TEXT, lora_type TEXT,
            rank INTEGER, has_block_weights INTEGER NOT NULL DEFAULT 0, block_layout TEXT,
            clip_contributor INTEGER NOT NULL DEFAULT 0, clip_tensor_count INTEGER NOT NULL DEFAULT -1,
            last_modified REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            stable_id TEXT)''')
        # These orphan IDs represent durable history whose catalogue row was lost.
        for table in catalogue.HISTORY_TABLES[1:]:
            conn.execute(f'CREATE TABLE {table} (stable_id TEXT, preserved_payload TEXT)')
        conn.commit()
        catalogue.initialise_catalogue_schema(conn)
    return catalogue.CatalogueService(database, root)


def weight(state, name='Mystery/No category/new.safetensors'):
    path = state.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'opaque payload: inventory must not inspect this')
    return path


def legacy(state, path, stable_id):
    with closing(state.connection()) as conn, conn:
        conn.execute('''INSERT INTO lora
            (file_path,filename,stable_id,base_model_code,category_code,model_family,rank,
             block_layout,last_modified,created_at,updated_at)
            VALUES(?,?,?,'FLX','PPL','legacy sentinel',23,'exact legacy layout',1.25,'old','old')''',
                     (str(path), Path(path).name, stable_id))


def snapshot(state):
    with closing(state.connection()) as conn:
        tables = sorted(row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                for table in tables}


def test_scope_is_configuration_not_presence_of_auxiliary_directories(state):
    excluded = [state.root / 'recipes' / 'old.safetensors',
                state.root / 'LORA_MANAGER_IMAGES' / 'old.safetensors',
                state.root / 'legacy.pt', state.root.parent / 'other.safetensors']
    excluded[2].write_bytes(b'existing legacy format')
    for index, path in enumerate(excluded):
        legacy(state, path, f'OLD-PPL-{index:03d}')
    legacy(state, state.root / 'gone.safetensors', 'OLD-PPL-099')
    receipt = state.refresh()
    assert receipt['counts']['out_of_scope'] == 4
    assert receipt['counts']['missing'] == 1
    assert [item['stable_id'] for item in state.search(presence='missing')['results']] == ['OLD-PPL-099']
    assert state.search()['total'] == 0


@pytest.mark.parametrize('history_table', catalogue.HISTORY_TABLES[1:])
def test_orphan_history_reserves_arbitrary_length_ids(state, history_table):
    with closing(state.connection()) as conn, conn:
        conn.execute(f'INSERT INTO {history_table} VALUES(?,?)', ('unk-unk-12345', 'unchanged exact history'))
    before = snapshot(state)
    weight(state)
    state.refresh()
    assert state.search()['results'][0]['stable_id'] == 'UNK-UNK-12346'
    after = snapshot(state)
    for table in catalogue.HISTORY_TABLES[1:]:
        assert after[table] == before[table]


def test_existing_rows_and_missing_blank_identity_are_not_rewritten(state):
    existing = weight(state)
    legacy(state, existing, 'EXACT-MANUAL-009')
    legacy(state, state.root / 'missing.safetensors', None)
    before = snapshot(state)['lora']
    state.refresh()
    assert snapshot(state)['lora'] == before
    current = state.search()['results'][0]
    assert current['stable_id'] == 'EXACT-MANUAL-009'
    assert current['metadata_provenance']['base_model_name'] == 'Mystery'
    assert current['model_family'] == 'legacy sentinel'
    assert current['architecture_verified'] is False
    assert current['legacy_analysis_unverified'] is True


def test_final_file_race_rolls_back_new_rows_assigned_ids_and_presence(state, monkeypatch):
    existing = weight(state, 'a.safetensors')
    legacy(state, existing, None)
    new = weight(state, 'b.safetensors')
    before = snapshot(state)
    original = catalogue.verify_inventory

    def changed(inventory, started):
        new.write_bytes(b'changed after SQL inserts')
        original(inventory, started)

    monkeypatch.setattr(catalogue, 'verify_inventory', changed)
    with pytest.raises(catalogue.CatalogueError) as error:
        state.refresh()
    assert error.value.code == 'inventory_changed'
    assert snapshot(state) == before


def test_late_directory_failure_preserves_previous_complete_receipt(state, monkeypatch):
    weight(state)
    state.refresh()
    before = snapshot(state)
    late = state.root / 'late-unreadable'
    late.mkdir()
    original = catalogue.os.scandir

    def denied(path):
        if Path(path) == late:
            raise PermissionError('synthetic late directory failure')
        return original(path)

    monkeypatch.setattr(catalogue.os, 'scandir', denied)
    with pytest.raises(catalogue.CatalogueError) as error:
        state.refresh()
    assert error.value.code == 'refresh_unavailable'
    assert snapshot(state) == before


def test_non_weight_entries_are_counted_before_directory_materialization(state, monkeypatch):
    for index in range(8):
        (state.root / f'{index}.txt').write_text('not a LoRA')
    monkeypatch.setattr(catalogue, 'MAX_ENTRIES', 3)
    original = catalogue.os.scandir
    consumed = []

    class CountedDirectory:
        def __init__(self, path):
            self.iterator = original(path)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.iterator.close()

        def __iter__(self):
            for entry in self.iterator:
                consumed.append(entry.name)
                yield entry

    monkeypatch.setattr(catalogue.os, 'scandir', CountedDirectory)
    before = snapshot(state)
    with pytest.raises(catalogue.CatalogueError) as error:
        state.refresh()
    assert error.value.code == 'inventory_budget'
    assert len(consumed) == 4
    assert snapshot(state) == before


@pytest.mark.parametrize('duplicate', ['path', 'identity'])
def test_ambiguous_legacy_identity_aborts_without_partial_updates(state, duplicate):
    path = weight(state, 'existing.safetensors')
    legacy(state, path, 'OLD-PPL-001')
    legacy(state, str(path).upper() if duplicate == 'path' else state.root / 'elsewhere.safetensors',
           'old-ppl-001' if duplicate == 'identity' else 'OTHER-PPL-001')
    weight(state, 'new.safetensors')
    before = snapshot(state)
    with pytest.raises(catalogue.CatalogueError) as error:
        state.refresh()
    assert error.value.code == ('duplicate_path' if duplicate == 'path' else 'duplicate_identity')
    assert snapshot(state) == before


def test_concurrent_history_commit_is_seen_and_second_refresh_is_locked(state, monkeypatch):
    weight(state)
    entered, release = threading.Event(), threading.Event()
    original = catalogue.discover

    def paused(root, started):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(root, started)

    monkeypatch.setattr(catalogue, 'discover', paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(state.refresh)
        try:
            assert entered.wait(5)
            other = catalogue.CatalogueService(state.database, state.root)
            with pytest.raises(catalogue.CatalogueError) as error:
                other.refresh()
            assert error.value.code == 'refresh_busy'
            with closing(state.connection()) as conn, conn:
                conn.execute("INSERT INTO lora_profile_versions VALUES('UNK-UNK-1000','concurrently saved')")
        finally:
            release.set()
        assert future.result(timeout=10)['status'] == 'complete'
    assert state.search()['results'][0]['stable_id'] == 'UNK-UNK-1001'
    assert snapshot(state)['lora_profile_versions'] == [('UNK-UNK-1000', 'concurrently saved')]


def test_linked_weight_and_directory_remain_out_of_scope(state):
    outside = state.root.parent / 'outside'
    outside.mkdir()
    source = outside / 'source.safetensors'
    source.write_bytes(b'outside payload')
    file_link, directory_link = state.root / 'linked.safetensors', state.root / 'linked-folder'
    try:
        file_link.symlink_to(source)
        directory_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f'This environment cannot create filesystem links: {exc}')
    legacy(state, file_link, 'OLD-PPL-001')
    legacy(state, directory_link / source.name, 'OLD-PPL-002')
    receipt = state.refresh()
    assert receipt['counts']['discovered'] == 0
    assert receipt['counts']['out_of_scope'] == 2
    assert source.read_bytes() == b'outside payload'


def test_refresh_does_not_open_weight_payloads(state, monkeypatch):
    weight(state)
    original = Path.open

    def no_payload(path, *args, **kwargs):
        assert path.suffix.casefold() != '.safetensors', 'Inventory attempted to open tensor payload'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', no_payload)
    assert state.refresh()['tensor_payload_read'] is False
