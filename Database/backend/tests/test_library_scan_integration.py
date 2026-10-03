"""The API must bind scanning to the launcher's selected database, not import state."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lora_api_server as api


def test_scanner_factory_uses_current_database_and_root_without_opening_them(tmp_path, monkeypatch):
    database = tmp_path / 'selected.sqlite'
    root = tmp_path / 'models'
    monkeypatch.setattr(api, 'DB_PATH', database)
    monkeypatch.setenv('LORA_ROOT', str(root))
    scanner = api.create_selected_library_scanner()
    assert scanner.catalogue.database == database.resolve()
    assert scanner.catalogue.root == root.resolve()
    assert scanner.status()['status'] == 'idle'
    assert not database.exists()
    assert not root.exists()
