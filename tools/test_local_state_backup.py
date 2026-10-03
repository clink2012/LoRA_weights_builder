import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from local_state_backup import restore, snapshot, verify


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        (self.project / "Database").mkdir(parents=True)
        self.db = self.project / "Database/lora_master.db"
        self.live = sqlite3.connect(self.db)
        self.addCleanup(self.live.close)
        self.live.execute("PRAGMA journal_mode=WAL")
        self.live.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, value TEXT)")
        self.live.execute("INSERT INTO profiles VALUES (1, 'original')")
        self.live.commit()
        self.saved = self.root / "saved"

    def test_wal_snapshot_restores_values_and_local_files_without_overwriting(self):
        profile = self.project / "Database/backend/profiles/demo.json"
        profile.parent.mkdir(parents=True)
        profile.write_text('{"weights":[0.1,0.2]}', encoding="utf-8")
        snapshot(self.project, self.saved)
        self.live.execute("UPDATE profiles SET value='later'")
        self.live.commit()
        output = self.root / "restored"
        restore(self.saved, output)
        with closing(sqlite3.connect(output / "Database/lora_master.db")) as restored:
            self.assertEqual(restored.execute("SELECT value FROM profiles").fetchone()[0], "original")
        self.assertEqual((output / profile.relative_to(self.project)).read_bytes(), profile.read_bytes())
        self.assertEqual(self.live.execute("SELECT value FROM profiles").fetchone()[0], "later")
        with self.assertRaises(FileExistsError):
            restore(self.saved, self.project)

    def test_corruption_is_rejected_before_restore_directory_created(self):
        snapshot(self.project, self.saved)
        (self.saved / "Database/lora_master.db").write_bytes(b"damaged")
        target = self.root / "must-not-exist"
        with self.assertRaises(ValueError):
            restore(self.saved, target)
        self.assertFalse(target.exists())

    def test_manifest_escape_is_rejected(self):
        snapshot(self.project, self.saved)
        path = self.saved / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["../outside.txt"] = "irrelevant"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify(self.saved)

    def test_existing_snapshot_is_never_replaced(self):
        snapshot(self.project, self.saved)
        before = (self.saved / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            snapshot(self.project, self.saved)
        self.assertEqual((self.saved / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
