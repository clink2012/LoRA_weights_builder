import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
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

    def test_selected_preview_wal_state_restores_to_standard_layout(self):
        preview = self.project / ".local/preview-data/Database/lora_master.db"
        preview.parent.mkdir(parents=True)
        with closing(sqlite3.connect(preview)) as live_preview:
            live_preview.execute("PRAGMA journal_mode=WAL")
            live_preview.execute("CREATE TABLE owner_versions (value TEXT)")
            live_preview.execute("INSERT INTO owner_versions VALUES ('saved personal version')")
            live_preview.commit()
            self.assertTrue(Path(str(preview) + "-wal").is_file())
            receipt = snapshot(self.project, self.saved, preview.relative_to(self.project))
            self.assertEqual(receipt["source_database"], str(preview.resolve()))
            self.assertEqual(receipt["source_database_relative"], ".local/preview-data/Database/lora_master.db")
            self.assertEqual(receipt["database"]["table_counts"], {"owner_versions": 1})
            self.assertEqual(live_preview.execute("SELECT value FROM owner_versions").fetchone()[0], "saved personal version")
        restored = self.root / "restored-preview"
        restore(self.saved, restored)
        with closing(sqlite3.connect(restored / "Database/lora_master.db")) as conn:
            self.assertEqual(conn.execute("SELECT value FROM owner_versions").fetchone()[0], "saved personal version")
        self.assertFalse((restored / ".local/preview-data/Database/lora_master.db").exists())
        self.assertEqual(self.live.execute("SELECT value FROM profiles").fetchone()[0], "original")
        default_receipt = snapshot(self.project, self.root / "default-snapshot")
        self.assertEqual(default_receipt["source_database_relative"], "Database/lora_master.db")
        self.assertEqual(default_receipt["database"]["table_counts"], {"profiles": 1})

    def test_cli_explicit_database_uses_project_relative_path(self):
        preview = self.project / "preview.db"
        with closing(sqlite3.connect(preview)) as conn:
            conn.execute("CREATE TABLE selected (value TEXT)")
        run = subprocess.run([sys.executable, str(Path(__file__).with_name("local_state_backup.py")),
                              "snapshot", str(self.saved), "--project", str(self.project),
                              "--database", "preview.db"], capture_output=True, text=True, check=True)
        result = json.loads(run.stdout)
        self.assertEqual(result["source_database"], str(preview.resolve()))
        self.assertEqual(verify(self.saved)["database"]["table_counts"], {"selected": 0})

    def test_invalid_database_selection_never_creates_snapshot(self):
        outside = self.root / "outside.db"
        outside.write_bytes(self.db.read_bytes())
        for candidate in (self.project / "missing.db", self.project / "Database", outside,
                          Path("../outside.db")):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    snapshot(self.project, self.saved, candidate)
                self.assertFalse(self.saved.exists())

    def test_database_link_and_linked_parent_are_rejected(self):
        direct = self.project / "alias.db"
        directory = self.project / "alias-directory"
        try:
            direct.symlink_to(self.db)
            directory.symlink_to(self.db.parent, target_is_directory=True)
        except OSError as error:
            self.skipTest("Symlink creation unavailable: " + str(error))
        for candidate in (direct, directory / self.db.name):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, "links or junctions"):
                    snapshot(self.project, self.saved, candidate)
                self.assertFalse(self.saved.exists())

    def test_original_format_one_receipt_still_restores(self):
        snapshot(self.project, self.saved)
        receipt_path = self.saved / "manifest.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        del receipt["source_database"]
        del receipt["source_database_relative"]
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        restored = self.root / "legacy-receipt-restore"
        restore(self.saved, restored)
        with closing(sqlite3.connect(restored / "Database/lora_master.db")) as conn:
            self.assertEqual(conn.execute("SELECT value FROM profiles").fetchone()[0], "original")


if __name__ == "__main__":
    unittest.main()
