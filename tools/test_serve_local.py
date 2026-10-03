"""Offline launcher checks: temporary fixtures only, no listener or real DB."""
from contextlib import asynccontextmanager, closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_local as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ui = self.root / 'Database/UI'
        (self.ui / 'src').mkdir(parents=True)
        (self.ui / 'dist/assets').mkdir(parents=True)
        (self.ui / 'src/main.jsx').write_text('source', encoding='utf-8')
        (self.ui / 'package-lock.json').write_text('{}', encoding='utf-8')
        (self.ui / 'package.json').write_text('{"scripts":{"build":"vite build"}}', encoding='utf-8')
        (self.ui / 'index.html').write_text('source shell', encoding='utf-8')
        (self.ui / 'dist/index.html').write_text('<h1>Local LoRA</h1>', encoding='utf-8')
        (self.ui / 'dist/assets/main.js').write_text('built', encoding='utf-8')
        self.database = self.root / 'Database/lora_master.db'
        with closing(sqlite3.connect(self.database)) as conn:
            conn.executescript('CREATE TABLE lora (id INTEGER); CREATE TABLE lora_block_weights (id INTEGER); INSERT INTO lora VALUES (7);')
            conn.commit()
        launcher.record_build(self.root)

    def test_build_content_not_git_revision_controls_staleness(self):
        before = launcher.verify_build(self.root)
        with patch.object(launcher, 'source_identity', return_value={'git_revision': 'different'}):
            self.assertEqual(launcher.verify_build(self.root), before)
        (self.ui / 'src/main.jsx').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(launcher.LocalLaunchError, 'stale'):
            launcher.verify_build(self.root)

    def test_output_tamper_missing_files_and_traversal_rejected(self):
        built = self.ui / 'dist/assets/main.js'
        built.write_text('tampered', encoding='utf-8')
        with self.assertRaises(launcher.LocalLaunchError):
            launcher.verify_build(self.root)
        launcher.record_build(self.root)
        receipt_path = self.root / '.local/runtime/ui-build.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['files']['../../../outside'] = 'bad'
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaises(launcher.LocalLaunchError):
            launcher.verify_build(self.root)

    def test_build_uses_project_vite_and_detects_midbuild_changes(self):
        vite = self.ui / 'node_modules/vite/bin/vite.js'
        vite.parent.mkdir(parents=True)
        vite.write_text('fixture')
        def build_changed(args, **kwargs):
            self.assertEqual(args, ['node-existing', str(vite), 'build'])
            self.assertEqual(kwargs['env']['VITE_API_BASE'], '/api')
            (self.ui / 'src/main.jsx').write_text('changed during build')
        with patch.object(launcher.shutil, 'which', return_value='node-existing'), patch.object(launcher.subprocess, 'run', side_effect=build_changed):
            with self.assertRaisesRegex(launcher.LocalLaunchError, 'changed while'):
                launcher.build_ui(self.root)

    def test_existing_healthy_catalogue_required_no_empty_database_created(self):
        missing = self.root / 'absent.db'
        with self.assertRaises(launcher.LocalLaunchError):
            launcher.validate_database(missing, self.root)
        self.assertFalse(missing.exists())
        wrong = self.root / 'wrong.db'
        with closing(sqlite3.connect(wrong)) as conn:
            conn.execute('CREATE TABLE unrelated (id INTEGER)')
        with self.assertRaisesRegex(launcher.LocalLaunchError, 'catalogue'):
            launcher.validate_database(wrong, self.root)
        self.assertEqual(launcher.validate_database(self.database, self.root)[1], 'main')

    def test_only_main_database_gets_verified_prestart_snapshot(self):
        copied = self.root / 'copied.db'
        copied.write_bytes(self.database.read_bytes())
        with patch('local_state_backup.snapshot') as snapshot, patch('local_state_backup.verify') as verify:
            self.assertIsNone(launcher.snapshot_main_database(copied, self.root))
            snapshot.assert_not_called()
            destination = launcher.snapshot_main_database(self.database, self.root)
            snapshot.assert_called_once()
            verify.assert_called_once_with(Path(destination))

    def test_port_conflict_prevents_snapshot_and_application_startup(self):
        with patch.object(launcher, 'verify_build'), patch.object(launcher, 'validate_database', return_value=(self.database, 'main')), patch.object(launcher.socket, 'socket') as socket, patch.object(launcher, 'snapshot_main_database') as backup, patch.object(launcher, 'create_local_app') as create:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError('busy')
            with self.assertRaisesRegex(launcher.LocalLaunchError, 'already in use'):
                launcher.serve(self.database, 5187)
            backup.assert_not_called()
            create.assert_not_called()

    def test_built_ui_api_startup_identity_and_local_origin(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        events = []
        @asynccontextmanager
        async def lifespan(app):
            events.append('startup')
            yield
            events.append('shutdown')
        backend_app = FastAPI(lifespan=lifespan)
        @backend_app.get('/api/probe')
        def probe():
            return {'database': str(fake.DB_PATH)}
        fake = SimpleNamespace(app=backend_app)
        with patch.object(launcher.importlib, 'import_module', return_value=fake), patch.dict(os.environ), patch.object(sys, 'path', list(sys.path)):
            app = launcher.create_local_app(self.database, project=self.root, run_id='receipt-id')
            with TestClient(app, base_url='http://127.0.0.1:5187') as client:
                self.assertEqual(events, ['startup'])
                self.assertIn('Local LoRA', client.get('/').text)
                self.assertEqual(client.get('/api/probe').json()['database'], str(self.database.resolve()))
                status = client.get('/local-app/status').json()
                self.assertEqual(status['run_id'], 'receipt-id')
                self.assertEqual(status['database_kind'], 'main')
                self.assertEqual(client.get('/docs').status_code, 404)
                self.assertEqual(client.get('/', headers={'host': 'other-machine:5187'}).status_code, 403)
                self.assertEqual(client.get('/', headers={'origin': 'https://outside.example'}).status_code, 403)
                self.assertEqual(client.get('/', headers={'origin': 'http://localhost:5187'}).status_code, 200)
            self.assertEqual(events, ['startup', 'shutdown'])


if __name__ == '__main__':
    unittest.main()
