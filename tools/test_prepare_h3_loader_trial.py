import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_h3_loader_trial as trial


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = [self.root / name for name in ('source.py', 'LICENSE', 'NOTICE')]
        for index, path in enumerate(self.inputs): path.write_text(f'harmless fixture {index}')
        pins = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in zip(('h3_block_loader.py', 'LICENSE', 'NOTICE'), self.inputs)}
        self.patcher = patch.object(trial, 'PINS', pins)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.output = self.root / '.local/h3-loader-trial/lora_builder_h3_loader_trial'

    def test_stage_is_idempotent_and_only_declared_files_are_present(self):
        first = trial.prepare(self.root, *self.inputs)
        second = trial.prepare(self.root, *self.inputs)
        self.assertEqual(first, second)
        self.assertFalse(first['install_performed'])
        self.assertFalse(first['export_verified'])
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'h3_block_loader.py', '__init__.py', 'LICENSE', 'NOTICE', 'manifest.json'})
        self.assertEqual(json.loads((self.output / 'manifest.json').read_text()), first)
        for name, digest in first['files'].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_changed_source_stages_nothing(self):
        self.inputs[0].write_text('different source')
        with self.assertRaisesRegex(ValueError, 'changed'): trial.prepare(self.root, *self.inputs)
        self.assertFalse(self.output.exists())

    def test_changed_license_stages_nothing(self):
        self.inputs[1].write_text('different license')
        with self.assertRaisesRegex(ValueError, 'changed'): trial.prepare(self.root, *self.inputs)
        self.assertFalse(self.output.exists())

    def test_existing_divergent_staging_is_preserved(self):
        trial.prepare(self.root, *self.inputs)
        (self.output / '__init__.py').write_text('personal change')
        with self.assertRaisesRegex(ValueError, 'overwritten'): trial.prepare(self.root, *self.inputs)
        self.assertEqual((self.output / '__init__.py').read_text(), 'personal change')

    def test_extra_files_are_preserved_and_block_restaging(self):
        trial.prepare(self.root, *self.inputs)
        (self.output / 'extra').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'overwritten'): trial.prepare(self.root, *self.inputs)
        self.assertEqual((self.output / 'extra').read_text(), 'keep')

    def test_missing_or_unbounded_source_stages_nothing(self):
        self.inputs[0].unlink()
        with self.assertRaisesRegex(ValueError, 'absent'): trial.prepare(self.root, *self.inputs)
        self.inputs[0].write_bytes(b'x' * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, 'unbounded'): trial.prepare(self.root, *self.inputs)
        self.assertFalse(self.output.exists())


if __name__ == '__main__': unittest.main()
