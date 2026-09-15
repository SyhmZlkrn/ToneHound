"""Boundary tests for the release exporter and conservative cache audit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load('export_source')
storage = load('storage_audit')


class ReleaseBoundaries(unittest.TestCase):
    def test_manifest_cannot_include_cache_or_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('.cache/account.txt', 'song.wav'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'test')
                with self.assertRaises(ValueError):
                    export.selected_files(root, {'include': [name], 'max_file_bytes': 100})

    def test_local_secret_is_detected_without_appearing_in_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = 'local-' + 'sensitive' * 4
            credential = root / '.cache/tone3000/tokens.json'
            credential.parent.mkdir(parents=True)
            credential.write_text(json.dumps({'access_token': token}))
            source = root / 'source.py'
            source.write_text('EXAMPLE = ' + repr(token))
            _, problems = export.inspect_files(root, [source])
            self.assertIn('source.py: contains a local credential', problems)
            self.assertNotIn(token, str(problems))

    def test_experiment_selection_preserves_reports_and_user_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('.cache/renders_djent/probe.npy', '.cache/reference_diagnostics/00/guitar.wav',
                     '.cache/reference_diagnostics/00/report.json', '.cache/native/media/user.wav',
                     '.cache/tone3000/tokens.json', '.cache/tone3000/profiles/amp.nam',
                     '.cache/toolchain/old.npy', 'assets/user_di/take.wav')
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'test')
            selected = [p.relative_to(root).as_posix() for p in storage.safe_experiment_files(root)]
            self.assertCountEqual(selected, names[:2])
            self.assertTrue(all((root / name).exists() for name in names))

    def test_text_export_preserves_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'engine/tonehound/sample.py'
            source.parent.mkdir(parents=True)
            source.write_text('answer = 42\n')
            files = export.selected_files(root, {'include': ['engine/tonehound/*.py'], 'max_file_bytes': 100})
            inventory, errors = export.inspect_files(root, files)
            self.assertFalse(errors)
            self.assertEqual(inventory[0]['path'], 'engine/tonehound/sample.py')
            self.assertEqual(len(inventory[0]['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
