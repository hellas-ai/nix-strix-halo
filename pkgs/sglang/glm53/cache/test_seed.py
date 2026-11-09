import importlib.util
import errno
import os
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('seed', Path(__file__).with_name('seed.py'))
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


class SeedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ident = {'runtime': '/nix/store/abc-sglang', 'target': 'gfx1151',
                      'configuration': {'graphs': [1, 2, 4, 8]}}
        self.cache = self.root / 'cache'
        self.d = self.cache / ('A' * 52)
        self.d.mkdir(parents=True)
        (self.d / 'kernel.hsaco').write_bytes(b'compiled')
        (self.d / 'kernel.json').write_text('{}')
        self.grp = self.d / '__grp__kernel.json'
        self.grp.write_text(json.dumps({'child_paths': {
            'kernel.hsaco': str(self.d / 'kernel.hsaco'), 'kernel.json': str(self.d / 'kernel.json')}}))
        self.seed = self.root / 'seed'

    def export(self):
        seed.export(self.cache, self.seed, self.ident)

    def test_unsupported_publish_rejects_existing_and_symlink(self):
        for code in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            with self.subTest(code=code), mock.patch.object(seed, '_rename_noreplace', side_effect=OSError(code, 'unsupported')):
                stage = self.root / f'stage-{code}'
                stage.mkdir()
                dest = self.root / f'dest-{code}'
                dest.mkdir()
                with self.assertRaises(FileExistsError): seed.publish(stage, dest)
                dest.rmdir()
                dest.symlink_to(self.root / 'missing')
                with self.assertRaises(FileExistsError): seed.publish(stage, dest)
                dest.unlink()
                seed.publish(stage, dest)
                self.assertTrue(dest.is_dir())
        stage = self.root / 'permission-stage'
        stage.mkdir()
        with mock.patch.object(seed, '_rename_noreplace', side_effect=OSError(errno.EPERM, 'denied')):
            with self.assertRaises(PermissionError): seed.publish(stage, self.root / 'permission-dest')

    def test_unsupported_concurrent_subprocess_publish(self):
        script = Path(__file__).with_name('seed.py').resolve()
        dest = self.root / 'winner'
        runner = """import errno,importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('seed',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def unsupported(*args): raise OSError(errno.EINVAL,'unsupported')
m._rename_noreplace=unsupported
try: m.publish(Path(sys.argv[2]),Path(sys.argv[3]))
except FileExistsError: sys.exit(17)
"""
        children = []
        for i in range(8):
            stage = self.root / f'candidate-{i}'
            stage.mkdir()
            (stage / 'marker').write_text(str(i))
            children.append(subprocess.Popen([sys.executable, '-c', runner, str(script), str(stage), str(dest)],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        statuses = []
        for child in children:
            _, err = child.communicate(timeout=10)
            statuses.append(child.returncode)
            self.assertIn(child.returncode, (0, 17), err.decode())
        self.assertEqual(statuses.count(0), 1)
        self.assertEqual(statuses.count(17), 7)
        self.assertIn((dest / 'marker').read_text(), {str(i) for i in range(8)})

    def test_cli_roundtrip(self):
        identity_path = self.root / 'identity.json'
        identity_path.write_text(json.dumps(self.ident))
        script = Path(__file__).with_name('seed.py')
        def invoke(command, *args):
            return subprocess.run([sys.executable, str(script), command, '--identity', str(identity_path),
                                   '--seed', str(self.seed), *map(str, args)],
                                  check=True, capture_output=True, text=True)
        invoke('export', '--cache', self.cache)
        invoke('verify')
        dest = self.root / 'cli-dest'
        invoke('install', '--dest', dest)
        paths = json.loads((dest / self.d.name / self.grp.name).read_text())['child_paths']
        self.assertEqual(Path(paths['kernel.hsaco']).read_bytes(), b'compiled')

    def test_relocation_and_verify(self):
        self.export()
        seed.verify(self.seed, self.ident)
        dest = self.root / 'new-cache'
        seed.install(self.seed, dest, self.ident)
        values = json.loads((dest / self.d.name / self.grp.name).read_text())['child_paths']
        self.assertEqual(values['kernel.hsaco'], str(dest / self.d.name / 'kernel.hsaco'))
        self.assertEqual(Path(values['kernel.hsaco']).read_bytes(), b'compiled')
        with self.assertRaises((ValueError, OSError)):
            seed.install(self.seed, dest, self.ident)

    def test_identity_and_corruption_rejected_without_destination(self):
        self.export()
        with self.assertRaises(ValueError):
            seed.verify(self.seed, {**self.ident, 'target': 'gfx000'})
        (self.seed / 'payload' / self.d.name / 'kernel.hsaco').write_bytes(b'corrupt')
        dest = self.root / 'new-cache'
        with self.assertRaises(ValueError):
            seed.install(self.seed, dest, self.ident)
        self.assertFalse(dest.exists())

    def test_foreign_group_and_symlink_rejected(self):
        self.grp.write_text(json.dumps({'child_paths': {'kernel.hsaco': '/tmp/kernel.hsaco'}}))
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.seed.exists())
        self.grp.write_text(json.dumps({'child_paths': {'kernel.hsaco': str(self.d / 'kernel.hsaco')}}))
        (self.d / 'kernel.hsaco').unlink()
        (self.d / 'kernel.hsaco').symlink_to(self.root / 'foreign')
        with self.assertRaises(ValueError):
            self.export()

    def test_unlisted_missing_and_traversal_rejected(self):
        self.export()
        extra = self.seed / 'payload' / self.d.name / 'extra'
        extra.write_bytes(b'x')
        with self.assertRaises(ValueError): seed.verify(self.seed, self.ident)
        extra.unlink()
        (self.seed / 'payload' / self.d.name / 'kernel.hsaco').unlink()
        with self.assertRaises(ValueError): seed.verify(self.seed, self.ident)
        with self.assertRaises(ValueError): seed.group(b'{"child_paths":{"../x":"../x"}}', self.d, source=False)

    def test_incomplete_cache_ignored_and_no_overwrite(self):
        (self.cache / ('B' * 52)).mkdir()
        self.export()
        self.assertFalse((self.seed / 'payload' / ('B' * 52)).exists())
        with self.assertRaises(ValueError): self.export()


if __name__ == '__main__':
    unittest.main()
