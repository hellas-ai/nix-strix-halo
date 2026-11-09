import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SEED_TOOL = HERE.parent / 'cache' / 'seed.py'
spec = importlib.util.spec_from_file_location('seed', SEED_TOOL)
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


class StageTest(unittest.TestCase):
    def test_idempotence_and_runtime_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ident = {'runtime':'/nix/store/abc-runtime', 'target':'gfx1151', 'configuration':{'x':1}}
            identity = root / 'identity.json'
            identity.write_text(json.dumps(ident))
            original = root / 'original' / ('A' * 52)
            original.mkdir(parents=True)
            (original / 'kernel.hsaco').write_bytes(b'valid')
            (original / '__grp__kernel.json').write_text(json.dumps({'child_paths':{'kernel.hsaco':str(original / 'kernel.hsaco')}}))
            seed.export(original.parent, root / 'seed', ident)
            dest = root / 'parent' / 'versioned'
            cmd = [sys.executable, str(HERE / 'stage.py'), '--seed-tool', str(SEED_TOOL), '--identity', str(identity),
                   '--seed', str(root / 'seed'), '--runtime', ident['runtime'], '--dest', str(dest)]
            subprocess.run(cmd, check=True)
            extra = dest / 'jit-added'
            extra.write_text('preserve')
            subprocess.run(cmd, check=True)
            self.assertEqual(extra.read_text(), 'preserve')
            bad = cmd.copy()
            bad[bad.index('--runtime') + 1] = '/nix/store/other-runtime'
            self.assertNotEqual(subprocess.run(bad, capture_output=True).returncode, 0)
            self.assertEqual(extra.read_text(), 'preserve')
            payload = root / 'seed' / 'payload' / original.name / 'kernel.hsaco'
            payload.write_bytes(b'corrupt')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(extra.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
