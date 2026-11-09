#!/usr/bin/env python3
"""CPU boundary test of a built launcher; replaces only its final node exec."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--launcher', type=Path, required=True)
a = p.parse_args()
script = a.launcher.read_text()
match = re.search(r'exec bash (/nix/store/[^\s]+-glm53-node\.sh) "\$rank"', script)
if not match:
    raise SystemExit('launcher node boundary not found')
node = Path(match.group(1))
with tempfile.TemporaryDirectory(prefix='glm53-launcher-test-') as td:
    root = Path(td)
    stub = root / 'node-stub'
    stub.write_text('''#!/bin/sh
python3 - "$@" <<'INNER'
import json, os, sys
with open(os.environ['RECEIPT'], 'w') as f:
    json.dump({'cache':os.environ.get('TRITON_CACHE_DIR'), 'runtime':os.environ.get('GLM_RUNTIME_ID'),
               'sglang':os.environ.get('GLM_SGLANG_BIN'), 'args':sys.argv[1:]}, f)
INNER
''')
    launcher = root / 'launcher'
    launcher.write_text(script.replace(str(node), str(stub)))
    receipt = root / 'receipt.json'
    env = {**os.environ, 'XDG_CACHE_HOME':str(root/'xdg'), 'RECEIPT':str(receipt), 'GLM_PM4':'0'}
    env.pop('TRITON_CACHE_DIR', None)
    env.pop('GLM_AOT', None)
    def launch(extra):
        e = {**env, **extra}
        subprocess.run(['bash',str(launcher),'0'], env=e, check=True)
        return json.loads(receipt.read_text())
    default = launch({})
    chosen = Path(default['cache'])
    assert chosen.is_dir() and len(list(chosen.iterdir())) == 13
    assert default['runtime'] in default['sglang']
    sentinel = chosen/'later-jit'
    sentinel.write_text('unchanged')
    assert launch({})['cache'] == str(chosen) and sentinel.read_text() == 'unchanged'
    explicit = root/'explicit-cache'
    assert launch({'TRITON_CACHE_DIR':str(explicit)})['cache'] == str(explicit)
    assert not explicit.exists()
    assert launch({'GLM_AOT':'0'})['cache'] is None
    assert default['args'][-4:] == ['1','2','4','8']

    # Exercise the unchanged node boundary independently, through a fake
    # verified checkpoint and fake sglang command; TCP avoids network discovery.
    model = root/'model'
    (model/'_staging').mkdir(parents=True)
    (model/'_staging'/'verified.json').write_text(json.dumps({'revision':'test','bytes':0,'files':[]}))
    server = root/'server'
    server.write_text('''#!/bin/sh
python3 - "$@" <<'INNER'
import json, os, sys
with open(os.environ['RECEIPT'], 'w') as f:
    json.dump({'cache':os.environ['TRITON_CACHE_DIR'], 'args':sys.argv[1:]}, f)
INNER
''')
    server.chmod(0o755)
    node_env = {**env, 'GLM_PYTHON':'python3', 'GLM_SGLANG_BIN':str(server),
                'GLM_TRANSPORT':'tcp', 'GLM_MODEL_PATH':str(model), 'GLM_MODEL_REVISION':'test',
                'GLM_MODEL_BYTES':'0', 'GLM_MODEL_FILES':'0', 'GLM_RUNTIME_ID':default['runtime']}
    subprocess.run(['bash', str(node), '0'], env=node_env, check=True)
    fallback = json.loads(receipt.read_text())['cache']
    assert fallback.endswith('/'+default['runtime']+'/triton')
    subprocess.run(['bash', str(node), '0'], env={**node_env,'TRITON_CACHE_DIR':str(explicit)}, check=True)
    assert json.loads(receipt.read_text())['cache'] == str(explicit)
print('PASS: AOT stage, idempotence, explicit bypass, GLM_AOT=0 and node fallback')
