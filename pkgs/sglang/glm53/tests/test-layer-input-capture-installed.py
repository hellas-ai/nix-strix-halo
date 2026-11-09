#!/usr/bin/env python3
"""Check installed diagnostic/source pins and the default-off import contract."""
import argparse
import ast
from array import array
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--plan', type=Path, required=True)
a = p.parse_args()
plan = json.loads(a.plan.read_bytes())
assert plan['schema'] == 'glm53-layer-input-capture-v1'
pins = plan['context_source_sha256'] | plan['applied_source_sha256']
pins |= plan['topology_guard_sources_sha256'] | plan['request_metadata_source_sha256']
assert len(pins) == 18
for name, expected in pins.items():
    assert hashlib.sha256((a.source/name).read_bytes()).hexdigest() == expected, name

original_import = builtins.__import__
def cpu_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton'):
        raise AssertionError('Disabled diagnostic imported a GPU package')
    return original_import(name, *args, **kwargs)

helper = a.source/'srt/managers/glm53_layer_input_capture.py'
with patch.dict(os.environ, {}, clear=True), patch('builtins.__import__', side_effect=cpu_import):
    spec = importlib.util.spec_from_file_location('installed_disabled_capture', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ENABLED is False and module._writer is None
    assert module.prepare(None, None, None) is None
    module.record(None, None, None, None)
    module.cache_metadata(None, None, None)
    assert module._writer is None
    # Req retains the CPU array supplied by the tokenizer. Only inspect its
    # source AST; manager imports would pull in GPU/distributed dependencies.
    tree = ast.parse((a.source/'srt/managers/schedule_batch.py').read_text())
    req = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Req')
    init = next(n for n in req.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    ids = next(arg for arg in init.args.args if arg.arg == 'origin_input_ids')
    assert ast.unparse(ids.annotation) == 'array[int]'
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == 'self.origin_input_ids')
    assert ast.unparse(assignment.value) == 'origin_input_ids'
    values = [0, 1, 154879]
    for ids in (array('i', values), array('q', values)):
        assert module.cpu_integer_sequence(ids) == values
        assert module.digest(module.cpu_integer_sequence(ids)) == module.digest(values)
        changed = array(ids.typecode, ids); changed[0] += 1
        assert module.digest(module.cpu_integer_sequence(changed)) != module.digest(values)
    for values in ([True], [1.5], array('d', [1.0]), object()):
        try:
            module.cpu_integer_sequence(values)
        except AssertionError:
            pass
        else:
            raise AssertionError('Noninteger CPU metadata accepted')
print('PASS: installed capture/context/topology/Req pins, CPU arrays and disabled import/calls')
