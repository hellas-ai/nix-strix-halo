#!/usr/bin/env python3
"""CPU state-machine test of the bounded Engram status ABI."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import torch


source_root = Path(os.environ.get(
    "DS41_ENGRAM_SOURCE_ROOT", Path(__file__).resolve().parent / "src-work"
))
source = source_root / "sglang/srt/layers/engram_preflight_status.py"
tree = ast.parse(source.read_text(), filename=str(source))
classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
namespace = {"torch": torch, "inplace_all_reduce": lambda *_args, **_kwargs: None}
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[])), str(source), "exec"), namespace)
Status = namespace["EngramPreflightStatus"]
Failure = namespace["EngramPreflightError"]

status = Status(device="cpu", tp_rank=0, tp_size=1, group_name="test")
generation = status.begin()
assert generation == 1
assert status.local_layer_word(0).tolist() == [0]
status.finish(generation)
status.check_before_model(generation)
try:
    status.begin()
except RuntimeError:
    pass
else:
    raise AssertionError("active generation reused")
status.retire(generation)

generation = status.begin()
assert generation == 2 and status.device_status.tolist() == [[0, 0]]
status.set_local_failure(1, 4)
status.finish(generation)
try:
    status.check_before_model(generation)
except Failure as exc:
    assert exc.generation == 2 and exc.status == ((0, 4),)
else:
    raise AssertionError("failed generation accepted")
try:
    status.retire(1)
except RuntimeError:
    pass
else:
    raise AssertionError("stale generation retired")

assert not torch.cuda.is_initialized()
print("PASS: status reset, layer failure, host pre-model abort, one outstanding generation, stale retirement; CUDA uninitialized")
