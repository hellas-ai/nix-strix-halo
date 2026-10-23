#!/usr/bin/env python3
"""CPU-only fatal Engram result-status ordering and generation test."""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


root = Path(os.environ.get(
    "DS41_ENGRAM_SOURCE_ROOT",
    Path(__file__).resolve().parent / "src-0072-work",
))


def selected_class(path, name, namespace, methods=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    if methods is not None:
        node.body = [item for item in node.body if getattr(item, "name", None) in methods]
        node.decorator_list = []
        node.name = "ResultCarrier"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, node], type_ignores=[]
    )), str(path), "exec"), namespace)


status_file = root / "sglang/srt/layers/engram_preflight_status.py"
ns = {"torch": torch, "inplace_all_reduce": lambda *_args, **_kwargs: None}
selected_class(status_file, "EngramPreflightError", ns)
selected_class(status_file, "EngramPreflightStatus", ns)
Status, Failure = ns["EngramPreflightStatus"], ns["EngramPreflightError"]

carrier_file = root / "sglang/srt/managers/utils.py"
selected_class(
    carrier_file, "GenerationBatchResult", ns,
    methods={"check_engram_status_after_copy"},
)
Carrier = ns["ResultCarrier"]


def result(status, generation, host):
    obj = Carrier()
    obj.copy_done = object()  # the production processor synchronizes it first
    obj.engram_status_owner = status
    obj.engram_status_generation = generation
    obj.engram_status_device = status.device_status
    obj.engram_status_host = host
    return obj


status = Status(device="cpu", tp_rank=0, tp_size=1, group_name="test")
gen = status.begin()
status.finish(gen)
good = result(status, gen, status.device_status.clone())
good.check_engram_status_after_copy()
assert status.active_generation is None and good.engram_status_owner is None
try:
    good.check_engram_status_after_copy()
except RuntimeError:
    raise AssertionError("clean absent status should be harmless")

gen = status.begin()
status.set_local_failure(1, 4)
status.finish(gen)
bad = result(status, gen, status.device_status.clone())
try:
    bad.check_engram_status_after_copy()
except Failure as exc:
    assert exc.status == ((0, 4),)
else:
    raise AssertionError("failed generation escaped the result gate")
assert status.active_generation == gen  # fatal: no later generation/retry
try:
    status.begin()
except RuntimeError:
    pass
else:
    raise AssertionError("failed generation was reused")

stale = Status(device="cpu", tp_rank=0, tp_size=1, group_name="test")
gen = stale.begin()
stale.finish(gen)
try:
    result(stale, gen + 1, stale.device_status.clone()).check_engram_status_after_copy()
except RuntimeError:
    pass
else:
    raise AssertionError("stale generation passed")
try:
    result(stale, gen, None).check_engram_status_after_copy()
except RuntimeError:
    pass
else:
    raise AssertionError("missing result copy passed")


class Store:
    def __init__(self, byte):
        self.byte = byte

    def lookup(self, ids):
        return SimpleNamespace(packed=np.full((ids.numel(), 4), self.byte, dtype=np.uint8))


class Hasher:
    def __init__(self):
        self.commits = 0

    def __call__(self, input_ids, _batch, *, defer_commit):
        assert defer_commit
        return torch.zeros((input_ids.numel(), 2, 24), dtype=torch.long), object()

    def commit_preflight(self, _pending, *, status_device):
        if not status_device.any():
            self.commits += 1


ns.update({"os": os, "logger": logging.getLogger(__name__), "ROW_BYTES": 4})
selected_class(root / "sglang/srt/layers/engram_native_preflight.py", "EngramNativePreflight", ns)
Native = ns["EngramNativePreflight"]
hasher = Hasher()
model = SimpleNamespace(
    engram_hasher=hasher,
    layers=[SimpleNamespace(engram=SimpleNamespace(
        layer_hash_index=i, embed=SimpleNamespace(file_store=Store(41 + i))
    )) for i in range(2)],
)
mode = SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False, is_extend=lambda: False)
fb = SimpleNamespace(input_ids=torch.tensor([7]), forward_mode=mode, engram_packed_rows=None)
os.environ["SGLANG_ENGRAM_PREFLIGHT_FAULT"] = "0:1:1"
native = Native(model=model, device="cpu", tp_rank=0, tp_size=1, group_name="test", result_gate=True)
owner, gen = native.prepare(fb)
assert fb.engram_packed_rows.shape == (1, 2, 24, 4)
assert hasher.commits == 0 and owner.active_generation == gen
try:
    result(owner, gen, owner.device_status.clone()).check_engram_status_after_copy()
except Failure:
    pass
else:
    raise AssertionError("native fatal-only fault escaped the result gate")
del os.environ["SGLANG_ENGRAM_PREFLIGHT_FAULT"]

processor = (root / "sglang/srt/managers/scheduler_components/batch_result_processor.py").read_text()
for name in ("process_batch_result_prefill", "process_batch_result_decode"):
    body = processor.split(f"def {name}(", 1)[1].split("\n    def ", 1)[0]
    gate = body.index("result.check_engram_status_after_copy()")
    for publication in (
        "free_group_begin()", "snapshot_auxiliary_output_starts(",
        "_stream_output_generation(", "maybe_cache_unfinished_req(",
    ):
        if publication in body:
            assert gate < body.index(publication), (name, publication)

assert not torch.cuda.is_initialized()
print("PASS: fatal result status, stale/missing copy, no retry, pre-publication ordering; CUDA uninitialized")
