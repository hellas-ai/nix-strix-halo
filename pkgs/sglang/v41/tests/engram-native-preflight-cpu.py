#!/usr/bin/env python3
"""Exercise the all-rank pre-model gate with local stores and injected faults."""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(os.environ.get(
    "DS41_ENGRAM_SOURCE_ROOT", Path(__file__).resolve().parent / "src-work"
)) / "sglang/srt/layers"


def classes(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(path), "exec"), namespace)


namespace = {"torch": torch, "inplace_all_reduce": lambda *_a, **_k: None,
             "ROW_BYTES": 4, "os": os, "logger": logging.getLogger(__name__)}
classes(ROOT / "engram_preflight_status.py", ("EngramPreflightStatus", "EngramPreflightError"), namespace)
classes(ROOT / "engram_native_preflight.py", ("EngramNativePreflight",), namespace)
Preflight = namespace["EngramNativePreflight"]
Failure = namespace["EngramPreflightError"]


class Store:
    def __init__(self, byte, fail=False):
        self.byte = byte
        self.fail = fail

    def lookup(self, ids):
        if self.fail:
            raise OSError("injected local shard read failure")
        return SimpleNamespace(packed=np.full((ids.numel(), 4), self.byte, dtype=np.uint8))


class Hasher:
    def __init__(self):
        self.commits = []

    def __call__(self, input_ids, forward_batch, *, defer_commit):
        assert defer_commit
        rows = input_ids.numel()
        ids = torch.arange(rows * 48).reshape(rows, 2, 24)
        return ids, (forward_batch, ids)

    def commit_preflight(self, pending, *, status_device=None):
        assert status_device is None  # blocking 0071 comparator
        self.commits.append(pending)


def make_model(*, fail_layer=None):
    h = Hasher()
    layers = [SimpleNamespace(engram=SimpleNamespace(layer_hash_index=i,
                embed=SimpleNamespace(file_store=Store(i + 41, fail_layer == i))))
              for i in range(2)]
    return SimpleNamespace(engram_hasher=h, layers=layers), h


class Mode:
    def is_decode(self): return True
    def is_target_verify(self): return False
    def is_extend(self): return False


def batch():
    return SimpleNamespace(input_ids=torch.tensor([17, 18]), forward_mode=Mode(),
                           engram_packed_rows=None)


model, h = make_model()
preflight = Preflight(model=model, device="cpu", tp_rank=0, tp_size=1, group_name="test")
fb = batch()
preflight.prepare(fb)
assert len(h.commits) == 1 and fb.engram_packed_rows.shape == (2, 2, 24, 4)
assert fb.engram_packed_rows[:, 0].unique().tolist() == [41]
assert fb.engram_packed_rows[:, 1].unique().tolist() == [42]
assert preflight.status.active_generation is None

model, h = make_model(fail_layer=1)
preflight = Preflight(model=model, device="cpu", tp_rank=0, tp_size=1, group_name="test")
fb = batch()
try:
    preflight.prepare(fb)
except Failure as exc:
    assert exc.status == ((0, 2),)
else:
    raise AssertionError("file failure reached the model")
assert h.commits == [] and fb.engram_packed_rows is None

os.environ["SGLANG_ENGRAM_PREFLIGHT_FAULT"] = "0:0:1"
model, h = make_model()
preflight = Preflight(model=model, device="cpu", tp_rank=0, tp_size=1, group_name="test")
fb = batch()
try:
    preflight.prepare(fb)
except Failure as exc:
    assert exc.status == ((5, 0),)
else:
    raise AssertionError("injected status reached the model")
assert h.commits == [] and fb.engram_packed_rows is None
del os.environ["SGLANG_ENGRAM_PREFLIGHT_FAULT"]

assert not torch.cuda.is_initialized()
print("PASS: native row bytes, before-model status gate, file fault and deterministic fault leave history and model input untouched; CUDA uninitialized")
