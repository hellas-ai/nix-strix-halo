#!/usr/bin/env python3
"""CPU orchestration check for opt-in mapped gather and fatal result gate.

The real HIP row/file/graph semantics are tested separately on one assigned
GPU. This verifies per-layer routine failures join the fixed generation gate
without early history retirement or row publication.
"""

import ast
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(os.environ.get(
    "DS41_ENGRAM_SOURCE_ROOT",
    Path(__file__).resolve().parent / "src-0073-work",
)) / "sglang/srt/layers"


def load_class(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    assert len(selected) == 1
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])),
                 str(path), "exec"), namespace)


class Mapped:
    def __init__(self, store, *, capacity, workers):
        assert capacity == 128 and workers == 2
        if store.init_fail:
            raise OSError("injected mapped registration failure")
        self.store = store

    def gather_into(self, ids, rows, status_word, *, wait_ms):
        assert ids.shape == (2, 24) and ids.dtype == torch.int64
        assert rows.shape == (2, 24, 4)
        assert rows.stride() == (2 * 24 * 4, 4, 1)
        assert wait_ms == 2_000
        if self.store.read_fail:
            status_word.fill_(4)
            return
        rows.fill_(self.store.layer + 41)


for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
    sys.modules.setdefault(name, types.ModuleType(name))
module = types.ModuleType("sglang.srt.layers.engram_mapped_cache")
module.EngramMappedCache = Mapped
sys.modules[module.__name__] = module

namespace = {
    "torch": torch, "os": os, "ROW_BYTES": 4,
    "inplace_all_reduce": lambda *_a, **_k: None,
    "logger": logging.getLogger(__name__),
}
load_class(ROOT / "engram_preflight_status.py", "EngramPreflightStatus", namespace)
load_class(ROOT / "engram_preflight_status.py", "EngramPreflightError", namespace)
load_class(Path(os.environ.get(
    "DS41_ENGRAM_PREFLIGHT_FILE", ROOT / "engram_native_preflight.py"
)), "EngramNativePreflight", namespace)
Preflight = namespace["EngramNativePreflight"]
Failure = namespace["EngramPreflightError"]


class Hasher:
    def __init__(self): self.commits = []

    def __call__(self, input_ids, _batch, *, defer_commit):
        assert defer_commit
        return torch.arange(input_ids.numel() * 48, dtype=torch.int64).reshape(-1, 2, 24), object()

    def commit_preflight(self, pending, *, status_device):
        assert pending is not None and status_device is not None
        self.commits.append(status_device)


class Mode:
    def is_decode(self): return True
    def is_target_verify(self): return False
    def is_extend(self): return False


def setup(*, fail_init=None, fail_read=None, rank=0, size=1):
    hasher = Hasher()
    layers = [SimpleNamespace(engram=SimpleNamespace(
        layer_hash_index=i, embed=SimpleNamespace(file_store=SimpleNamespace(
            layer=i, init_fail=i == fail_init, read_fail=i == fail_read
        )))) for i in range(2)]
    model = SimpleNamespace(engram_hasher=hasher, layers=layers)
    owner = Preflight(model=model, device="cpu", tp_rank=rank, tp_size=size,
                      group_name="test", result_gate=True)
    batch = SimpleNamespace(input_ids=torch.tensor([17, 18]),
                            forward_mode=Mode(), engram_packed_rows=None)
    return owner, hasher, batch


os.environ["SGLANG_ENGRAM_MAPPED_CACHE"] = "1"
os.environ["SGLANG_ENGRAM_MAPPED_CAPACITY"] = "128"
os.environ["SGLANG_ENGRAM_MAPPED_WORKERS"] = "2"
os.environ["SGLANG_ENGRAM_MAPPED_WAIT_MS"] = "2000"
os.environ.pop("SGLANG_ENGRAM_PREFLIGHT_FAULT", None)

owner, hasher, batch = setup()
status, generation = owner.prepare(batch)
assert generation == 1 and status.active_generation == 1
assert len(hasher.commits) == 1
assert batch.engram_packed_rows.shape == (2, 2, 24, 4)
assert batch.engram_packed_rows[:, 0].unique().tolist() == [41]
assert batch.engram_packed_rows[:, 1].unique().tolist() == [42]
status.check_after_result_copy(generation, status.device_status.clone())
assert status.active_generation is None

for kind, fail_layer, code in (("registration", 0, 6), ("file", 1, 4)):
    owner, hasher, batch = setup(
        fail_init=fail_layer if kind == "registration" else None,
        fail_read=fail_layer if kind == "file" else None,
    )
    status, generation = owner.prepare(batch)
    assert status.active_generation == generation
    assert int(status.device_status[0, fail_layer]) == code
    assert not bool(batch.engram_packed_rows[:, fail_layer].any())
    try:
        status.check_after_result_copy(generation, status.device_status.clone())
    except Failure as exc:
        assert exc.status[0][fail_layer] == code
    else:
        raise AssertionError(f"{kind} fault escaped the result gate")
    assert status.active_generation == generation, "failed generation was reused"

# A fault in the mapped branch must enter the fixed rank/layer matrix on the
# requested generation, after both gathers and before finish. The no-op TP
# reduction stub lets this CPU test inspect a two-rank-shaped status; live TP
# collective behavior remains a separate serving gate.
os.environ["SGLANG_ENGRAM_PREFLIGHT_FAULT"] = "1:0:2"
owner, hasher, first = setup(rank=1, size=2)
status, generation = owner.prepare(first)
assert generation == 1 and status.finished
assert status.device_status.tolist() == [[0, 0], [0, 0]]
status.check_after_result_copy(generation, status.device_status.clone())
second = SimpleNamespace(input_ids=torch.tensor([19, 20]),
                         forward_mode=Mode(), engram_packed_rows=None)
status, generation = owner.prepare(second)
assert generation == 2 and status.finished
assert status.device_status.tolist() == [[0, 0], [5, 0]]
assert len(hasher.commits) == 2 and hasher.commits[-1] is status.device_status
try:
    status.check_after_result_copy(generation, status.device_status.clone())
except Failure as exc:
    assert exc.generation == 2 and exc.status == ((0, 0), (5, 0))
else:
    raise AssertionError("mapped rank/layer/generation fault escaped result gate")
assert status.active_generation == 2

# The same configured fault must not affect another rank.
other, _, batch = setup(rank=0, size=2)
other_status, other_generation = other.prepare(batch)
assert other_status.device_status.tolist() == [[0, 0], [0, 0]]
other_status.check_after_result_copy(other_generation, other_status.device_status.clone())
os.environ.pop("SGLANG_ENGRAM_PREFLIGHT_FAULT", None)

assert not torch.cuda.is_initialized()
print("PASS: mapped orchestration, registration/file/injected rank-layer-generation status, result gate, no early retirement; CUDA uninitialized")
