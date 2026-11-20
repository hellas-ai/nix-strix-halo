#!/usr/bin/env python3
"""Stable graph staging of already preflighted Engram rows, on CPU tensors."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch


source_root = Path(os.environ.get(
    "DS41_ENGRAM_SOURCE_ROOT", Path(__file__).resolve().parent / "src-work"
))
source = source_root / "sglang/srt/layers/engram_graph_prestage.py"
spec = importlib.util.spec_from_file_location("preflight_graph_prestage", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Event:
    def record(self, _stream): pass
    def synchronize(self): pass


stream = SimpleNamespace(wait_event=lambda _event: None)


def allocate(shape, *, dtype, device=None, pin_memory=False):
    assert device in (None, "cpu")
    return torch.zeros(shape, dtype=dtype)


module.torch = SimpleNamespace(
    zeros=allocate, empty=allocate, uint8=torch.uint8,
    cuda=SimpleNamespace(Event=Event, current_stream=lambda: stream),
)
ROW_BYTES = module.ROW_BYTES


class Hasher:
    calls = 0

    def __call__(self, *_args):
        self.calls += 1
        raise AssertionError("external preflight rows must bypass graph hashing")


class Store:
    def lookup(self, *_args):
        raise AssertionError("external preflight rows must bypass graph file lookup")


hasher = Hasher()
model = SimpleNamespace(
    engram_hasher=hasher,
    layers=[SimpleNamespace(engram=SimpleNamespace(layer_hash_index=i,
            embed=SimpleNamespace(file_store=Store()))) for i in range(2)],
)
os.environ["SGLANG_ENGRAM_NATIVE_PREFLIGHT"] = "1"
prestage = module.NativeEngramGraphPrestage(model, 4, "cpu", capture_bs=(1, 2, 4))


def batch(bs, byte):
    values = torch.full((bs, 2, 24, ROW_BYTES), byte, dtype=torch.uint8)
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False),
        input_ids=torch.arange(bs), positions=torch.arange(bs),
        req_pool_indices=torch.arange(bs), out_cache_loc=torch.arange(1, bs + 1),
        engram_packed_rows=values,
    )


first = batch(3, 71)
gen = prestage.prepare(first, 4)
prestage.install(first, gen)
assert torch.equal(first.engram_packed_rows, torch.full((3, 2, 24, ROW_BYTES), 71, dtype=torch.uint8))
assert not prestage.gpu_rows[3:].any()
prestage.retire_after_replay(gen)

second = batch(1, 93)
gen = prestage.prepare(second, 1)
prestage.install(second, gen)
assert second.engram_packed_rows.unique().tolist() == [93]
assert hasher.calls == 0
prestage.retire_after_replay(gen)
del os.environ["SGLANG_ENGRAM_NATIVE_PREFLIGHT"]
assert not torch.cuda.is_initialized()
print("PASS: preflight rows use stable graph input, clear padding, preserve generation retirement, no rehash/file lookup; CUDA uninitialized")
