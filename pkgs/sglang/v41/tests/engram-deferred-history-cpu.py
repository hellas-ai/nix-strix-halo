#!/usr/bin/env python3
"""Compare default and deferred Engram history on CPU without importing GPU code."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(os.environ.get("DS41_ENGRAM_SOURCE_ROOT", HERE / "src-work"))
source = SOURCE_ROOT / "sglang/srt/layers/engram.py"
tree = ast.parse(source.read_text(), filename=str(source))
selected = [
    node for node in tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    and node.name in (
        "EngramPendingHistoryCommit", "EngramHasher", "compute_engram_hash_ids"
    )
]
namespace = {
    "torch": torch,
    "nn": torch.nn,
    "dataclass": dataclass,
    "Optional": Optional,
    "_cuda_kernels": lambda _: False,
    "MODE_DECODE": 0,
    "MODE_VERIFY": 1,
    "MODE_EXTEND": 2,
    "MM_PAD_SHIFT_VALUE": 1 << 20,
}
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(source), "exec"), namespace)
Hasher = namespace["EngramHasher"]


def hasher():
    obj = Hasher.__new__(Hasher)
    torch.nn.Module.__init__(obj)
    obj.max_ngram_size = 4
    obj.pad_id = 0
    obj.image_token_id = 255
    obj.token_map = (torch.arange(256) * 17) % 127
    obj.multipliers = torch.tensor([[101, 103, 107, 109], [113, 127, 131, 137]])
    obj.primes = torch.arange(3, 51).reshape(2, 3, 8)
    obj.offsets = torch.arange(48).reshape(2, 24) * 53
    obj.init_history(9, "cpu")
    obj.history[:9] = torch.arange(27).reshape(9, 3) + 31
    return obj


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "decode"

    def is_target_verify(self):
        return self.name == "verify"

    def is_extend(self):
        return self.name == "extend"


def batch(name):
    common = dict(req_pool_indices=torch.tensor([6, 1]), engram_history=None)
    if name == "decode":
        return SimpleNamespace(
            **common, forward_mode=Mode(name), input_ids=torch.tensor([91, 92]),
            positions=torch.tensor([61440, 0]), out_cache_loc=torch.tensor([10, 0]),
        )
    if name == "verify":
        return SimpleNamespace(
            **common, forward_mode=Mode(name), input_ids=torch.tensor([91, 92, 93, 94]),
            positions=torch.tensor([61440, 61441, 2, 3]), out_cache_loc=torch.tensor([10, 11, 12, 13]),
            spec_info=SimpleNamespace(draft_token_num=2),
        )
    return SimpleNamespace(
        **common, forward_mode=Mode(name), input_ids=torch.tensor([91, 92, 93, 94, 95]),
        positions=torch.tensor([61440, 61441, 2, 3, 4]), out_cache_loc=torch.tensor([10, 11, 12, 13, 14]),
        extend_seq_lens=torch.tensor([2, 3]), extend_start_loc=torch.tensor([0, 2]),
    )


for name in ("decode", "verify", "extend"):
    current = batch(name)
    standard, staged = hasher(), hasher()
    before = staged.history.clone()
    expected_ids = standard(current.input_ids, current)
    actual_ids, pending = staged(current.input_ids, current, defer_commit=True)
    assert torch.equal(expected_ids, actual_ids), name
    assert torch.equal(staged.history, before), f"{name}: hash-only mutated history"
    staged.commit_preflight(pending)
    assert torch.equal(
        staged.history[:staged.pad_row], standard.history[:standard.pad_row]
    ), f"{name}: live delayed history differs"
    assert torch.equal(
        staged.history[staged.pad_row], before[staged.pad_row]
    ), f"{name}: graph padding modified shared history row"
    try:
        staged.commit_preflight(pending)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{name}: pending history committed twice")
    failed = hasher()
    failed_before = failed.history.clone()
    failed(current.input_ids, current, defer_commit=True)
    assert torch.equal(failed.history, failed_before), f"{name}: failed preflight changed history"

assert not torch.cuda.is_initialized()
print("PASS: canonical IDs/live history match default decode/verify/extend; padding and failures leave history unchanged; one-shot commit; CUDA uninitialized")
