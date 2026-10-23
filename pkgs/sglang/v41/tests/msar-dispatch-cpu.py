#!/usr/bin/env python3
"""CPU-only checks of the MSCCL++ small-message all-reduce (0090): the prebuilt library exists and links MSCCL++, the dispatch is
default-off and guarded, and the eligibility rule is shape/dtype only (all ranks must agree without communicating).

  msar-dispatch-cpu.py RUNTIME_ROOT
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(sys.argv[1]) / "lib/python3.13/site-packages/sglang"
lib = root / "srt/distributed/device_communicators/libdsv41_msar.so"
blob = lib.read_bytes()
assert blob[:4] == b"\x7fELF" and b"msar_allreduce" in blob and b"msar_create" in blob, "prebuilt library missing its C ABI"
assert b"libmscclpp.so" in blob, "library must link MSCCL++"
assert b"libibverbs.so.1" in blob, "libibverbs path must be pinned at build time"

ps = (root / "srt/distributed/parallel_state.py").read_text()
assert "dsv41_msar_comm.should_use(input_)" in ps and "dsv41_msar.enabled_for(group_name)" in ps, "hook missing in parallel_state"
assert ps.index("dsv41_msar_comm.should_use(input_)") > ps.index("torch.compiler.is_compiling()"), "the hook must come after the Dynamo branch"

def load():
    spec = importlib.util.spec_from_file_location("dsv41_msar_under_test", root / "srt/distributed/device_communicators/dsv41_msar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


os.environ.pop("SGLANG_DSV41_MSAR", None)
off = load()
assert off.enabled_for("tp") is False, "must be off by default"
os.environ["SGLANG_DSV41_MSAR"] = "1"
on = load()
assert on.enabled_for("tp") and on.enabled_for("attention_tp") and not on.enabled_for("moe_ep") and not on.enabled_for("world")

import torch  # noqa: E402


def make(dtype, n, contiguous=True, cuda=True):
    t = NS(dtype=dtype, is_cuda=cuda, is_contiguous=lambda: contiguous, numel=lambda: n)
    return t


inst = object.__new__(on.Dsv41Msar)
inst.disabled, inst.world, inst.max_bytes = False, 4, 327680
cases = 0
assert inst.should_use(make(torch.bfloat16, 5120)); cases += 1            # one row of hidden 5120, bf16 (10 KiB)
assert inst.should_use(make(torch.float32, 16 * 5120)); cases += 1        # sixteen fp32 rows (320 KiB): the largest handled
assert not inst.should_use(make(torch.float32, 16 * 5120 + 64)); cases += 1  # above the cap
assert not inst.should_use(make(torch.float16, 5120)); cases += 1         # other dtypes keep the existing path
assert not inst.should_use(make(torch.bfloat16, 5120, contiguous=False)); cases += 1
assert not inst.should_use(make(torch.bfloat16, 5120, cuda=False)); cases += 1
assert not inst.should_use(make(torch.bfloat16, 33)); cases += 1          # slice must be a multiple of 16 bytes
inst.disabled = True
assert not inst.should_use(make(torch.bfloat16, 5120)); cases += 1
print(f"PASS {cases} CPU MSCCL++ all-reduce dispatch cases; library present and linked")
