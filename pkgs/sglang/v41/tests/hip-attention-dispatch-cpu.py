#!/usr/bin/env python3
"""CPU-only checks of the HIP block attention dispatch guard (0085) and that the prebuilt library exports its entry point.

  hip-attention-dispatch-cpu.py RUNTIME_ROOT
"""
import importlib.util
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "lib/python3.13/site-packages/sglang/kernels/ops/attention/nsa_triton_decode"
# Loading the library initialises the HIP runtime (device code is registered at dlopen), which aborts without a GPU in the build sandbox;
# check the file and its exported symbol name instead.
blob = (root / "libdsv41_attn_block.so").read_bytes()
assert blob[:4] == b"\x7fELF" and b"dsv41_attn_block" in blob, "prebuilt library missing or lacks dsv41_attn_block"


def load(env):
    for key in ("SGLANG_DSV41_HIP_ATTENTION", "SGLANG_DSV41_HIP_ATTENTION_DECODE", "SGLANG_DSV41_HIP_ATTENTION_MIN_ROWS"):
        os.environ.pop(key, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("dsv41_hip_attention_under_test", root / "dsv41_hip_attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._is_gfx1151 = lambda device: True      # the guard asks the GPU; here every device qualifies
    return mod


import torch  # noqa: E402  (the module under test imports it too)


def args(rows=64, *, extra=True, sink=True, lens=True, page_main=256, page_extra=128, q_dtype=torch.bfloat16, heads=16):
    q = torch.zeros(rows, 1, heads, 512, dtype=q_dtype)
    main = torch.zeros(7, page_main * 528, dtype=torch.uint8).view(7, page_main, 1, 528)
    ext = torch.zeros(9, page_extra * 528, dtype=torch.uint8).view(9, page_extra, 1, 528) if extra else None
    ids = torch.zeros(rows, 1, 128, dtype=torch.int32)
    eids = torch.zeros(rows, 1, 512, dtype=torch.int32) if extra else None
    ln = torch.zeros(rows, dtype=torch.int32) if lens else None
    s = torch.zeros(16, dtype=torch.float32) if sink else None
    return [q, main, ids, ln, ext, eids, ln if extra else None, s, 1]


count = 0
off = load({})
assert not off.enabled() and not off.supported(*args())
count += 1

on = load({"SGLANG_DSV41_HIP_ATTENTION": "1"})
assert on.enabled()
for rows in (17, 64, 1536, 2047):
    assert on.supported(*args(rows)), rows
    count += 1
for rows in (1, 2, 4, 16):                                   # decode / verify shapes keep the installed kernels
    assert not on.supported(*args(rows)), rows
    count += 1
assert on.supported(*args(64, extra=False)) and on.supported(*args(64, sink=False)) and on.supported(*args(64, lens=False))
assert on.supported(*args(64, page_extra=256)) and on.supported(*args(64, page_main=128, page_extra=128))
count += 4
bad = [dict(q_dtype=torch.float32), dict(heads=32), dict(page_main=192)]
for kw in bad:
    try:
        ok = on.supported(*args(64, **kw))
    except Exception:
        ok = False
    assert not ok, kw
    count += 1
a = args(64); a[-1] = 2                                       # s_q != 1
assert not on.supported(*a)
a = args(64); a[2] = a[2].to(torch.int64)
assert not on.supported(*a)
a = args(64); a[7] = a[7].to(torch.bfloat16)
assert not on.supported(*a)
a = args(64); a[4] = None                                     # extra cache without extra indices
assert not on.supported(*a)
a = args(64); a[1] = torch.zeros(7, 256, 1, 520, dtype=torch.uint8)       # not the 528-byte V4.1 slot layout
assert not on.supported(*a)
count += 5

dec = load({"SGLANG_DSV41_HIP_ATTENTION_DECODE": "1"})
assert dec.enabled() and all(dec.supported(*args(r)) for r in (1, 4, 16))
assert not any(dec.supported(*args(r)) for r in (17, 64, 1536))        # the decode flag alone does not route prefill rows
count += 2
both = load({"SGLANG_DSV41_HIP_ATTENTION": "1", "SGLANG_DSV41_HIP_ATTENTION_DECODE": "1"})
assert all(both.supported(*args(r)) for r in (1, 4, 16, 17, 64, 1536))
count += 1
mr = load({"SGLANG_DSV41_HIP_ATTENTION": "1", "SGLANG_DSV41_HIP_ATTENTION_MIN_ROWS": "128"})
assert not mr.supported(*args(64)) and mr.supported(*args(128))
count += 2
print(f"PASS {count} CPU HIP attention dispatch cases; library exports dsv41_attn_block")
