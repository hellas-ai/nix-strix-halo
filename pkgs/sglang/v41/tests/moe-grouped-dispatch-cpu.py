#!/usr/bin/env python3
"""CPU-only checks of the grouped MoE GEMM dispatch (0100/0101): flags, eligibility, scale-byte guard, variant selection; the prebuilt library
exports its entry points; the two edited modules compile.

  moe-grouped-dispatch-cpu.py RUNTIME_ROOT
"""
import importlib.util
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "lib/python3.13/site-packages/sglang"
moe = root / "kernels/ops/moe"
# Loading the library initialises the HIP runtime (device code is registered at dlopen), which aborts without a GPU in the build sandbox;
# check the file and its exported symbols instead.
blob = (moe / "libdsv41_moe_grouped.so").read_bytes()
assert blob[:4] == b"\x7fELF"
for sym in (b"dsv41_moe_route", b"dsv41_moe_gemm", b"dsv41_moe_gemm_decode", b"dsv41_moe_tiles_capacity"):
    assert sym in blob, sym
for src in (moe / "dsv41_moe_grouped.py", moe / "dsv41_mxfp4_decode.py", root / "srt/layers/moe/moe_runner/triton_utils/mxfp4_moe_amd.py"):
    compile(src.read_text(), str(src), "exec")

import torch  # noqa: E402

KEYS = ("SGLANG_DSV41_MOE_GROUPED", "SGLANG_DSV41_MOE_GROUPED_DECODE", "SGLANG_DSV41_MOE_GROUPED_CFG", "SGLANG_DSV41_MOE_GROUPED_DCFG",
        "SGLANG_DSV41_MOE_GROUPED_ORDER", "SGLANG_DSV41_MOE_GROUPED_MIN_ROWS")


def load(env):
    for key in KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("dsv41_moe_grouped_under_test", moe / "dsv41_moe_grouped.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._is_gfx1151 = lambda device: True      # the guard asks the GPU; here every device qualifies
    mod._lib_ok = lambda: True                 # dlopen would initialise the HIP runtime
    return mod


_BIG = {}


def _big(name, shape, dtype, value=0):
    key = (name, shape, dtype, value)
    if key not in _BIG:
        _BIG[key] = torch.full(shape, value, dtype=dtype)
    return _BIG[key]


def tensors(m, *, x_dtype=torch.bfloat16, scale_value=120, ids_dtype=torch.int32, s13_shape=None, w_dtype=torch.uint8):
    x = torch.zeros(m, 5120, dtype=x_dtype)
    w13 = _big("w13", (384, 1152, 2560), w_dtype)
    w2 = _big("w2", (384, 5120, 288), w_dtype)
    s13 = _big("s13", s13_shape or (384, 1152, 160), torch.uint8, scale_value)
    s2 = _big("s2", (384, 5120, 18), torch.uint8, scale_value)
    ids = torch.zeros(m, 6, dtype=ids_dtype)
    return x, w13, w2, s13, s2, ids


count = 0
off = load({})
assert not off.enabled()
assert not off.supported(*tensors(64))
count += 2

on = load({"SGLANG_DSV41_MOE_GROUPED": "1"})
on.torch.cuda.is_current_stream_capturing = lambda: False     # CPU tensors: no stream to ask
assert on.enabled()
for m in (17, 64, 128, 1536, 2047):
    assert on.supported(*tensors(m)), m
    count += 1
for m in (1, 4, 16):                                          # the prefill flag alone does not take decode rows
    assert not on.supported(*tensors(m)), m
    count += 1
assert on.supported(*tensors(17, w_dtype=torch.int8))
count += 1
for kw in (dict(x_dtype=torch.float16), dict(ids_dtype=torch.int64), dict(s13_shape=(384, 1152, 80)), dict(w_dtype=torch.float16)):
    try:
        ok = on.supported(*tensors(64, **kw))
    except Exception:
        ok = False
    assert not ok, kw
    count += 1
x, w13, w2, s13, s2, ids = tensors(64)
assert not on.supported(x, w13[:383], w2, s13, s2, ids)       # other expert count
assert not on.supported(x[:, :5000], w13, w2, s13, s2, ids)    # other hidden size
assert not on.supported(x, w13, w2, s13, s2, ids[:, :5])       # other top-k
count += 3

# a library that cannot be loaded disables the path
nolib = load({"SGLANG_DSV41_MOE_GROUPED": "1", "SGLANG_DSV41_MOE_GROUPED_DECODE": "1"})
nolib.torch.cuda.is_current_stream_capturing = lambda: False
assert nolib.supported(*tensors(64)) and nolib.decode_weights_ok(*tensors(4)[3:5])
nolib._lib_ok = lambda: False
assert not nolib.supported(*tensors(64)) and not nolib.decode_weights_ok(*tensors(4)[3:5])
count += 2

# smallest prefill row count and tile order
mr = load({"SGLANG_DSV41_MOE_GROUPED": "1", "SGLANG_DSV41_MOE_GROUPED_MIN_ROWS": "256", "SGLANG_DSV41_MOE_GROUPED_ORDER": "1"})
mr.torch.cuda.is_current_stream_capturing = lambda: False
assert not mr.supported(*tensors(128)) and mr.supported(*tensors(256)) and mr.supported(*tensors(1536))
assert mr._ORDER == 1 and load({})._ORDER == 1 and load({"SGLANG_DSV41_MOE_GROUPED_ORDER": "0"})._ORDER == 0
count += 3

# scale-byte guard: every E8M0 byte must be in [1, 252]; checked once per tensor
for value, want in ((117, True), (127, True), (1, True), (252, True), (0, False), (253, False), (255, False)):
    mod = load({"SGLANG_DSV41_MOE_GROUPED": "1"})
    mod.torch.cuda.is_current_stream_capturing = lambda: False
    assert mod.scales_ok(torch.full((4, 8, 32), value, dtype=torch.uint8)) is want, value
    count += 1
mod = load({"SGLANG_DSV41_MOE_GROUPED": "1"})
mod.torch.cuda.is_current_stream_capturing = lambda: False
s = torch.full((4, 8, 32), 120, dtype=torch.uint8)
assert mod.scales_ok(s)
s[2, 3, 4] = 0                                                # cached by pointer and shape: the answer for this tensor does not change
assert mod.scales_ok(s)
s2_ = torch.full((4, 8, 32), 120, dtype=torch.uint8); s2_[3, 7, 31] = 255
assert not mod.scales_ok(s2_)
count += 3
mod.torch.cuda.is_current_stream_capturing = lambda: True     # an unseen tensor is refused while a CUDA graph is captured
assert not mod.scales_ok(torch.full((4, 8, 32), 120, dtype=torch.uint8))
count += 1

# decode flag
dec = load({"SGLANG_DSV41_MOE_GROUPED_DECODE": "1"})
dec.torch.cuda.is_current_stream_capturing = lambda: False
assert dec.enabled() and all(dec.supported(*tensors(m)) for m in (1, 2, 4, 8, 16))
assert not any(dec.supported(*tensors(m)) for m in (17, 64))   # the decode flag alone does not route prefill rows
assert all(dec.decode_serves(m, g) for m in range(1, 17) for g in (True, False))
assert not dec.decode_serves(17, True) and not dec.decode_serves(0, False)
count += 4
exact = load({"SGLANG_DSV41_MOE_GROUPED_DECODE": "exact"})
assert all(exact.decode_serves(m, False) for m in range(1, 17))        # down: every row count (the installed kernel is WMMA)
assert all(exact.decode_serves(m, True) == (m not in (2, 4, 8)) for m in range(1, 17))
count += 2
assert not load({"SGLANG_DSV41_MOE_GROUPED_DECODE": "0"}).decode_serves(4, False)
count += 1

# variants
c = load({})
assert (c.cfg_for("gate_up"), c.cfg_for("down")) == (c._DEFAULT_CFG["gate_up"], c._DEFAULT_CFG["down"])
c = load({"SGLANG_DSV41_MOE_GROUPED_CFG": "3,13", "SGLANG_DSV41_MOE_GROUPED_DCFG": "1,5"})
assert (c.cfg_for("gate_up"), c.cfg_for("down"), c.decode_cfg_for("gate_up"), c.decode_cfg_for("down")) == (3, 13, 1, 5)
for bad in ("12,12", "0,0", "7,13", "x", "1"):
    c = load({"SGLANG_DSV41_MOE_GROUPED_CFG": bad})
    ok = True
    for kind in ("gate_up", "down"):
        try:
            c.cfg_for(kind)
        except ValueError:
            ok = False
    assert not ok, bad
count += 7
print(f"PASS {count} CPU grouped MoE dispatch cases; library exports its entry points")
