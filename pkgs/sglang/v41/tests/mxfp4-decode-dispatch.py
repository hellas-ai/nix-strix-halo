"""Installed MXFP4 decode dispatch regression with CPU tensors and device stubs (no GPU).

Checks the guard of `dsv41_mxfp4_decode.try_decode_routed` and the caller hook in `fused_experts_mxfp4`:
exact TP4 geometry, dtypes, strides, row range, arch and the SGLANG_DSV41_MXFP4_DECODE switch.  The GPU test
(`mxfp4-decode.py`) separately checks every stage's bits against the installed sequence."""

import ast
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
assert not Path("/dev/kfd").exists() or not torch.cuda.is_available()


class Tensor:
    def __init__(self, shape, dtype, contiguous=True, device="cuda", index=0):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device, index=index)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc-:xnack-")
with patch.object(torch.version, "hip", "7.0"), patch.object(
    torch.cuda, "get_device_properties", lambda index: props
), patch.object(torch.cuda, "current_device", lambda: 0):
    spec = importlib.util.spec_from_file_location(
        "installed_dsv41_mxfp4_decode",
        SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_decode.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    def args(rows=1, **override):
        a = dict(
            hidden_states=Tensor((rows, 5120), torch.bfloat16),
            w13=Tensor((384, 1152, 2560), torch.uint8),
            w2=Tensor((384, 5120, 288), torch.int8),
            s13=Tensor((384, 1152, 160), torch.float8_e8m0fnu),
            s2=Tensor((384, 5120, 18), torch.uint8),
            topk_weights=Tensor((rows, 6), torch.float32),
            topk_ids=Tensor((rows, 6), torch.int32),
            routed_scaling_factor=1.5,
            swiglu_limit=10,
        )
        a.update(override)
        return a

    def ok(**kw):
        m._is_gfx1151.cache_clear()
        return m.decode_supported(**args(**kw))

    os.environ.pop("SGLANG_DSV41_MXFP4_DECODE", None)
    for rows in range(1, 17):
        assert ok(rows=rows), rows
    for rows in (0, 17, 32, 512):
        assert not ok(rows=rows), rows
    # every operand: wrong dtype / shape / stride / device is rejected
    bad = {
        "hidden_states": [Tensor((1, 5120), torch.float16), Tensor((1, 5119), torch.bfloat16),
                          Tensor((1, 5120), torch.bfloat16, contiguous=False),
                          Tensor((1, 5120), torch.bfloat16, device="cpu")],
        "w13": [Tensor((384, 1152, 2560), torch.float8_e4m3fn), Tensor((383, 1152, 2560), torch.uint8),
                Tensor((384, 1152, 2559), torch.uint8), Tensor((384, 1152, 2560), torch.uint8, contiguous=False)],
        "w2": [Tensor((384, 5120, 144), torch.uint8), Tensor((384, 5120, 288), torch.int16)],
        "s13": [Tensor((384, 1152, 160), torch.float32), Tensor((384, 1152, 80), torch.uint8)],
        "s2": [Tensor((384, 5120, 18), torch.float32), Tensor((384, 5120, 36), torch.uint8)],
        "topk_weights": [Tensor((1, 6), torch.bfloat16), Tensor((1, 7), torch.float32),
                         Tensor((1, 6), torch.float32, contiguous=False)],
        "topk_ids": [Tensor((1, 6), torch.int64), Tensor((1, 5), torch.int32), Tensor((1, 6), torch.int32, contiguous=False)],
    }
    for name, values in bad.items():
        for value in values:
            assert not ok(**{name: value}), (name, value.shape, value.dtype)
    assert not ok(w13=Tensor((384, 1152, 2560), torch.uint8, index=1))  # operand on another device
    assert not ok(swiglu_limit=None) and not ok(swiglu_limit=11)
    assert not ok(routed_scaling_factor="1.5")
    assert ok(routed_scaling_factor=None) and ok(routed_scaling_factor=1)
    # arch / platform / switch
    for arch in ("gfx942", "gfx1100", "gfx950"):
        props.gcnArchName = arch
        assert not ok(), arch
    props.gcnArchName = "gfx1151:sramecc-:xnack-"
    with patch.object(torch.version, "hip", None):
        assert not ok()
    for off in ("0", "false", "FALSE", "off", "no", " 0 "):
        os.environ["SGLANG_DSV41_MXFP4_DECODE"] = off
        assert not ok(), off
    for on in ("1", "true", "yes", ""):
        os.environ["SGLANG_DSV41_MXFP4_DECODE"] = on
        assert ok(), on
    os.environ.pop("SGLANG_DSV41_MXFP4_DECODE")
    # gate/up policy for 2/4/8 rows: the installed FMA live rows unless WMMA is requested explicitly
    os.environ.pop("SGLANG_DSV41_MXFP4_GATE_UP_ROWS", None)
    assert m.gate_up_rows_policy() == "live"
    for value, expected in (("live", "live"), ("WMMA", "wmma"), (" wmma ", "wmma")):
        os.environ["SGLANG_DSV41_MXFP4_GATE_UP_ROWS"] = value
        assert m.gate_up_rows_policy() == expected, value
    os.environ["SGLANG_DSV41_MXFP4_GATE_UP_ROWS"] = "fma"
    try:
        m.gate_up_rows_policy()
    except ValueError:
        pass
    else:
        raise AssertionError("unknown gate/up policy accepted")
    os.environ.pop("SGLANG_DSV41_MXFP4_GATE_UP_ROWS")
    os.environ.pop("SGLANG_DSV41_MXFP4_DECODE_GEMM", None)
    assert m.gemm_backend() == "owner"
    for value, expected in (("owner", "owner"), (" AITER ", "aiter")):
        os.environ["SGLANG_DSV41_MXFP4_DECODE_GEMM"] = value
        assert m.gemm_backend() == expected, value
    os.environ["SGLANG_DSV41_MXFP4_DECODE_GEMM"] = "wmma"
    try:
        m.gemm_backend()
    except ValueError:
        pass
    else:
        raise AssertionError("unknown GEMM backend accepted")
    os.environ.pop("SGLANG_DSV41_MXFP4_DECODE_GEMM")
    # launch geometry tables: every configured tile divides its GEMM and leaves a power-of-two K tail
    for table, n, k in ((m.GATE_UP_CFG, 1152, 5120), (m.DOWN_CFG, 5120, 576)):
        for rows in (0, 1, 2, 3, 4, 8, 16):
            cfg = m._cfg(table, rows, n, k, "GATE_UP" if n == 1152 else "DOWN")
            assert n % cfg["BLOCK_N"] == 0 and (k % cfg["BLOCK_K"]) in (0, 32, 64)
    os.environ["SGLANG_DSV41_MXFP4_DOWN_CFG"] = "64,128,4,3"
    assert m._cfg(m.DOWN_CFG, 4, 5120, 576, "DOWN") == dict(BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3)
    for bad in ("48,128,4,2", "32,100,2,2", "32,256,2,9", "32,256,3,2", "32,1024,2,2"):
        os.environ["SGLANG_DSV41_MXFP4_DOWN_CFG"] = bad
        try:
            m._cfg(m.DOWN_CFG, 1, 5120, 576, "DOWN")
        except AssertionError:
            pass
        else:
            raise AssertionError("bad tile override accepted: " + bad)
    os.environ.pop("SGLANG_DSV41_MXFP4_DOWN_CFG")
    assert m._route_pad(4) == 32 and m._route_pad(5) == 32 and m._route_pad(6) == 128 and m._route_pad(16) == 128
    # try_decode_routed hands exactly the guarded call to the five-kernel chain, else returns None
    calls = []
    with patch.object(m, "decode_routed_experts", lambda *a, **k: calls.append(a) or "routed"):
        m._is_gfx1151.cache_clear()
        assert m.try_decode_routed(**args(rows=4)) == "routed" and len(calls) == 1
        assert m.try_decode_routed(**args(rows=17)) is None and len(calls) == 1

# Caller hook: reached only inside the official-routed-policy block of fused_experts_mxfp4.
path = SITE / "sglang/srt/layers/moe/moe_runner/triton_utils/mxfp4_moe_amd.py"
tree = ast.parse(path.read_text())
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fused_experts_mxfp4")
hooks = [n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and n.module == "sglang.kernels.ops.moe.dsv41_mxfp4_decode"]
assert len(hooks) == 1
policy_blocks = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "official_routed_policy"]
assert any(hooks[0] in ast.walk(block) for block in policy_blocks)
assert sum(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "try_decode_routed" for n in ast.walk(fn)) == 1
# the legacy dispatch helpers keep their exact installed shape (the GPU controls parse them)
helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_mxfp4_gemm")
assert ast.unparse(helper.body[0].test) == (
    "a.shape in ((2, 5120), (4, 5120), (8, 5120)) and top_k == 6 and (not mul_routed_weight)"
)
print("mxfp4 decode dispatch ok")
