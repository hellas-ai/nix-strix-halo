"""DSpark draft MoE chain dispatch regression with CPU tensors and device stubs (no GPU).

Checks the guard of `dsv41_mxfp4_draft.try_draft_routed` (128 experts, top-3, 1..16 rows, bf16, gfx1151, contiguous operands,
SGLANG_DSV41_DRAFT_MOE switch), the in-place / live-row plumbing of the chain, and the caller hook in `fused_experts_mxfp4`
(reached only when neither the official policy nor no_combine / apply_router_weight_on_input is active).  The kernels themselves
are covered by `draft-moe-interp.py` (interpreter) and `draft-moe-gpu.py` (GPU, bitwise vs the installed sequence)."""

import ast
import importlib.util
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
    spec = importlib.util.spec_from_file_location("installed_dsv41_mxfp4_draft", SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_draft.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    k2 = sys.modules["sglang.kernels.ops.moe.dsv41_mxfp4_decode"]

    def args(rows=3, **override):
        a = dict(
            hidden_states=Tensor((rows, 5120), torch.bfloat16),
            w13=Tensor((128, 1152, 2560), torch.uint8),
            w2=Tensor((128, 5120, 288), torch.int8),
            s13=Tensor((128, 1152, 160), torch.float8_e8m0fnu),
            s2=Tensor((128, 5120, 18), torch.uint8),
            topk_weights=Tensor((rows, 3), torch.float32),
            topk_ids=Tensor((rows, 3), torch.int32),
            routed_scaling_factor=1.5,
            swiglu_limit=10,
        )
        a.update(override)
        return a

    def ok(**kw):
        k2._is_gfx1151.cache_clear()
        return m.decode_supported(**args(**kw))

    os.environ.pop("SGLANG_DSV41_DRAFT_MOE", None)
    for rows in range(1, 17):
        assert ok(rows=rows), rows
    for rows in (0, 17, 32, 512):
        assert not ok(rows=rows), rows
    bad = {
        "hidden_states": [Tensor((3, 5120), torch.float16), Tensor((3, 5119), torch.bfloat16),
                          Tensor((3, 5120), torch.bfloat16, contiguous=False), Tensor((3, 5120), torch.bfloat16, device="cpu")],
        "w13": [Tensor((384, 1152, 2560), torch.uint8), Tensor((127, 1152, 2560), torch.uint8), Tensor((128, 1152, 2559), torch.uint8),
                Tensor((128, 1152, 2560), torch.float8_e4m3fn), Tensor((128, 1152, 2560), torch.uint8, contiguous=False)],
        "w2": [Tensor((384, 5120, 288), torch.uint8), Tensor((128, 5120, 144), torch.uint8), Tensor((128, 5120, 288), torch.int16)],
        "s13": [Tensor((384, 1152, 160), torch.uint8), Tensor((128, 1152, 160), torch.float32), Tensor((128, 1152, 80), torch.uint8)],
        "s2": [Tensor((128, 5120, 36), torch.uint8), Tensor((128, 5120, 18), torch.float32)],
        "topk_weights": [Tensor((3, 6), torch.float32), Tensor((3, 3), torch.bfloat16), Tensor((3, 3), torch.float32, contiguous=False)],
        "topk_ids": [Tensor((3, 6), torch.int32), Tensor((3, 3), torch.int64), Tensor((3, 3), torch.int32, contiguous=False)],
    }
    for name, values in bad.items():
        for value in values:
            assert not ok(**{name: value}), (name, value.shape, value.dtype)
    # the 384-expert / top-6 target geometry is not the draft's: it keeps the other module
    assert not ok(w13=Tensor((384, 1152, 2560), torch.uint8), w2=Tensor((384, 5120, 288), torch.uint8),
                  s13=Tensor((384, 1152, 160), torch.uint8), s2=Tensor((384, 5120, 18), torch.uint8),
                  topk_weights=Tensor((3, 6), torch.float32), topk_ids=Tensor((3, 6), torch.int32))
    assert not ok(w13=Tensor((128, 1152, 2560), torch.uint8, index=1))
    assert not ok(swiglu_limit=None) and not ok(swiglu_limit=11)
    assert not ok(routed_scaling_factor="1.5")
    assert ok(routed_scaling_factor=None) and ok(routed_scaling_factor=1)
    for arch in ("gfx942", "gfx1100", "gfx950"):
        props.gcnArchName = arch
        assert not ok(), arch
    props.gcnArchName = "gfx1151:sramecc-:xnack-"
    with patch.object(torch.version, "hip", None):
        assert not ok()
    for off in ("0", "false", "FALSE", "off", "no", " 0 "):
        os.environ["SGLANG_DSV41_DRAFT_MOE"] = off
        assert not ok(), off
    for on in ("1", "true", "yes", ""):
        os.environ["SGLANG_DSV41_DRAFT_MOE"] = on
        assert ok(), on
    os.environ.pop("SGLANG_DSV41_DRAFT_MOE")

    # route padding covers every flattened route; tile tables serve every draft row count
    for rows in range(1, 17):
        rp = m._route_pad(rows)
        assert rp >= rows * 3 and rp & (rp - 1) == 0 and rp >= 32, (rows, rp)
        for table, n, k, name in ((m.GATE_UP_CFG, 1152, 5120, "GATE_UP"), (m.DOWN_CFG, 5120, 576, "DOWN")):
            cfg = k2._cfg(table, rows, n, k, name)
            assert n % cfg["BLOCK_N"] == 0 and (k % cfg["BLOCK_K"]) in (0, 32, 64)

    # try_draft_routed: guarded call, in-place semantics, live-row tensor only inside a capture scope
    seen = []

    def fake_chain(*a, **k):
        seen.append((a, k))
        return "routed"

    with patch.object(m, "decode_draft_experts", fake_chain):
        k2._is_gfx1151.cache_clear()
        call = args(rows=3)
        call["inplace"] = True
        assert m.try_draft_routed(**call) == "routed"
        a, k = seen[-1]
        assert k["out"] is call["hidden_states"] and k["live_rows"] is None
        call["inplace"] = False
        assert m.try_draft_routed(**call) == "routed" and seen[-1][1]["out"] is None
        call = args(rows=17)
        call["inplace"] = True
        assert m.try_draft_routed(**call) is None and len(seen) == 2
        import sglang.srt.layers.live_rows as live_pkg

        sentinel = torch.full((1,), 2, dtype=torch.int32)
        call = args(rows=3)
        call["inplace"] = False
        with live_pkg.scope(sentinel):
            assert m.try_draft_routed(**call) == "routed"
        assert seen[-1][1]["live_rows"] is sentinel
        assert m.try_draft_routed(**call) == "routed" and seen[-1][1]["live_rows"] is None

    # chain plumbing: four launches, masked ids only with a live tensor, inplace output is the caller's tensor
    calls = []

    def rec(name):
        def fn(*a, **k):
            calls.append((name, a, k))
        return fn

    stubs = {n: rec(n) for n in ("launch_mask_ids", "launch_gate_up", "launch_swiglu", "launch_down", "launch_combine")}
    x = torch.zeros(6, 5120, dtype=torch.bfloat16)
    ids = torch.arange(18, dtype=torch.int32).reshape(6, 3)
    tw = torch.zeros(6, 3)
    with patch.multiple(m, **stubs):
        calls.clear()
        out = m.decode_draft_experts(x, None, None, None, None, tw, ids, 1.5)
        assert [c[0] for c in calls] == ["launch_gate_up", "launch_swiglu", "launch_down", "launch_combine"]
        assert out.shape == (6, 5120) and out.dtype == torch.bfloat16 and calls[0][1][3] is ids and calls[2][1][3] is ids
        calls.clear()
        out = m.decode_draft_experts(x, None, None, None, None, tw, ids, 1.5, out=x)
        assert out is x and calls[3][1][1] is x
        calls.clear()
        live = torch.full((1,), 4, dtype=torch.int32)
        m.decode_draft_experts(x, None, None, None, None, tw, ids, None, live_rows=live)
        assert [c[0] for c in calls] == ["launch_mask_ids", "launch_gate_up", "launch_swiglu", "launch_down", "launch_combine"]
        masked = calls[0][1][1]
        assert masked is not ids and calls[1][1][3] is masked and calls[3][1][3] is masked and calls[3][1][4] is tw
        assert calls[4][1][2] is None            # no routed_scaling_factor: no scaling in the combine
    try:
        m.launch_mask_ids(ids, torch.empty_like(ids), torch.zeros(1, dtype=torch.int64))
    except AssertionError:
        pass
    else:
        raise AssertionError("bad live-row tensor accepted")

# Caller hook: reached only when the official policy is off and the routed outputs are combined in place of the caller.
path = SITE / "sglang/srt/layers/moe/moe_runner/triton_utils/mxfp4_moe_amd.py"
tree = ast.parse(path.read_text())
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fused_experts_mxfp4")
hooks = [n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and n.module == "sglang.kernels.ops.moe.dsv41_mxfp4_draft"]
assert len(hooks) == 1
guards = [
    n for n in ast.walk(fn)
    if isinstance(n, ast.If) and ast.unparse(n.test) == "not official_routed_policy and (not no_combine) and (not apply_router_weight_on_input)"
]
assert len(guards) == 1 and hooks[0] in ast.walk(guards[0])
assert sum(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "try_draft_routed" for n in ast.walk(fn)) == 1
call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "try_draft_routed")
assert ast.unparse(call.args[-1]) == "inplace" and ast.unparse(call.args[-2]) == "swiglu_limit"
# it precedes the installed sequence (AITER config lookup) and follows the target-policy block
lines = {ast.unparse(n)[:40]: n.lineno for n in fn.body if isinstance(n, ast.Assign)}
assert hooks[0].lineno < next(n.lineno for n in fn.body if isinstance(n, ast.Assign) and "num_tokens" in ast.unparse(n.targets[0]))
print("draft moe dispatch ok")
