"""Live-row mask of padded decode graphs: plumbing regression with CPU tensors and kernel stubs (no GPU).

Checks (a) the device-side live-row registry (`sglang.srt.layers.live_rows`), (b) the three places where the decode graph
runner uses it (allocate / set at capture / publish before replay / scope during capture, in source order), and (c) that the
MXFP4 decode chain masks routing ids through kernel P only when a live-row tensor is active and then hands the masked ids to
every later stage.  The Triton kernel itself is covered by `live-row-mask-interp.py` (interpreter) and `live-row-mask-gpu.py`."""

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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------
# (a) registry
# ---------------------------------------------------------------------------------------------
live = load("installed_live_rows", SITE / "sglang/srt/layers/live_rows.py")
os.environ.pop("SGLANG_DSV41_MASK_PADDED_ROWS", None)
assert live.enabled()
for off in ("0", "false", "FALSE", "off", "no", " 0 "):
    os.environ["SGLANG_DSV41_MASK_PADDED_ROWS"] = off
    assert not live.enabled(), off
for on in ("1", "true", "yes", ""):
    os.environ["SGLANG_DSV41_MASK_PADDED_ROWS"] = on
    assert live.enabled(), on
os.environ.pop("SGLANG_DSV41_MASK_PADDED_ROWS")

assert live.active() is None
outer, inner = torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32)
with live.scope(outer):
    assert live.active() is outer
    with live.scope(inner):
        assert live.active() is inner
    assert live.active() is outer
    with live.scope(None):
        assert live.active() is None
    assert live.active() is outer
assert live.active() is None
try:
    with live.scope(outer):
        raise RuntimeError("boom")
except RuntimeError:
    pass
assert live.active() is None, "scope must restore on exceptions"

fills = []


rows = live.LiveRows(8, "cpu")
orig_fill = torch.Tensor.fill_


def counting_fill(self, value):
    if self is rows.tensor:
        fills.append(value)
    return orig_fill(self, value)


with patch.object(torch.Tensor, "fill_", counting_fill):
    assert rows.tensor.dtype == torch.int32 and rows.tensor.shape == (1,) and int(rows.tensor) == 8
    rows.publish(8)                      # unchanged: no launch
    rows.set_capture(8)
    assert fills == []
    rows.publish(6)                      # 3 -> 4 style padding: one fill, value visible on the device tensor
    assert fills == [6] and int(rows.tensor) == 6
    rows.publish(6)
    rows.publish(6)
    assert fills == [6]
    rows.publish(8)
    assert fills == [6, 8] and int(rows.tensor) == 8
    rows.set_capture(4)                  # capture of the 4-row bucket
    rows.publish(4)                      # replay of that bucket without padding: nothing to do
    assert fills == [6, 8, 4] and int(rows.tensor) == 4
    ptr = rows.tensor.data_ptr()
    rows.publish(2)
    assert rows.tensor.data_ptr() == ptr, "the pointer baked into captured graphs must be stable"

# ---------------------------------------------------------------------------------------------
# (b) graph runner source structure
# ---------------------------------------------------------------------------------------------
runner_path = SITE / "sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
tree = ast.parse(runner_path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeCudaGraphRunner")
methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

init_src = ast.unparse(methods["__init__"])
assert "live_rows_mod.LiveRows(self.max_num_token, self.device)" in init_src
assert "not self.ragged_verify_mode" in init_src and "live_rows_mod.enabled()" in init_src

prepare = ast.unparse(methods["capture_prepare"])
assert "self._live_rows.set_capture(num_tokens)" in prepare

capture = methods["capture_one_shape"]
scoped = [
    n for n in ast.walk(capture)
    if isinstance(n, ast.With) and any("live_rows_mod.scope" in ast.unparse(i.context_expr) for i in n.items)
]
assert len(scoped) == 1, "capture_one_shape must run warmup and capture inside the live-row scope"
inside = ast.unparse(scoped[0])
assert "self.backend.capture_one(" in inside and "def run_once" in inside and "forward(" in inside
assert "self._live_rows.tensor" in ast.unparse(scoped[0].items[-1].context_expr)

execute = methods["_execute_impl"]
order = []
for node in ast.walk(execute):
    text = ast.unparse(node) if isinstance(node, (ast.Expr, ast.Assign)) else ""
    if "self.load_batch(" in text and isinstance(node, ast.Expr):
        order.append((node.lineno, "load_batch"))
    if "self._live_rows.publish(self.raw_num_token)" in text and isinstance(node, ast.Expr):
        order.append((node.lineno, "publish"))
    if "self.backend.replay(" in text and isinstance(node, ast.Assign):
        order.append((node.lineno, "replay"))
assert [name for _, name in sorted(order)] == ["load_batch", "publish", "replay"], order

# every replay of this runner goes through _execute_impl: no other call site replays a captured key
replays = [
    n.name for n in cls.body if isinstance(n, ast.FunctionDef)
    and "backend.replay(" in ast.unparse(n) and n.name != "_execute_impl"
]
assert replays == [], replays

# ---------------------------------------------------------------------------------------------
# (c) chain plumbing
# ---------------------------------------------------------------------------------------------
props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc-:xnack-")
with patch.object(torch.version, "hip", "7.0"), patch.object(
    torch.cuda, "get_device_properties", lambda index: props
), patch.object(torch.cuda, "current_device", lambda: 0):
    m = load("installed_dsv41_mxfp4_decode_live", SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_decode.py")

    calls = []

    def rec(name, ret=None):
        def fn(*a, **k):
            calls.append((name, a, k))
            return ret
        return fn

    stubs = {
        "launch_prep": rec("prep"),
        "launch_prep_live": rec("prep_live"),
        "launch_gate_up_wmma": rec("gate_up_wmma"),
        "_launch_live_rows_gate_up": rec("gate_up_live", True),
        "launch_swiglu": rec("swiglu"),
        "launch_down_wmma": rec("down"),
        "launch_combine": rec("combine"),
    }
    os.environ.pop("SGLANG_DSV41_MXFP4_DECODE_GEMM", None)
    os.environ.pop("SGLANG_DSV41_MXFP4_GATE_UP_ROWS", None)

    def run(rows, live_tensor):
        calls.clear()
        x = torch.zeros(rows, 5120, dtype=torch.bfloat16)
        ids = torch.arange(rows * 6, dtype=torch.int32).reshape(rows, 6)
        tw = torch.zeros(rows, 6)
        with patch.multiple(m, **stubs):
            out = m.decode_routed_experts(
                x, None, None, None, None, tw, ids, 1.5, 10,
                live_rows=live_tensor,
            )
        assert out.shape == (rows, 5120) and out.dtype == torch.float32
        return ids, [c[0] for c in calls]

    for rows in (1, 2, 3, 4, 5, 8, 16):
        ids, names = run(rows, None)
        assert names[0] == "prep" and "prep_live" not in names, names
        gate = "gate_up_live" if rows in (2, 4, 8) else "gate_up_wmma"
        assert names == ["prep", gate, "swiglu", "down", "combine"], names
        for name, args, kwargs in calls:
            if name in ("gate_up_live", "gate_up_wmma"):
                assert args[3] is ids
            if name == "down":
                assert args[3] is ids
            if name == "combine":
                assert args[1] is ids

        live_tensor = torch.full((1,), max(1, rows - 1), dtype=torch.int32)
        ids, names = run(rows, live_tensor)
        assert names == ["prep_live", gate, "swiglu", "down", "combine"], names
        prep = calls[0]
        assert prep[1][2] is ids and prep[1][4] is live_tensor        # (x, a1, topk_ids, ids_out, live_rows)
        masked = prep[1][3]
        assert masked is not ids and masked.shape == ids.shape and masked.dtype == torch.int32
        for name, args, kwargs in calls[1:]:
            if name in ("gate_up_live", "gate_up_wmma", "down"):
                assert args[3] is masked, name
            if name == "combine":
                assert args[1] is masked
        # inputs of the caller are never modified
        assert torch.equal(ids, torch.arange(rows * 6, dtype=torch.int32).reshape(rows, 6))

    # launcher argument validation (fails before any launch)
    x = torch.zeros(4, 5120, dtype=torch.bfloat16)
    ids = torch.zeros(4, 6, dtype=torch.int32)
    for bad_live in (torch.zeros(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int32)):
        try:
            m.launch_prep_live(x, torch.empty_like(x), ids, torch.empty_like(ids), bad_live)
        except AssertionError:
            pass
        else:
            raise AssertionError("bad live-row tensor accepted")
    try:
        m.launch_prep_live(x, torch.empty_like(x), ids, torch.empty(4, 6, dtype=torch.int64), torch.zeros(1, dtype=torch.int32))
    except AssertionError:
        pass
    else:
        raise AssertionError("bad id buffer accepted")

    # try_decode_routed forwards the active scope tensor, and only inside it
    seen = []
    with patch.object(m, "decode_supported", lambda *a, **k: True), patch.object(
        m, "decode_routed_experts", lambda *a, **k: seen.append(k.get("live_rows")) or "routed"
    ):
        args = (None,) * 5 + (None, None, 1.5, 10)
        assert m.try_decode_routed(*args) == "routed"
        sentinel = torch.full((1,), 3, dtype=torch.int32)
        import sglang.srt.layers.live_rows as live_pkg
        with live_pkg.scope(sentinel):
            assert m.try_decode_routed(*args) == "routed"
        assert m.try_decode_routed(*args) == "routed"
        assert seen == [None, sentinel, None], seen

print("live-row mask plumbing ok")
