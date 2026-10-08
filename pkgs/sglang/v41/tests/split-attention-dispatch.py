"""Exercise the installed attention selector/wrappers with a CPU launch boundary."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

runtime = Path(sys.argv[1]).resolve()
source = runtime / "lib/python3.13/site-packages/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_fused.py"
assert not Path("/dev/kfd").exists() and not Path("/dev/dri").exists()
assert not torch.cuda.is_initialized()
# The module's existing FP8-platform constant queries device properties on HIP.
# Substitute that driver boundary; load the complete actual installed module.
with patch.object(torch.cuda, "get_device_properties", return_value=types.SimpleNamespace(gcnArchName="gfx1151:sramecc-")):
    spec = importlib.util.spec_from_file_location("installed_split_attention", source)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert m._is_gfx1151_attention_device(torch.device("cuda", 0))
    assert not m._is_gfx1151_attention_device(torch.device("cpu"))
m._is_gfx1151_attention_device.cache_clear()
with patch.object(torch.cuda, "get_device_properties", return_value=types.SimpleNamespace(gcnArchName="gfx950")):
    assert not m._is_gfx1151_attention_device(torch.device("cuda", 0))
m._is_gfx1151_attention_device.cache_clear()
with patch.object(torch.version, "hip", None), patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("non-HIP device query")):
    assert not m._is_gfx1151_attention_device(torch.device("cuda", 0))
m._is_gfx1151_attention_device.cache_clear()

# Exact range plus neighboring shapes keep the original selector behavior.
for b in range(1, 17):
    assert m._decide_splitk_dual_scope(b, 16, 640, gfx1151_v41_bf16=True) == 4
    assert m._decide_splitk_dual_scope(b, 16, 640) == 0
for b in (0, 17, 32):
    assert m._decide_splitk_dual_scope(b, 16, 640, gfx1151_v41_bf16=True) == 0
for h in (8, 32, 64, 128):
    for width in (256, 512, 639, 640, 641, 1024, 2048):
        assert m._decide_splitk_dual_scope(4, h, width, gfx1151_v41_bf16=True) == m._decide_splitk_dual_scope(4, h, width)
for width in (256, 512, 639, 641, 1024, 2048):
    assert m._decide_splitk_dual_scope(4, 16, width, gfx1151_v41_bf16=True) == m._decide_splitk_dual_scope(4, 16, width)

q = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
with patch.object(m, "_is_gfx1151_attention_device", return_value=True):
    assert m._use_gfx1151_v41_splitk(q, True, 128, 512, 256, 128, 1)
    assert m._use_gfx1151_v41_splitk(q, True, 128, 512, 256, 256, 1)
    assert not m._use_gfx1151_v41_splitk(q, True, 128, 512, 256, 128, 2)
    assert not m._use_gfx1151_v41_splitk(q, True, 320, 320, 256, 128, 1)
    assert not m._use_gfx1151_v41_splitk(q, True, 128, 512, 128, 128, 1)
    assert not m._use_gfx1151_v41_splitk(q, True, 128, 512, 256, 64, 1)

class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            resolved = grid({"BLOCK_H": 16, "BLOCK_N": 64, "BLOCK_D": 128}) if callable(grid) else grid
            self.calls.append((resolved, args, kwargs))
        return launch

nosplit, split, combine = Launch(), Launch(), Launch()
m._fused_gather_attn_dsv4_dual_scope_kernel = nosplit
m._fused_gather_attn_dsv4_dual_scope_splitk_kernel = split
m._combine_splitk_kernel = combine
m._combine_splitk_kernel_8_optimized = combine
m._combine_splitk_kernel_2 = combine

cases = 0
for entry in (m.fused_gather_attn_decode_dsv4_dual_scope, m.fused_gather_attn_decode_dsv4_dual_scope_low_overhead):
    for b, heads, v41, dtype, gfx, force, sequence, expected in (
        (1, 16, True, torch.bfloat16, True, False, 1, 4),
        (4, 16, True, torch.bfloat16, True, False, 2, 0),
        (3, 16, True, torch.bfloat16, True, False, 1, 4),
        (5, 16, True, torch.bfloat16, True, False, 1, 4),
        (16, 16, True, torch.bfloat16, True, False, 1, 4),
        (17, 16, True, torch.bfloat16, True, False, 1, 0),
        (4, 32, True, torch.bfloat16, True, False, 1, 0),
        (4, 16, False, torch.bfloat16, True, False, 1, 0),
        (4, 16, True, torch.float16, True, False, 1, 4),
        (4, 16, True, torch.bfloat16, False, False, 1, 0),
        (4, 16, True, torch.bfloat16, True, True, 1, 0),
    ):
        if force and entry is m.fused_gather_attn_decode_dsv4_dual_scope_low_overhead:
            continue  # This public wrapper does not expose force_no_splitk.
        width = 528 if v41 else 584
        cache = torch.zeros((2, 256, 1, width), dtype=torch.uint8)
        q = torch.zeros((b, heads, 512), dtype=dtype)
        main = torch.zeros((b, 128), dtype=torch.int32)
        extra = torch.zeros((b, 512), dtype=torch.int32)
        kwargs = {"s_q": sequence}
        if force:
            kwargs["force_no_splitk"] = True
        nosplit.calls.clear(); split.calls.clear(); combine.calls.clear()
        with patch.object(m, "_is_gfx1151_attention_device", return_value=gfx):
            output, lse = entry(q, cache, main, 256, cache, extra, 256, 512**-0.5, **kwargs)
        assert output.shape == (b, heads, 512) and lse.shape == (b, heads)
        assert len(split.calls) == (expected != 0)
        assert len(nosplit.calls) == (expected == 0)
        # Both existing wrappers canonicalize Q to contiguous BF16 before dispatch.
        launch_q = (split.calls if expected else nosplit.calls)[0][1][0]
        assert launch_q.dtype == torch.bfloat16 and launch_q.is_contiguous()
        assert torch.equal(launch_q, q.to(torch.bfloat16))
        if expected:
            assert split.calls[0][0][2] == expected
            assert len(combine.calls) == 1
        cases += 1
assert not torch.cuda.is_initialized()
print(f"PASS: actual installed module; selector boundaries + {cases} public wrapper cases; no GPU initialization")
