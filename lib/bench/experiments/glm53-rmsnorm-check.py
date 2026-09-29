#!/usr/bin/env python3
"""Qualify the gfx1151 plain RMSNorm fallback, including checkpoint norm weights."""

import argparse
import json
import re
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from sglang.kernels.ops.layernorm.rocm_rmsnorm import rmsnorm
from sglang.srt.layers.layernorm import RMSNorm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-path", type=Path)
cli = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(5319)
torch.cuda.set_per_process_memory_fraction(0.02)


def check(module, x, label):
    xd = x.cpu().double()
    weight = module.weight.detach().cpu().double()
    reference = xd * (xd.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * weight
    rounded = reference.bfloat16().double()
    original = x.clone()
    with (
        torch.no_grad(),
        patch("sglang.srt.layers.layernorm.rocm_rmsnorm", wraps=rmsnorm) as fused,
    ):
        actual = module.forward_hip(x)
        assert fused.call_count == 1, "Expected the fused fallback to run"
        torch.testing.assert_close(actual, module.forward_native(x), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    actual = actual.cpu().double()
    # One BF16 rounding step after FP32 statistics/weight multiplication.
    torch.testing.assert_close(actual, reference, rtol=0.004, atol=1e-8)
    error = float((actual - rounded).norm() / rounded.norm().clamp(min=1e-30))
    assert error < 0.0004, (label, error)
    print(json.dumps({"case": label, "relative_l2_rounded_fp64": error}), flush=True)


for width in (512, 1536, 4096):
    for weight_dtype in (torch.bfloat16, torch.float32):
        module = RMSNorm(width, eps=1e-5, weight_dtype=weight_dtype).cuda()
        with torch.no_grad():
            module.weight.copy_(torch.randn_like(module.weight) * 0.5 + 1)
        for rows in (1, 17, 1024):
            for magnitude in (0, 1e-5, 1, 100):
                x = (
                    torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                    * magnitude
                )
                check(module, x, [width, str(weight_dtype), rows, magnitude])
        x = torch.randn(1, width, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            module.forward_hip(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = module.forward_hip(x)
            for _ in range(8):
                x.copy_(torch.randn_like(x))
                graph.replay()
                torch.testing.assert_close(
                    replayed, module.forward_hip(x), rtol=0, atol=0
                )
        print(json.dumps({"graph_exact": True, "width": width}), flush=True)

# Special contracts must continue through the existing native implementation.
for label, kwargs in [
    ("cast_before_weight", {"cast_x_before_out_mul": True}),
    ("variance_subset", {"var_hidden_size": 256}),
    ("output_override", {"override_orig_dtype": torch.float32}),
    ("strided", {}),
    ("residual", {}),
    ("post_residual", {"fp32_residual": True}),
    ("empty", {}),
    ("3d", {}),
    ("unsupported_width", {}),
    ("fp32_input", {}),
]:
    width = 257 if label == "unsupported_width" else 512
    module = RMSNorm(width, eps=1e-5, **kwargs).cuda().bfloat16()
    x = torch.randn(4, width, device="cuda", dtype=torch.bfloat16)
    if label == "strided":
        x = x.repeat_interleave(2, 0)[::2]
    elif label == "empty":
        x = x[:0]
    elif label == "3d":
        x = x.view(2, 2, width)
    elif label == "fp32_input":
        x = x.float()
    residual = torch.randn_like(x) if "residual" in label else None
    post = torch.randn_like(x) if label == "post_residual" else None
    with (
        torch.no_grad(),
        patch(
            "sglang.srt.layers.layernorm.rocm_rmsnorm",
            side_effect=AssertionError(label),
        ),
    ):
        actual = module.forward_hip(x, residual, post)
        expected = module.forward_native(x, residual, post)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(json.dumps({"native_fallback_exact": label}), flush=True)

if cli.model_path:
    index = json.loads((cli.model_path / "model.safetensors.index.json").read_text())
    count = 0
    for name, shard in index["weight_map"].items():
        if not (
            re.search(r"\.layers\.(\d+)\.", name)
            and int(re.search(r"\.layers\.(\d+)\.", name)[1]) < 45
            and name.endswith(
                (
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".q_a_layernorm.weight",
                    ".kv_a_layernorm.weight",
                )
            )
        ) and name not in ("model.norm.weight", "model.language_model.norm.weight"):
            continue
        with safe_open(cli.model_path / shard, framework="pt", device="cpu") as f:
            weight = f.get_tensor(name)
        module = RMSNorm(weight.numel(), eps=1e-5, weight_dtype=weight.dtype).cuda()
        with torch.no_grad():
            module.weight.copy_(weight)
        for rows in (1, 17):
            x = torch.randn(rows, weight.numel(), device="cuda", dtype=torch.bfloat16)
            check(module, x, [name, rows])
        count += 1
    assert count == 113, count
    print(json.dumps({"checkpoint_norms_checked": count}), flush=True)
print("PASS gfx1151 RMSNorm references, dispatch, fallbacks and graph replay")
