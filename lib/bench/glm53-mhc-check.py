#!/usr/bin/env python3
"""Check fused ROCm mHC against a CPU double reference and time both paths."""

import json
import os
import time
from functools import partial

os.environ["SGLANG_OPT_USE_TILELANG_MHC_PRE"] = "0"
import torch
from sglang.kernels.ops.layernorm.mhc import (
    _mhc_pre_dispatch,
    _mhc_pre_torch,
    _mhc_project_hip,
)


def reference(residual, fn, scale, base, eps=1e-6):
    # CPU double projection and scalar-sized matrix normalization, independently
    # accumulated to avoid the GPU kernel's fused reductions and BF16 arithmetic.
    x = residual.cpu().double()
    s, n, h = x.shape
    flat = x.reshape(s, n * h)
    logits = (flat @ fn.cpu().double().T) / (
        flat.square().sum(-1, keepdim=True) / (n * h) + eps
    ).sqrt()
    a, b, c = scale.cpu().double()
    bias = base.cpu().double()
    pre = torch.sigmoid(a * logits[:, :n] + bias[:n]) + eps
    post = 2 * torch.sigmoid(b * logits[:, n : 2 * n] + bias[n : 2 * n])
    logits = (c * logits[:, 2 * n :] + bias[2 * n :]).reshape(s, n, n)
    matrix = torch.exp(logits - logits.amax(-1, keepdim=True))
    matrix = matrix / matrix.sum(-1, keepdim=True) + eps
    for iteration in range(20):
        if iteration:
            matrix = matrix / (matrix.sum(-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + eps)
    activation = torch.einsum("sn,snh->sh", pre, x)
    return post.unsqueeze(-1), matrix, activation


def timing(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    return (time.monotonic() - start) * 1000 / 50


torch.manual_seed(5353)
torch.set_num_threads(4)
for tokens in [1, 17, 32, 33, 1024]:
    for base_scale in [1, 10]:
        x = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
        fn = torch.randn(24, 16384, device="cuda") * 0.01
        scale = torch.tensor([0.1, 0.1, 0.1], device="cuda")
        base = torch.randn(24, device="cuda") * base_scale
        actual = _mhc_pre_dispatch(x, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)[:3]
        expected = reference(x, fn, scale, base)
        rel = []
        for i, (a, b) in enumerate(zip(actual, expected)):
            a = a.cpu().double()
            rel.append(((a - b).norm() / b.norm()).item())
            torch.testing.assert_close(
                a, b, rtol=0.01 if i == 2 else 1e-4, atol=0.03 if i == 2 else 2e-6
            )
            assert rel[-1] < (0.004 if i == 2 else 1e-5)
        row = {"tokens": tokens, "base_scale": base_scale, "relative_l2": rel}
        if base_scale == 1:
            row["eager_ms"] = timing(
                partial(_mhc_pre_torch, x, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
            )
            row["hip_ms"] = timing(
                partial(
                    _mhc_pre_dispatch, x, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20
                )
            )
        print(json.dumps(row), flush=True)
print("PASS mHC CPU double reference")

# Different epsilons/post scaling must stay on the general fallback.
x = torch.randn(7, 4, 4096, device="cuda", dtype=torch.bfloat16)
for pre_eps, post_mult in [(1e-5, 2.0), (1e-6, 3.0)]:
    args = (x, fn, scale, base, 1e-6, pre_eps, 1e-6, post_mult, 20)
    for actual, expected in zip(
        _mhc_pre_dispatch(*args)[:3], _mhc_pre_torch(*args), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print("PASS nonstandard mHC fallback")

# The decode projection uses a split reduction, while larger batches retain
# BLAS. Check its actual model dimensions against independent FP64 sums,
# including zero/tiny inputs and the model's RMS epsilon.
for tokens in (1, 4, 32):
    for magnitude in (0, 1e-5, 1, 100):
        x = (
            torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
            * magnitude
        )
        fn = torch.randn(24, 16384, device="cuda") * 0.01
        flat = x.cpu().double().reshape(tokens, -1)
        expected = (flat @ fn.cpu().double().T) * (
            flat.square().mean(-1, keepdim=True) + 1e-5
        ).rsqrt()
        actual = _mhc_project_hip(x, fn, 1e-5)
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=2e-6, rtol=1e-4
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = _mhc_project_hip(x, fn, 1e-5)
        for _ in range(8):
            graph.replay()
            torch.testing.assert_close(replayed, actual, atol=0, rtol=0)
        print(
            json.dumps(
                {
                    "projection": True,
                    "tokens": tokens,
                    "input_magnitude": magnitude,
                    "max_abs_double": float(
                        (actual.cpu().double() - expected).abs().max()
                    ),
                    "graph_repeat_exact": True,
                }
            ),
            flush=True,
        )
print("PASS split mHC projection CPU double reference and graph replay")
