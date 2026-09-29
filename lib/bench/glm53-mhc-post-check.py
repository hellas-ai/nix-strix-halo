#!/usr/bin/env python3
"""Check fused ROCm mHC post mixing against CPU double, including strided inputs."""

import json
import os
import time
from functools import partial

import torch

os.environ["SGLANG_OPT_USE_TILELANG_MHC_POST"] = "0"
from sglang.kernels.ops.layernorm.mhc import _mhc_post_dispatch as post
from sglang.kernels.ops.layernorm.mhc import _mhc_post_torch

torch.set_num_threads(4)
torch.manual_seed(5354)


def timing(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.monotonic()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    return (time.monotonic() - t) * 1000 / 50


for n, h, strided in [
    (0, 4096, False),
    (1, 4096, False),
    (17, 4096, False),
    (1024, 4096, False),
    (7, 257, False),
    (17, 4096, True),
]:
    for dtype in [torch.bfloat16, torch.float32]:
        x = torch.randn(n, h, device="cuda", dtype=dtype)
        r = torch.randn(n, 4, h, device="cuda", dtype=dtype)
        p = torch.rand(n, 4, 1, device="cuda") * 2
        c = torch.rand(n, 4, 4, device="cuda")
        c /= c.sum(1, keepdim=True)
        if strided:
            x = x.repeat_interleave(2, 0)[::2]
            r = r.repeat_interleave(2, 0)[::2]
            p = p.transpose(0, 1).contiguous().transpose(0, 1)
            c = c.transpose(1, 2)
        out = post(x, r, p, c)
        ref = p.cpu().double() * x.cpu().double().unsqueeze(1) + torch.einsum(
            "sij,sih->sjh", c.cpu().double(), r.cpu().double()
        )
        rel = 0 if n == 0 else float((out.cpu().double() - ref).norm() / ref.norm())
        assert rel < (0.004 if dtype == torch.bfloat16 else 1e-6), (n, h, dtype, rel)
        row = {
            "tokens": n,
            "hidden": h,
            "strided": strided,
            "dtype": str(dtype),
            "relative_l2": rel,
        }
        if n and not strided and dtype == torch.bfloat16:
            row["torch_ms"] = timing(partial(_mhc_post_torch, x, r, p, c))
            row["triton_ms"] = timing(partial(post, x, r, p, c))
        print(json.dumps(row), flush=True)
print("PASS mHC post independent CPU double reference", flush=True)

# Unsupported expansion counts, mix dtypes, and hidden-column strides preserve
# the general torch fallback exactly.
for n, dtype, stride in [
    (3, torch.float32, 1),
    (4, torch.bfloat16, 1),
    (4, torch.float32, 2),
]:
    x = torch.randn(7, 258, device="cuda", dtype=torch.bfloat16)[:, ::stride]
    r = torch.randn(7, n, 258, device="cuda", dtype=torch.bfloat16)[:, :, ::stride]
    p = torch.rand(7, n, 1, device="cuda", dtype=dtype)
    c = torch.rand(7, n, n, device="cuda", dtype=dtype)
    torch.testing.assert_close(
        post(x, r, p, c), _mhc_post_torch(x, r, p, c), rtol=0, atol=0
    )
print("PASS general mHC post fallback", flush=True)
