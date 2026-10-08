#!/usr/bin/env python3
"""GPU test of the blocked UE8M0 per-token-group fp8 quantiser (0088, SGLANG_DSV41_QUANT_BLOCKED): bit-identical to the one-program-per-group kernel.

Real shapes of the V4.1 prefill (hidden 5120, 576, 1536-row chunks and merged tails, late-layer 128-row calls), bf16 inputs with outliers, zeros and
very small values, row counts that are not a multiple of the block, and a timing comparison.

    sglang-python quant-blocked-gpu.py            # needs a free GPU
"""
import sys

import torch
import triton

from sglang.kernels.ops.quantization import fp8_kernel as fk

dev = "cuda"
torch.manual_seed(0)
fp8_min, fp8_max = -448.0, 448.0


def run(kernel_fn, x):
    q = torch.empty_like(x, dtype=fk.fp8_dtype)
    s = torch.empty(x.shape[:-1] + (x.shape[-1] // 32,), dtype=torch.float32, device=dev)
    kernel_fn(x, q, s)
    return q, s


def reference(x, q, s):
    groups = x.numel() // 32
    fk._per_token_group_quant_8bit_ue8m0_rowmajor[(groups,)](x, q, s, 32, 1e-10, fp8_min, fp8_max, BLOCK=32)


def blocked(x, q, s):
    groups = x.numel() // 32
    fk._per_token_group_quant_8bit_ue8m0_rowmajor_blocked[(triton.cdiv(groups, 16),)](x, q, s, groups, 32, 1e-10, fp8_min, fp8_max, GROUPS=16)


def timed(fn, x, iters=20):
    q = torch.empty_like(x, dtype=fk.fp8_dtype)
    s = torch.empty(x.shape[:-1] + (x.shape[-1] // 32,), dtype=torch.float32, device=dev)
    for _ in range(3):
        fn(x, q, s)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(x, q, s); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


ok = True
for rows, cols in ((1536, 5120), (2047, 5120), (128, 5120), (1536, 4096), (9216, 576), (1537, 5120), (77, 5120), (1536, 1280), (1, 5120)):
    x = (torch.randn(rows, cols, device=dev) * torch.exp(torch.randn(rows, cols // 32, 1, device=dev).repeat_interleave(32, -1).view(rows, cols) * 2)).to(torch.bfloat16)
    x[::7, ::13] = 0
    x[1::11, 5::17] *= 1e-6
    x[2::5, 3] = 3e4
    qa, sa = run(reference, x)
    qb, sb = run(blocked, x)
    same = torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)) and torch.equal(sa.view(torch.int32), sb.view(torch.int32))
    ok &= same
    ta, tb = timed(reference, x), timed(blocked, x)
    print(f"{'ok  ' if same else 'FAIL'} [{rows:5d},{cols:5d}] bit-identical {same}   one-program-per-group {ta*1000:7.1f} us   blocked {tb*1000:7.1f} us   x{ta/tb:4.1f}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
