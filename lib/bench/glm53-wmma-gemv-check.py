#!/usr/bin/env python3
"""GPU check for sglang-glm53-rocm patch 0011 (WMMA block-FP8 GEMV, 2..8 rows) against the generic Triton matmul.

Run with the GLM runtime's sglang-python on a node GPU. For every GLM TP4 dense FP8 shape and 2/4/8 rows: max relative
error of the WMMA result against w8a8_block_fp8_matmul_triton (same quantised inputs), and the median time of both.
"""
import os
import statistics as st

import torch

os.environ.setdefault("SGLANG_GLM53_WMMA_GEMV", "0")
from sglang.kernels.ops.gemm import glm53_wmma_gemv as wg
from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton

torch.manual_seed(0)
dev = "cuda"


def timed(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)


worst = 0.0
for n, k in sorted(wg.SHAPES):
    w = (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ws = torch.rand(n // 128, k // 128, device=dev) * 0.01 + 0.001
    for m in (2, 4, 8):
        x = (torch.randn(m, k, device=dev) * 2).to(torch.float8_e4m3fn)
        xs = torch.rand(m, k // 128, device=dev) * 0.05 + 0.01
        ref = w8a8_block_fp8_matmul_triton(x, w, xs, ws, [128, 128], torch.bfloat16)
        got = wg.wmma_gemv(x, xs, w, ws)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        worst = max(worst, rel)
        t_ref = timed(lambda: w8a8_block_fp8_matmul_triton(x, w, xs, ws, [128, 128], torch.bfloat16))
        t_got = timed(lambda: wg.wmma_gemv(x, xs, w, ws))
        gb = n * k / 1e9
        print(f"N={n:5d} K={k:5d} rows={m}: rel L2 {rel:.2e}  generic {t_ref*1e3:7.1f} us ({gb/(t_ref/1e3):5.0f} GB/s)"
              f"  wmma {t_got*1e3:7.1f} us ({gb/(t_got/1e3):5.0f} GB/s)")
print(f"worst rel L2 {worst:.2e}")
