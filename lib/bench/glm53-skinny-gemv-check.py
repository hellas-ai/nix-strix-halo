#!/usr/bin/env python3
"""Skinny GEMV candidates for GLM-5.3-Flash TP4 decode on gfx1151: y[M, N] = x[M, K] @ W[N, K]^T with M <= 8.

An eager decode profile (2026-10-09) put the BF16 linear-attention projections at ~75% of memory bandwidth through
hipBLASLt (MT16x16x128, 101 calls per token) and the FP32 MoE router (F.linear(x.float(), W_fp32), N=288) at
~60 us for a 4.7 MB weight. For each GLM shape this compares, at 1/2/4/8 rows:
  hipblaslt  F.linear as SGLang calls it today (router: x.float() and an FP32 weight)
  rocblas    torch.mv / torch.mm against the same weight (M = 1 uses the GEMV path)
  triton     a deterministic one-pass Triton GEMV (no split-K atomics), BF16 weight, FP32 accumulation, best of a
             small config sweep
and reports median time, achieved bandwidth on the weight bytes actually read, max relative error against an FP64
reference, and whether two Triton runs are bitwise identical. Run on a free node GPU with the GLM runtime's
sglang-python (torch + triton only).
"""
import statistics as st
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.manual_seed(0)
dev = "cuda"

# (name, N, K, weight bytes per element as SGLang stores it today)
SHAPES = [
    ("router (fp32 today)", 288, 4096),
    ("kda fused q/k/v/b/f_a/g_a", 6416, 4096),
    ("kda o_proj", 4096, 2048),
    ("kda f_b / g_b", 2048, 128),
    ("lm_head", 38720, 4096),
]


@triton.jit
def _skinny_gemv(X, W, Y, M, N, K, stride_xm, stride_wn, stride_ym,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    rows = tl.arange(0, M_PAD)
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0).to(tl.float32)
        x = tl.load(X + rows[:, None] * stride_xm + offs_k[None, :], mask=(rows[:, None] < M) & k_mask[None, :],
                    other=0.0).to(tl.float32)
        if M_PAD == 1:
            acc += tl.sum(w * x, axis=1)[None, :]
        else:
            # The weight tile is loaded once and reused for every row; rows are picked out by mask (M_PAD <= 8).
            for m in tl.static_range(M_PAD):
                xm = tl.sum(tl.where(rows[:, None] == m, x, 0.0), axis=0)
                part = tl.sum(w * xm[None, :], axis=1)
                acc += tl.where(rows[:, None] == m, part[None, :], 0.0)
    tl.store(Y + rows[:, None] * stride_ym + offs_n[None, :], acc.to(Y.dtype.element_ty),
             mask=(rows[:, None] < M) & n_mask[None, :])


def triton_gemv(x, w, out_dtype, block_n, block_k, num_warps):
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=out_dtype)
    m_pad = 1 if m == 1 else (2 if m == 2 else (4 if m <= 4 else 8))
    _skinny_gemv[(triton.cdiv(n, block_n),)](x, w, y, m, n, k, x.stride(0), w.stride(0), y.stride(0),
                                             BLOCK_N=block_n, BLOCK_K=block_k, M_PAD=m_pad, num_warps=num_warps)
    return y


def timed(fn, reps=60):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return st.median(ts) * 1e3  # us


CONFIGS = [(bn, bk, nw) for bn in (4, 8, 16, 32) for bk in (128, 256, 512) for nw in (2, 4, 8)]
rows_list = [int(r) for r in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 8]
print(f"{torch.cuda.get_device_name()}  triton {triton.__version__}  torch {torch.__version__}")
for name, n, k in SHAPES:
    router = name.startswith("router")
    w16 = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    w32 = w16.float()  # what SGLang holds for the router today (exact upcast of the checkpoint's BF16)
    out_dtype = torch.float32 if router else torch.bfloat16
    for m in rows_list:
        x = (torch.randn(m, k, device=dev)).to(torch.bfloat16)
        ref = (x.double() @ w16.double().t())
        if router:
            t_blas = timed(lambda: F.linear(x.float(), w32))
            blas_bytes = n * k * 4
            t_roc = timed(lambda: torch.mv(w32, x[0].float())) if m == 1 else timed(lambda: torch.mm(x.float(), w32.t()))
        else:
            t_blas = timed(lambda: F.linear(x, w16))
            blas_bytes = n * k * 2
            t_roc = timed(lambda: torch.mv(w16, x[0])) if m == 1 else timed(lambda: torch.mm(x, w16.t()))
        best = None
        for bn, bk, nw in CONFIGS:
            if bk > max(k, 128):
                continue
            try:
                t = timed(lambda: triton_gemv(x, w16, out_dtype, bn, bk, nw), reps=20)
            except Exception:  # noqa: BLE001 - a config that does not compile is skipped
                continue
            if best is None or t < best[0]:
                best = (t, bn, bk, nw)
        t_tri, bn, bk, nw = best
        t_tri = timed(lambda: triton_gemv(x, w16, out_dtype, bn, bk, nw))
        y1 = triton_gemv(x, w16, out_dtype, bn, bk, nw)
        y2 = triton_gemv(x, w16, out_dtype, bn, bk, nw)
        rel = ((y1.double() - ref).abs().max() / ref.abs().max()).item()
        gbs = lambda b, t: b / (t * 1e3)  # noqa: E731 - bytes / us -> GB/s
        print(f"{name:28s} N={n:5d} K={k:5d} M={m}: hipblaslt {t_blas:7.1f} us ({gbs(blas_bytes, t_blas):4.0f} GB/s)"
              f"  rocblas {t_roc:7.1f} us  triton {t_tri:7.1f} us ({gbs(n * k * 2, t_tri):4.0f} GB/s, BN{bn} BK{bk} "
              f"w{nw})  rel {rel:.1e}  deterministic {torch.equal(y1, y2)}", flush=True)
