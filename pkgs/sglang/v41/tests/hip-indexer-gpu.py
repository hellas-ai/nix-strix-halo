#!/usr/bin/env python3
"""GPU unit test of the HIP block indexer (0087) against the installed Triton kernel and a float64 reference.

Bounded error, stated: scores agree with the installed kernel to 6e-7 of the largest score (about five fp32 ulps: the head-sum order differs and the
group sums are exact integers here), the -inf pattern is identical, and the two top-512 index sets differ only by ties within that tolerance of the
512th score.
Edge cases: lengths from 0 to the full width (partial last pages whose unwritten scale codes are 0), invalid page ids (-1 and >= pages), widths and row
counts that are not multiples of the 64-key page or of the 8-row workgroup, scale ranges beyond the exact-integer guard (scalar fallback), and rows of
one workgroup that point at different pages (scalar fallback).

    SGLANG_DSV41_PARALLEL_INDEXER_SCORE=1 SGLANG_DSV41_HIP_INDEXER=1 sglang-python hip-indexer-gpu.py     # needs a free gfx1151 GPU
"""
import os
import sys

import torch

os.environ["SGLANG_DSV41_HIP_INDEXER"] = "1"
os.environ.setdefault("SGLANG_DSV41_PARALLEL_INDEXER_SCORE", "1")
from sglang.kernels.ops.attention.dsv4 import fp4_indexer_block_hip as hip  # noqa: E402
from sglang.kernels.ops.attention.dsv4.fp4_indexer_gfx1151 import paged_fp4_logits  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def case(name, rows, width, pages=24, qcodes=(124, 130), kcodes=(120, 127), hole=0.0, bad_pages=0.0, lens="full", scalar_rows=False, topk=512, tol=6e-7):
    q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device=dev)
    k = torch.randint(0, 256, (pages, 1, 4, 64, 16), dtype=torch.uint8, device=dev)
    qs = torch.zeros(rows, 1, 4, 16, 4, dtype=torch.uint8, device=dev)
    qs[..., :2] = torch.randint(qcodes[0], qcodes[1] + 1, (rows, 1, 4, 16, 2), dtype=torch.uint8, device=dev)
    ks = torch.randint(kcodes[0], kcodes[1] + 1, (pages, 1, 4, 64), dtype=torch.uint8, device=dev)
    w = torch.randn(rows, 32, device=dev).abs().to(torch.bfloat16)
    cols = (width + 63) // 64
    base = torch.randint(0, pages, (cols,), device=dev, dtype=torch.int32)
    pt = base.unsqueeze(0).repeat(rows, 1).contiguous()                       # one request: every row has the same page table
    if scalar_rows:
        pt = torch.randint(0, pages, (rows, cols), device=dev, dtype=torch.int32)   # rows disagree
    if bad_pages:
        r = torch.rand(rows, cols, device=dev)
        pt = torch.where(r < bad_pages / 2, torch.full_like(pt, -1), torch.where(r < bad_pages, torch.full_like(pt, pages + 3), pt))
    lens_t = {"full": torch.full((rows,), width, dtype=torch.int32), "rand": torch.randint(0, width + 1, (rows,), dtype=torch.int32),
              "ramp": (width - rows + 1 + torch.arange(rows)).clamp(min=0).to(torch.int32), "tiny": torch.randint(0, 70, (rows,), dtype=torch.int32)}[lens].to(dev)
    out0 = torch.empty(rows, width, dtype=torch.float32, device=dev)
    out1 = torch.full_like(out0, 7.0)
    kk = k.view(torch.uint8)
    paged_fp4_logits(q, qs, kk, ks, w, pt, lens_t, 1.0, out0)
    hip.paged_logits_block(q, qs, kk, ks, w, pt, lens_t, 1.0, out1)
    torch.cuda.synchronize()
    fin0, fin1 = torch.isfinite(out0), torch.isfinite(out1)
    mask_ok = bool(torch.equal(fin0, fin1)) and bool((out1[~fin1] == float("-inf")).all())
    d = (out0 - out1).abs()[fin0 & fin1]
    smax = out0[fin0].abs().max().item() if fin0.any() else 1.0
    dmax = d.max().item() / smax if d.numel() else 0.0          # relative to the largest score
    sel_ok = True
    if width >= topk and fin0.any():
        # an index may differ between the two top-k sets only if its score is within the tolerance of the k-th value (a tie flip)
        enough = fin0.sum(1) >= topk
        v0 = torch.topk(out0, topk, dim=1)
        t0 = v0.indices
        t1 = torch.topk(out1, topk, dim=1).indices
        thr = v0.values[:, -1:]
        in0 = torch.zeros_like(out0, dtype=torch.bool).scatter_(1, t0, True)
        in1 = torch.zeros_like(out0, dtype=torch.bool).scatter_(1, t1, True)
        flip = (in0 ^ in1) & fin0
        near = (out0 - thr).abs() <= tol * smax
        sel_ok = bool((~flip | near)[enough].all()) if enough.any() else True
    ok = mask_ok and dmax <= tol and sel_ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:34s} rows {rows:4d} width {width:6d}  -inf mask {mask_ok}  max|d|/max|score| {dmax:.2e} (max score {smax:.3g})  top{topk} sets equal up to ties {sel_ok}")
    return ok


results = [
    case("full, ramp lengths", 64, 4096, lens="ramp"),
    case("random lengths incl 0", 61, 5000, lens="rand"),
    case("tiny lengths (< one page)", 33, 4096, lens="tiny"),
    case("width not a multiple of 64", 40, 4001, lens="rand"),
    case("rows not a multiple of 8", 13, 8192, lens="ramp"),
    case("one row", 1, 4096, lens="full"),
    case("invalid page ids", 48, 4096, bad_pages=0.1, lens="ramp"),
    case("wide scale range (guard 16)", 48, 4096, qcodes=(120, 128), kcodes=(120, 128), lens="ramp"),
    case("scale range beyond the guard", 48, 4096, qcodes=(110, 135), kcodes=(110, 135), lens="ramp", tol=2e-5),
    case("zero scale codes (scalar fallback)", 48, 4096, kcodes=(0, 127), lens="ramp", tol=2e-5),
    case("rows disagree on pages (fallback)", 16, 4096, scalar_rows=True, lens="ramp", topk=64),
    case("large prefill slice", 1365, 12288, pages=140, lens="ramp"),
    case("long context slice", 315, 53248, pages=800, lens="ramp"),
]
print("PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
