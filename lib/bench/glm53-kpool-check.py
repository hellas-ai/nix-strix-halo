#!/usr/bin/env python3
"""Compare pooled DSA ROCm logits to FP64 CPU scores; run in the SGLang environment."""

import json

import torch
from sglang.kernels.ops.attention.dsa import rocm_kpool_mqa as m

torch.manual_seed(1709)
torch.set_num_threads(4)


def ref(q, k, s, w):
    return (
        (
            torch.einsum(
                "qhd,kd->qhk", q.float().cpu().double(), k.float().cpu().double()
            ).relu()
            * w.cpu().double()[..., None]
        ).sum(1)
        * s.cpu().double()
    ).float()


def check(got, want, label):
    got = got.cpu()
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-4)
    mask = torch.isfinite(want)
    print(
        json.dumps(
            {
                "case": label,
                "max_absolute_error": (got[mask] - want[mask]).abs().max().item()
                if mask.any()
                else 0,
            }
        ),
        flush=True,
    )


for h, amplitude in ((16, 2), (32, 2), (48, 2), (64, 2), (32, 100)):
    q = (torch.randn(7, h, 128) * amplitude).clamp(-448, 448).to(torch.float8_e4m3fn)
    k = (torch.randn(259, 128) * amplitude).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = torch.rand(259) * 0.05
    w = torch.randn(7, h) * 0.4 / amplitude**2
    starts = torch.tensor([0, 0, 1, 63, 64, 128, 200], dtype=torch.int32)
    ends = torch.tensor([0, 1, 63, 64, 129, 257, 259], dtype=torch.int32)
    want = ref(q, k, scales, w)
    cols = torch.arange(259)[None, :]
    want.masked_fill_((cols < starts[:, None]) | (cols >= ends[:, None]), -float("inf"))
    for clean in (True, False):
        got = m.fp8_mqa_logits(
            q.cuda(),
            (k.cuda(), scales.cuda()),
            w.cuda(),
            starts.cuda(),
            ends.cuda(),
            clean,
        )
        check(got, want, f"ragged-h{h}-amplitude{amplitude}-clean{clean}")
# Sparse, shuffled physical pages, per-request table strides, mixed context lengths.
b, h, n = 7, 32, 259
q = torch.randn(b, h, 128).to(torch.float8_e4m3fn)
w = torch.randn(b, h) * 0.1
cache = torch.zeros(41, 64 * 132, dtype=torch.uint8)
keys = torch.randn(41, 64, 128).to(torch.float8_e4m3fn)
scales = torch.rand(41, 64) * 0.05
cache[:, : 64 * 128] = keys.view(torch.uint8).reshape(41, -1)
cache[:, 64 * 128 :] = scales.contiguous().view(torch.uint8).reshape(41, -1)
page_table = torch.randperm(41)[: b * 5].reshape(b, 5).int()
ends = torch.tensor([0, 1, 63, 64, 65, 256, 259], dtype=torch.int32)
want = torch.full((b, n), -float("inf"))
for i in range(b):
    k = keys.float()[page_table[i].long()].reshape(-1, 128).to(torch.float8_e4m3fn)[:n]
    s = scales[page_table[i].long()].reshape(-1)[:n]
    want[i] = ref(q[i : i + 1], k, s, w[i : i + 1])[0]
    want[i, int(ends[i]) :] = -float("inf")
# Deliberately make invalid entries negative: masked rows must not dereference them.
for i in range(b):
    page_table[i, (int(ends[i]) + 63) // 64 :] = -1
for clean in (True, False):
    got = m.fp8_paged_mqa_logits(
        q.cuda().unsqueeze(1),
        cache.cuda().view(41, 64, 1, 132),
        w.cuda(),
        ends.cuda().view(-1, 1),
        page_table.cuda(),
        None,
        n,
        clean,
    )
    check(got, want, f"paged-clean{clean}")
print("PASS pooled logits CPU reference", flush=True)
