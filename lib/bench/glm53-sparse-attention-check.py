#!/usr/bin/env python3
"""Check DSA prefill/decode, tail columns, and zero-RoPE geometry against CPU."""

import json

import torch
from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)

torch.manual_seed(941)
torch.set_num_threads(4)
for rows, tail_dim, topk in [(3, 0, 3), (3, 0, 2051), (40, 0, 2051), (3, 64, 2051)]:
    heads, dim, slots = 16, 512, 4096
    query = torch.randn(rows, heads, dim + tail_dim).bfloat16()
    cache = torch.randn(slots, 1, dim + tail_dim).bfloat16()
    indices = torch.full((rows, 1, topk), -1, dtype=torch.int32)
    for row in range(rows - 1):
        count = min(topk, 1 + row * 137)
        indices[row, 0, :count] = torch.randperm(slots)[:count].int()
        # Ensure the last three columns have a measurable effect and exercise
        # non-block-aligned split-K reduction. The final row stays fully masked.
        indices[row, 0, -min(3, topk) :] = torch.arange(slots - min(3, topk), slots)
    scale = (dim + tail_dim) ** -0.5
    gathered = cache[:, 0][indices[:, 0].clamp(min=0).long()].double()
    scores = torch.einsum("rhd,rkd->rhk", query.double(), gathered) * scale
    scores.masked_fill_(indices[:, 0, None, :] < 0, -float("inf"))
    probabilities = scores.softmax(-1).nan_to_num(0)
    expected = torch.einsum("rhk,rkd->rhd", probabilities, gathered[..., :dim]).float()
    query_gpu = query.cuda()
    for name, function in [
        ("prefill", triton_sparse_mla_fwd),
        ("decode", triton_sparse_mla_decode_splitk),
    ]:
        result = (
            function(
                q_nope=query_gpu[..., :dim],
                q_rope=query_gpu[..., dim:],
                kv=cache.cuda(),
                indices=indices.cuda(),
                sm_scale=scale,
                d_v=dim,
            )
            .reshape(rows, heads, dim)
            .float()
            .cpu()
        )
        torch.testing.assert_close(result, expected, rtol=0.015, atol=0.008)
        relative_l2 = ((result - expected).norm() / expected.norm()).item()
        assert relative_l2 < 0.006, relative_l2
        assert torch.equal(result[-1], torch.zeros_like(result[-1]))
        print(
            json.dumps(
                {
                    "backend": name,
                    "rows": rows,
                    "rope_dim": tail_dim,
                    "topk": topk,
                    "relative_l2": relative_l2,
                    "max_absolute_error": (result - expected).abs().max().item(),
                }
            ),
            flush=True,
        )
print("PASS sparse attention CPU reference", flush=True)
