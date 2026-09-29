#!/usr/bin/env python3
"""Check pooled index rotation, selection, mapping and live tails against CPU."""

import math

import torch
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)

torch.manual_seed(732)
torch.set_num_threads(4)
hadamard = torch.ones(1, 1, dtype=torch.float64)
while hadamard.shape[0] < 128:
    hadamard = torch.cat(
        (torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0
    )
for dtype in (torch.bfloat16, torch.float32):
    for rows in (0, 1, 97):
        x = torch.randn(rows, 256).to(dtype)
        expected = ((x[:, ::2].double() @ hadamard) / math.sqrt(128)).to(dtype)
        result = rotate_activation(x.cuda()[:, ::2]).cpu()
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=1e-6)
        print("PASS rotation", dtype, rows, flush=True)

lengths = torch.tensor([0, 1, 511, 512, 513, 650], dtype=torch.int32)
starts = torch.tensor([0, 3, 7, 11, 13, 17], dtype=torch.int32)
seq_lens = lengths * 4 + torch.tensor([0, 1, 2, 3, 0, 3], dtype=torch.int32)
scores = torch.full((6, 800), 1000.0)
for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
    scores[row, start : start + length] = (torch.randn(length) * 4).round() / 4
page_table = torch.stack([torch.randperm(4096) + i * 4096 for i in range(6)]).int()
page_rows = torch.tensor([2, 0, 1, 1, 2, 0], dtype=torch.int32)
offsets = torch.arange(6, dtype=torch.int32) * 4096
for mode in ("logical", "offset", "page", "page-row-index"):
    expected = torch.full((8, 2051), -1, dtype=torch.int32)
    for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
        groups = sorted(
            sorted(range(length), key=lambda i: (-float(scores[row, start + i]), i))[
                :512
            ]
        )
        tokens = [group * 4 + slot for group in groups for slot in range(4)]
        tokens += list(range(length * 4, int(seq_lens[row])))
        if mode == "offset":
            tokens = [token + int(offsets[row]) for token in tokens]
        elif mode.startswith("page"):
            mapped_row = int(page_rows[row]) if mode == "page-row-index" else row
            tokens = [int(page_table[mapped_row, token]) for token in tokens]
        expected[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    result = topk_from_pooled_history_logits(
        scores.cuda(),
        lengths.cuda(),
        pool_size=4,
        topk=2048,
        page_table=page_table.cuda() if mode.startswith("page") else None,
        page_table_row_index=page_rows.cuda() if mode == "page-row-index" else None,
        topk_offsets=offsets.cuda() if mode == "offset" else None,
        seq_lens=seq_lens.cuda(),
        row_starts=starts.cuda(),
        out_rows=8,
    ).cpu()
    assert torch.equal(result, expected), mode
    print("PASS pooled selection", mode, flush=True)
expected = torch.full((4, 2051), -1, dtype=torch.int32)
for row in range(4):
    expected[row, :row] = torch.arange(row, dtype=torch.int32)
result = topk_from_pooled_history_logits(
    torch.empty(4, 0, device="cuda"),
    torch.zeros(4, device="cuda", dtype=torch.int32),
    4,
    2048,
    seq_lens=torch.arange(4, device="cuda", dtype=torch.int32),
).cpu()
assert torch.equal(result, expected)
print("PASS empty history with live tail", flush=True)
