#!/usr/bin/env python3
"""Check dense MLP clamping with identity projections and a CPU reference."""

import torch
import torch.nn.functional as F
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP


class Projection(torch.nn.Module):
    def forward(self, x):
        return x, None


m = DeepseekV2MLP.__new__(DeepseekV2MLP)
torch.nn.Module.__init__(m)
m.tp_size = 1
m.swiglu_limit = 10.0
m.use_fused_clamp_act_mul = False
m.gate_up_proj = Projection()
m.down_proj = Projection()
m.down_proj.reduce_results = False
for rows in (1, 7, 128):
    torch.manual_seed(rows)
    x = (torch.randn(rows, 4096) * 20).bfloat16()
    gate, up = x.chunk(2, dim=-1)
    expected = (
        F.silu(gate.float().clamp(max=10)).bfloat16().float()
        * up.float().clamp(-10, 10)
    ).bfloat16()
    got = m(x.cuda()).cpu()
    torch.testing.assert_close(got, expected, rtol=0, atol=0.0009765625)
    print(
        "PASS dense clamp",
        rows,
        (got.float() - expected.float()).abs().max().item(),
        flush=True,
    )
