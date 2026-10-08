#!/usr/bin/env python3
"""GPU check for the opt-in ROCm Triton RMSNorm (sglang-glm53-rocm patch 0006) against RMSNorm.forward_native.

Run with the GLM runtime's sglang-python on a node's GPU. Hidden 4096 (GLM-5.3) and 512/1536 (MLA norms), rows 1-32,
bf16 activations with bf16 and fp32 weights, with and without a residual. Prints the max absolute difference against
forward_native and the fraction of elements that are not bitwise equal; exits non-zero above 1 bf16 ulp at unit scale.
"""
import os
import sys

os.environ["SGLANG_ROCM_TRITON_RMSNORM"] = "1"
import torch

from sglang.kernels.ops.layernorm.rocm_rmsnorm import rmsnorm
from sglang.srt.layers.layernorm import RMSNorm

torch.manual_seed(0)
worst = 0.0
for hidden in (4096, 1536, 512):
    for wdtype in (torch.bfloat16, torch.float32):
        norm = RMSNorm(hidden, eps=1e-5).cuda()
        with torch.no_grad():
            norm.weight.copy_(torch.randn(hidden) * 0.1 + 1.0)
        norm.weight.data = norm.weight.data.to(wdtype)
        for rows in (1, 2, 4, 8, 32):
            x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 3
            r = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
            ref = norm.forward_native(x.clone())
            got = rmsnorm(x, norm.weight.data, norm.variance_epsilon, x.dtype)
            ref2, rres = norm.forward_native(x.clone(), r.clone())
            got2, gres = rmsnorm(x, norm.weight.data, norm.variance_epsilon, x.dtype, r)
            for a, b in ((ref, got), (ref2, got2), (rres, gres)):
                d = (a.float() - b.float()).abs().max().item()
                worst = max(worst, d)
            ne = (ref2 != got2).float().mean().item()
            print(f"hidden {hidden} weight {wdtype} rows {rows}: max |d| {max((ref - got).abs().max().item(), (ref2 - got2).abs().max().item()):.4g}"
                  f" residual |d| {(rres.float() - gres.float()).abs().max().item():.3g} non-bitwise {ne:.4f}")
print(f"worst max |d| {worst:.4g}")
sys.exit(0 if worst <= 0.0625 else 1)
