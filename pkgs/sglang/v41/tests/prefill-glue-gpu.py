#!/usr/bin/env python3
"""GPU test of SGLANG_DSV41_PREFILL_GLUE (0089): the decode module's P and E kernels at prefill row counts give a bit-identical routed MoE output.

Runs the real `fused_experts_mxfp4` (official routed policy, native TP4 geometry: 384 experts, top-6, hidden 5120, per-rank intermediate 576) with the glue
off and on at 17..2047 rows and compares the fp32 outputs bitwise; also compares the two intermediate operands (bf16 quantise-dequantise of the hidden
states and of the SwiGLU activations) against the installed torch sequences, and reports the glue-only times.

    sglang-python prefill-glue-gpu.py            # needs a free GPU (about 3 GB)
"""
import os
import sys

os.environ["SGLANG_DSV41_MXFP4_DECODE"] = "1"
import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe import dsv41_mxfp4_decode as glue
from sglang.srt.layers.moe.moe_runner.triton_utils import mxfp4_moe_amd as moe

dev = "cuda"
torch.manual_seed(0)
E, H, I, TOPK = 384, 5120, 576, 6
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 130, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 130, (E, H, I // 32), dtype=torch.uint8, device=dev)
kw = dict(activation="silu", is_gated=True, inplace=False, no_combine=False, apply_router_weight_on_input=False,
          routed_scaling_factor=1.5, swiglu_limit=10, official_routed_policy=True)


def timed(fn, iters=10):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


ok = True
for m in (17, 64, 128, 1000, 1536, 1537, 2047):
    x = (torch.randn(m, H, device=dev) * torch.exp(torch.randn(m, H // 32, 1, device=dev).repeat_interleave(32, -1).view(m, H))).to(torch.bfloat16)
    x[::5, 3::97] = 0
    pop = torch.rand(E, device=dev) ** 3 + 0.02
    ids = torch.multinomial(pop.expand(m, E), TOPK).to(torch.int32)
    wts = torch.rand(m, TOPK, device=dev, dtype=torch.float32)
    # operand P
    ref_a1 = moe._official_routed_operand(x)
    a1 = glue.prefill_operand(x)
    same_p = torch.equal(ref_a1.view(torch.uint8), a1.view(torch.uint8))
    # operand E against the installed chain on a random gate/up tensor with outliers
    gu = (torch.randn(m, TOPK, 2 * I, device=dev) * 4).to(torch.bfloat16)
    gu[::3, :, ::41] = 30.0
    gu[1::7, :, 5::53] = -30.0
    gate, up = gu.chunk(2, dim=-1)
    act = F.silu(gate.float().clamp(max=10)) * up.float().clamp(min=-10, max=10)
    act = (( wts * 1.5).unsqueeze(-1) * act).to(torch.bfloat16).reshape(m * TOPK, I)
    ref_a2 = moe._official_routed_operand(act)
    a2 = glue.prefill_swiglu(gu.contiguous(), wts, 1.5)
    same_e = torch.equal(ref_a2.view(torch.uint8), a2.view(torch.uint8))
    # whole routed MoE
    glue.PREFILL_GLUE = False
    off = moe.fused_experts_mxfp4(x, w13, w2, wts, ids, s13, s2, **kw)
    glue.PREFILL_GLUE = True
    on = moe.fused_experts_mxfp4(x, w13, w2, wts, ids, s13, s2, **kw)
    same_all = torch.equal(off.view(torch.int32), on.view(torch.int32))
    t_ref = timed(lambda: (moe._official_routed_operand(x), moe._official_routed_operand(act)))
    t_new = timed(lambda: (glue.prefill_operand(x), glue.prefill_swiglu(gu, wts, 1.5)))
    t_off = timed(lambda: (setattr(glue, "PREFILL_GLUE", False), moe.fused_experts_mxfp4(x, w13, w2, wts, ids, s13, s2, **kw))[1])
    t_on = timed(lambda: (setattr(glue, "PREFILL_GLUE", True), moe.fused_experts_mxfp4(x, w13, w2, wts, ids, s13, s2, **kw))[1])
    glue.PREFILL_GLUE = False
    good = same_p and same_e and same_all
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} M={m:5d}  P bitwise {same_p}  E bitwise {same_e}  routed MoE bitwise {same_all}   "
          f"quant operands (installed, 2 calls) {t_ref:.2f} ms -> P+E {t_new:.2f} ms;   fused_experts_mxfp4 {t_off:.2f} -> {t_on:.2f} ms")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
