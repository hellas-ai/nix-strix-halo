#!/usr/bin/env python3
"""Tile sweep for SGLang's block-FP8 fused_moe_kernel at GLM-5.3-Flash TP4 prefill shapes on gfx1151.

A 1,024-token prefill chunk spends ~46% of its 3.34 s in fused_moe_kernel (18.3 ms per call, 84 calls; torch
profile 2026-10-09), against ~8.8 ms per layer to read all 288 experts' slices once at ~205 GB/s. Per rank:
E=288 experts, hidden K=4096, moe_intermediate 2048/4 = 512, so gate_up is N=1024 x K=4096 and down is N=4096 x
K=512; top-8 routing, block [128, 128] FP8 weights with per-token-group FP8 activations, BF16 output. For M tokens
(default 1024; uniform random routing) this times every config of the sweep through invoke_fused_moe_kernel, as
fused_experts calls it, and prints the best per GEMM next to the shipped config (configs/triton_3_8_0
E=288,N=512,...,block_shape=[128, 128][_down].json) as JSON ready for that file. Run on a free node GPU with the GLM
runtime's sglang-python.

    glm53-prefill-moe-tune.py [M ...]
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
import sglang.srt.layers.moe.moe_runner.triton_utils as triton_utils
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

dev = "cuda"
torch.manual_seed(0)
E, H, I, TOPK = 288, 4096, 512, 8
fp8 = torch.float8_e4m3fn


def timed(fn, reps=10):
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
    return st.median(ts)


w1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(fp8)
w1_s = torch.rand(E, 2 * I // 128, H // 128, device=dev) * 0.01 + 0.001
w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(fp8)
w2_s = torch.rand(E, H // 128, I // 128, device=dev) * 0.01 + 0.001

sweep = [
    {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": g, "num_warps": nw, "num_stages": ns}
    for bm, bn, g, nw, ns in itertools.product((16, 32, 64), (32, 64, 128), (1, 8), (2, 4, 8), (1, 2))
]

for m in [int(a) for a in sys.argv[1:]] or [1024]:
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(m)]).to(torch.int32)
    topk_w = torch.softmax(torch.randn(m, TOPK, device=dev), dim=-1)
    a1 = (torch.randn(m, H, device=dev)).to(fp8)
    a1_s = torch.rand(m, H // 128, device=dev) * 0.05 + 0.01
    a2 = (torch.randn(m * TOPK, I, device=dev)).to(fp8)
    a2_s = torch.rand(m * TOPK, I // 128, device=dev) * 0.05 + 0.01
    c1 = torch.empty(m, TOPK, 2 * I, device=dev, dtype=torch.bfloat16)
    c2 = torch.empty(m, TOPK, H, device=dev, dtype=torch.bfloat16)
    shipped = {}
    cfg_dir = Path(triton_utils.__file__).parent / "configs" / "triton_3_8_0"
    for down in (False, True):
        f = cfg_dir / f"E={E},N={I},device_name=AMD_Radeon_8060S_Graphics,dtype=fp8_w8a8,block_shape=[128, 128]{'_down' if down else ''}.json"
        cfgs = json.loads(f.read_text()) if f.exists() else None
        shipped[down] = cfgs[min(cfgs, key=lambda k: abs(int(k) - m))] if cfgs else None

    def run(cfg, down):
        st_ids, e_ids, n_pad = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], E)
        if not down:
            return lambda: invoke_fused_moe_kernel(
                a1, w1, None, c1, a1_s, w1_s, None, topk_w, topk_ids, st_ids, e_ids, n_pad, False, TOPK, cfg,
                tl.bfloat16, True, False, False, False, False, block_shape=[128, 128])
        return lambda: invoke_fused_moe_kernel(
            a2, w2, None, c2, a2_s, w2_s, None, topk_w, topk_ids, st_ids, e_ids, n_pad, True, 1, cfg,
            tl.bfloat16, True, False, False, False, False, block_shape=[128, 128])

    for down, name in ((False, "gate_up N=1024 K=4096"), (True, "down N=4096 K=512")):
        base = shipped[down]
        t_base = timed(run(base, down)) if base else float("nan")
        results = []
        for cfg in sweep:
            try:
                results.append((timed(run(cfg, down), reps=5), cfg))
            except Exception:  # noqa: BLE001 - configs that do not compile or fit are skipped
                continue
        results.sort(key=lambda r: r[0])
        best_t, best = results[0]
        best_t = timed(run(best, down))
        wbytes = E * (2 * I * H if not down else H * I)
        print(f"M={m} {name}: shipped {t_base:7.2f} ms {json.dumps(base)}")
        print(f"M={m} {name}: best    {best_t:7.2f} ms {json.dumps(best)}  "
              f"({wbytes / best_t / 1e6:.0f} GB/s on all experts' weights)")
        for t, cfg in results[1:5]:
            print(f"          {t:7.2f} ms {json.dumps(cfg)}")
        sys.stdout.flush()
