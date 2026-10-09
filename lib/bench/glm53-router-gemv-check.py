#!/usr/bin/env python3
"""GLM-5.3-Flash MoE router (N=288 experts, K=4096) logits on gfx1151: today's F.linear(x.float(), W_fp32) against
torch.mv on the FP32 weight and the patch-0014 Triton GEMV on a BF16 copy, hot (weight resident in the 32 MB on-die
cache, as a microbenchmark leaves it) and cold (128 MB streamed between calls, as decode leaves it: each layer's
router is touched once per token between ~150 MB of other weights). Sweeps finer Triton tilings than
glm53-skinny-gemv-check.py. Run on a free node GPU with the GLM runtime's sglang-python.
"""
import importlib.util
import statistics as st
import sys

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("sg", sys.argv[1])
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)
dev = "cuda"
torch.manual_seed(0)
N, K = 288, 4096
w16 = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
w32 = w16.float()
scratch = torch.empty(128 * 2**20 // 4, device=dev, dtype=torch.float32)


def timed(fn, cold, reps=80):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if cold:
            scratch.add_(1.0)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return st.median(ts) * 1e3


for m in (1, 2, 4):
    x = torch.randn(m, K, device=dev).to(torch.bfloat16)
    ref = x.double() @ w16.double().t()
    for cold in (False, True):
        tag = "cold" if cold else "hot "
        t_lin = timed(lambda: F.linear(x.float(), w32), cold)
        t_mv = timed(lambda: torch.mv(w32, x[0].float()), cold) if m == 1 else float("nan")
        best = None
        for bn in (1, 2, 4, 8):
            for bk in (256, 512, 1024, 2048):
                for nw in (1, 2, 4, 8):
                    sg.CONFIGS[(N, K)] = (bn, bk, nw)
                    try:
                        t = timed(lambda: sg.skinny_gemv(x, w16, torch.float32), cold, reps=25)
                    except Exception:  # noqa: BLE001 - configs that do not compile are skipped
                        continue
                    if best is None or t < best[0]:
                        best = (t, bn, bk, nw)
        sg.CONFIGS[(N, K)] = best[1:]
        t_tri = timed(lambda: sg.skinny_gemv(x, w16, torch.float32), cold)
        y = sg.skinny_gemv(x, w16, torch.float32)
        rel = ((y.double() - ref).abs().max() / ref.abs().max()).item()
        same = torch.equal(y, sg.skinny_gemv(x, w16, torch.float32))
        print(f"M={m} {tag}: F.linear fp32 {t_lin:6.1f} us  torch.mv fp32 {t_mv:6.1f} us  triton bf16 {t_tri:6.1f} us "
              f"(BN{best[1]} BK{best[2]} w{best[3]})  rel {rel:.1e}  deterministic {same}", flush=True)
