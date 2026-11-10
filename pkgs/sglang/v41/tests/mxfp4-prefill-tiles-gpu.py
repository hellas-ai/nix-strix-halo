#!/usr/bin/env python3
"""GPU qualification of SGLANG_DSV41_MXFP4_PREFILL_TILES: the opt-in gate/up and down tiles are bit-identical to the guarded tiles.

Runs the real AITER MXFP4 grouped GEMM at the V4.1 TP4 shapes (E=384, top-6, w13 [384,1152,2560], w2 [384,5120,288] packed FP4, E8M0 scales),
with real-looking routing (skewed expert popularity), for several row counts, and compares the outputs bitwise between the config the
production dispatch picks with the flag off and the config it picks with the flag on. Also prints both configs and the kernel times.

    sglang-python mxfp4-prefill-tiles-gpu.py           # needs a free GPU
"""
import sys
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import mxfp4_moe_amd as mod
from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

E, H, I, TOP_K = 384, 5120, 576, 6
dev = "cuda"
torch.manual_seed(0)
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 130, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 130, (E, H, I // 32), dtype=torch.uint8, device=dev)


def timed(fn, iters=8):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def configs(M, flag):
    base = get_optimal_moe_config(torch.bfloat16, use_mxfp4=True, M=M)
    cfg = mod.sparse_moe_config(base, "gfx1151", M, TOP_K, E, H, I)
    mod._PREFILL_TILES = flag
    gate = mod.gate_up_moe_config(cfg, "gfx1151", M, TOP_K, E, H, I)
    down = mod.down_moe_config(cfg, "gfx1151", M, TOP_K, E, H, I)
    return cfg, gate, down


failures = 0
for M in (64, 128, 256, 512, 1024, 1536, 1537, 1792, 2047):
    pop = torch.rand(E, device=dev) ** 3 + 0.02                       # skewed expert popularity
    ids = torch.multinomial(pop.expand(M, E), TOP_K).to(torch.int32)
    wts = torch.rand(M, TOP_K, device=dev, dtype=torch.float32)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    act = torch.randn(M * TOP_K, I, device=dev, dtype=torch.bfloat16)
    res = {}
    for flag in (False, True):
        cfg, gate_cfg, down_cfg = configs(M, flag)
        st, ex, npad = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)
        gu = torch.empty(M, TOP_K, 2 * I, device=dev, dtype=torch.bfloat16)
        dn = torch.empty(M, TOP_K, H, device=dev, dtype=torch.bfloat16)
        run_gu = lambda: mod._run_mxfp4_gemm(x, w13, gu, s13, wts, ids, st, ex, npad, config=gate_cfg, top_k=TOP_K, mul_routed_weight=False)
        run_dn = lambda: mod._run_mxfp4_gemm(act, w2, dn, s2, wts, ids, st, ex, npad, config=down_cfg, top_k=1, mul_routed_weight=False)
        t_gu, t_dn = timed(run_gu), timed(run_dn)
        res[flag] = (gu.clone(), dn.clone(), t_gu, t_dn, gate_cfg, down_cfg)
    same_gu = torch.equal(res[False][0], res[True][0])
    same_dn = torch.equal(res[False][1], res[True][1])
    short = lambda c: f"BM{c['BLOCK_SIZE_M']} BN{c['BLOCK_SIZE_N']} BK{c['BLOCK_SIZE_K']} w{c.get('num_warps')} s{c.get('num_stages')} wpe{c.get('waves_per_eu')}"
    print(f"M={M:5d} gate/up {res[False][2]:6.2f} -> {res[True][2]:6.2f} ms ({short(res[False][4])} -> {short(res[True][4])}) bit-identical={same_gu}"
          f" | down {res[False][3]:5.2f} -> {res[True][3]:5.2f} ms ({short(res[False][5])} -> {short(res[True][5])}) bit-identical={same_dn}")
    failures += (not same_gu) + (not same_dn)
mod._PREFILL_TILES = False
print("PASS" if not failures else f"FAIL ({failures})")
sys.exit(1 if failures else 0)
