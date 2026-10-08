"""GPU microbench + bitwise check of the DSpark draft MoE chain (128 experts, top-3, rank-0 TP4 geometry) vs the installed sequence.

Run ONLY through session_t3.sh (sentinel + idle checks); one GPU, <8 GB.

Per row count (draft rows: 3 at batch 1, 6 at batch 2; 1..16 supported):
  stage-by-stage bitwise (installed AITER stage vs this module's stage, fed with identical inputs):
     G gate/up      AITER fused_moe_mxfp4 (moe_align_block_size, top_k=3)   vs  owner-routed WMMA
     S swiglu       torch bf16 clamp/silu/mul on the kernel's gate/up      vs  _swiglu_bf16
     D down         AITER (top_k=1, MUL_ROUTED_WEIGHT)                      vs  _down_wmma_weighted
     C combine      torch routed_output.sum(dim=1).mul_(factor)            vs  _combine_bf16
  end to end: fused_experts_mxfp4 with SGLANG_DSV41_DRAFT_MOE=0 (installed) vs =1 (this chain through the real hook)
  timing: CUDA-graph replay of both, distinct data per iteration, per layer (x3 stages per draft pass), distinct experts,
          effective GB/s over the unique experts read and % of 242.25 GB/s.
A mismatch is REPORTED, never tolerated; the exit code is 1 if any stage differs.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
os.environ.setdefault("SGLANG_USE_AITER", "0")
import torch

import benchlib as B

E, K1, H, N1, N2, TOPK = 128, 5120, 576, 1152, 5120, 3
STREAM = B.STREAM_GBPS
BYTES_PER_EXPERT = B.GATE_UP_BYTES + B.DOWN_BYTES


def bits(t):
    return t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def mism(a, b):
    return int((bits(a) != bits(b)).sum())


def synth(dev, seed=0, lo=112, hi=124):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w13 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, device=dev, generator=g)
    w2 = torch.randint(0, 256, (E, N2, H // 2), dtype=torch.uint8, device=dev, generator=g)
    s13 = torch.randint(lo, hi + 1, (E, N1, K1 // 32), dtype=torch.uint8, device=dev, generator=g)
    s2 = torch.randint(lo, hi + 1, (E, N2, H // 32), dtype=torch.uint8, device=dev, generator=g)
    return w13, w2, s13, s2


def routes(m, dev, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]).to(torch.int32)
    w = torch.rand(m, TOPK, generator=g) + 0.05
    return ids.to(dev), (w / w.sum(1, keepdim=True)).float().to(dev)


def activations(m, dev, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = torch.randn(m, K1, device=dev, generator=g)
    chan = torch.exp(0.6 * torch.randn(K1, device=dev, generator=g))
    chan[torch.randint(0, K1, (8,), device=dev, generator=g)] *= 25.0
    return (x * chan * 0.8).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16])
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--gu", default=None, help="override BN,BK,warps,stages for gate/up at every row count")
    ap.add_argument("--dn", default=None, help="same for down")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    B.require_sentinel()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    B.check_budget(8.0)
    import sglang.kernels.ops.moe.dsv41_mxfp4_draft as D
    import sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd as moe
    from aiter.ops.triton.utils._triton.arch_info import get_arch
    from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

    for text, table in ((a.gu, D.GATE_UP_CFG), (a.dn, D.DOWN_CFG)):
        if text:
            bn, bk, w_, s_ = [int(v) for v in text.split(",")]
            table.clear()
            table[0] = dict(BLOCK_N=bn, BLOCK_K=bk, num_warps=w_, num_stages=s_)

    dev = "cuda"
    w13, w2, s13, s2 = synth(dev)
    s13e, s2e = s13.view(torch.float8_e8m0fnu), s2.view(torch.float8_e8m0fnu)
    kw = dict(activation="silu", is_gated=True, inplace=False, no_combine=False, apply_router_weight_on_input=False,
              routed_scaling_factor=a.factor, swiglu_limit=10, official_routed_policy=False)

    def installed(x, tw, ids):
        os.environ["SGLANG_DSV41_DRAFT_MOE"] = "0"
        try:
            return moe.fused_experts_mxfp4(x, w13, w2, tw, ids, s13e, s2e, **kw)
        finally:
            os.environ.pop("SGLANG_DSV41_DRAFT_MOE", None)

    def candidate(x, tw, ids):
        os.environ["SGLANG_DSV41_DRAFT_MOE"] = "1"
        return moe.fused_experts_mxfp4(x, w13, w2, tw, ids, s13e, s2e, **kw)

    rows_out, failures = [], []
    for m in a.rows:
        n = a.iters
        rs = [routes(m, dev, 400 + i) for i in range(n)]
        ids_l, w_l = [r[0] for r in rs], [r[1] for r in rs]
        xs = [activations(m, dev, 31 + i) for i in range(n)]
        uniq = sum(int(torch.unique(i).numel()) for i in ids_l) / n

        # ---- stage-by-stage
        cfg0 = get_optimal_moe_config(torch.bfloat16, use_mxfp4=True, M=m)
        cfg0 = moe.sparse_moe_config(cfg0, get_arch(), m, TOPK, E, K1, H)
        gu_cfg = moe.gate_up_moe_config(cfg0, get_arch(), m, TOPK, E, K1, H)
        bad = dict(G=0, S=0, D=0, C=0, end_to_end=0)
        for i in range(n):
            x, tw, ids = xs[i], w_l[i], ids_l[i]
            sorted_ids, expert_ids, post = moe_align_block_size(ids, cfg0["BLOCK_SIZE_M"], E)
            gu_ref = torch.empty((m, TOPK, N1), dtype=torch.bfloat16, device=dev)
            moe._run_mxfp4_gemm(x, w13, gu_ref, s13e, tw, ids, sorted_ids, expert_ids, post, config=gu_cfg, top_k=TOPK, mul_routed_weight=False)
            gu, act, ro, out = D.decode_draft_experts(x, w13, w2, s13e, s2e, tw, ids, a.factor, stages=True)
            bad["G"] += mism(gu_ref, gu)
            gate, up = gu.chunk(2, dim=-1)
            act_ref = (torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)).reshape(m * TOPK, H)
            bad["S"] += mism(act_ref, act)
            ro_ref = torch.empty((m, TOPK, N2), dtype=torch.bfloat16, device=dev)
            moe._run_mxfp4_gemm(act, w2, ro_ref, s2e, tw, ids, sorted_ids, expert_ids, post, config=cfg0, top_k=1, mul_routed_weight=True)
            bad["D"] += mism(ro_ref, ro)
            comb = ro.sum(dim=1)
            comb.mul_(a.factor)
            bad["C"] += mism(comb, out)
            bad["end_to_end"] += mism(installed(x.clone(), tw, ids), candidate(x.clone(), tw, ids))
        torch.cuda.synchronize()

        # ---- timing
        def run_installed(i):
            installed(xs[i], w_l[i], ids_l[i])

        def run_candidate(i):
            candidate(xs[i], w_l[i], ids_l[i])

        t_i, gi = B.time_graph(run_installed, n, reps=a.reps)
        t_c, gc = B.time_graph(run_candidate, n, reps=a.reps)
        med = lambda v: sorted(v)[len(v) // 2]
        r = dict(rows=m, distinct_experts=uniq, installed_us=med(t_i), candidate_us=med(t_c),
                 saving_us_per_layer=med(t_i) - med(t_c), saving_ms_per_draft_pass_est=(med(t_i) - med(t_c)) * 3 / 1000.0,
                 installed_gbps=uniq * BYTES_PER_EXPERT / (med(t_i) * 1e-6) / 1e9,
                 candidate_gbps=uniq * BYTES_PER_EXPERT / (med(t_c) * 1e-6) / 1e9,
                 mismatching_elements=bad)
        r["candidate_pct_of_242"] = 100.0 * r["candidate_gbps"] / STREAM
        r["installed_pct_of_242"] = 100.0 * r["installed_gbps"] / STREAM
        print(json.dumps(r), flush=True)
        rows_out.append(r)
        if any(bad.values()):
            failures.append((m, bad))
    if a.out:
        json.dump(rows_out, open(a.out, "w"), indent=1)
    if failures:
        print("MISMATCHES (reported, not tolerated):", failures)
        sys.exit(1)
    print("all stages bitwise equal to the installed sequence")


if __name__ == "__main__":
    main()
