#!/usr/bin/env python3
"""GPU qualification of the grouped WMMA MoE GEMMs (0100 prefill, 0101 decode): bit-equality with the installed kernels on the dumped real routing,
and kernel times against the memory floor of the experts that routing touches.

Prefill rows (the dumped 1536- and 128-row chunks): gate/up and down are compared bitwise with AITER's `_fused_moe_kernel_mxfp4` in the tile
configuration the production dispatch picks (SGLANG_DSV41_MXFP4_PREFILL_TILES on), stage by stage and through `fused_experts_mxfp4` (fp32 output).
Decode rows (1..16): the chain of `dsv41_mxfp4_decode` with and without SGLANG_DSV41_MOE_GROUPED_DECODE, stage by stage. For 2, 4 and 8 rows the installed
gate/up is an FMA kernel whose values the WMMA kernel does not reproduce bit for bit: those rows report the number of differing elements and the largest
difference in bf16 ulps instead of a failure.

Weights are random packed FP4 with E8M0 scales in [117, 127] (the range of the real checkpoint); activations and routing are the dumped ones.

    sglang-python moe-grouped-gpu.py --dump DIR [--layers 2,20,24,36] [--decode] [--sweep] [--json out.json]     # needs a free GPU (~3 GB)
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config
from sglang.kernels.ops.moe import dsv41_moe_grouped as grouped
from sglang.kernels.ops.moe import dsv41_mxfp4_decode as dec
from sglang.srt.layers.moe.moe_runner.triton_utils import mxfp4_moe_amd as installed
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

E, H, I, TOPK = 384, 5120, 576, 6
GU_BYTES, DN_BYTES = 1152 * (H // 2 + H // 32), H * (I // 2 + I // 32)       # packed weights + scale bytes of one expert
FACTOR = 1.5
dev = "cuda"
FLOORS = (242.0, 226.0)


def timed(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def graph_time(fn, reps=40, inner=4):
    """Median per-call time of `fn` replayed `inner` times inside one CUDA graph (launch gaps of the chain included, host launch cost not)."""
    fn(); fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) / inner)
    return statistics.median(ts)


def gbps(nbytes, ms):
    return nbytes / (ms * 1e6)


def floor_pct(nbytes, ms):
    return [100.0 * (nbytes / (f * 1e6)) / ms for f in FLOORS]


def bf16_diff(a, b):
    """(number of differing elements, largest difference in bf16 ulps of the larger magnitude)."""
    neq = (a != b)
    n = int(neq.sum())
    if n == 0:
        return 0, 0.0
    af, bf = a.float()[neq], b.float()[neq]
    ulp = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-30).log2().floor().exp2() * 2.0 ** -7
    return n, float(((af - bf).abs() / ulp).max())


def make_weights(seed=0):
    torch.manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(117, 128, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
    s2 = torch.randint(117, 128, (E, H, I // 32), dtype=torch.uint8, device=dev)
    return w13, s13, w2, s2


def prefill_case(name, x, ids, tw, w13, s13, w2, s2, iters, results, sweep):
    M = x.shape[0]
    arch = get_arch()
    uniq = int(torch.unique(ids).numel())
    gu_bytes, dn_bytes = uniq * GU_BYTES, uniq * DN_BYTES
    installed._PREFILL_TILES = True
    base = get_optimal_moe_config(torch.bfloat16, use_mxfp4=True, M=M)
    cfg = installed.sparse_moe_config(base, arch, M, TOPK, E, H, I)
    gate_cfg = installed.gate_up_moe_config(cfg, arch, M, TOPK, E, H, I)
    down_cfg = installed.down_moe_config(cfg, arch, M, TOPK, E, H, I)
    st, ex, npad = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)
    a1 = installed._official_routed_operand(x)
    gu_ref = torch.empty(M, TOPK, 2 * I, dtype=torch.bfloat16, device=dev)
    run_gu = lambda: installed._run_mxfp4_gemm(a1, w13, gu_ref, s13, tw, ids, st, ex, npad, config=gate_cfg, top_k=TOPK, mul_routed_weight=False)
    t_gu = timed(run_gu, iters)
    a2 = dec.prefill_swiglu(gu_ref, tw, FACTOR)
    dn_ref = torch.empty(M, TOPK, H, dtype=torch.bfloat16, device=dev)
    run_dn = lambda: installed._run_mxfp4_gemm(a2, w2, dn_ref, s2, tw, ids, st, ex, npad, config=down_cfg, top_k=1, mul_routed_weight=False)
    t_dn = timed(run_dn, iters)
    t_align = timed(lambda: moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E), iters)

    routes = grouped.Routes(ids)
    t_route = timed(lambda: grouped.Routes(ids), iters)
    row = {"case": name, "M": M, "unique_experts": uniq, "installed_gate_up_ms": t_gu, "installed_down_ms": t_dn, "installed_align_ms": t_align,
           "route_ms": t_route}
    cfgs_gu = grouped._GATE_UP_CFGS if sweep else (grouped.cfg_for("gate_up"),)
    cfgs_dn = grouped._DOWN_CFGS if sweep else (grouped.cfg_for("down"),)
    best = {}
    ok = True
    for kind, cfgs, fn_new, ref, shape, nbytes, t_ref in (
        ("gate_up", cfgs_gu, lambda c, out: grouped._gemm(a1, TOPK, w13, s13, routes, out, 2 * I, H, c), gu_ref, (M, TOPK, 2 * I), gu_bytes, t_gu),
        ("down", cfgs_dn, lambda c, out: grouped._gemm(a2, 1, w2, s2, routes, out, H, I, c), dn_ref, (M, TOPK, H), dn_bytes, t_dn),
    ):
        for c in cfgs:
            out = torch.empty(shape, dtype=torch.bfloat16, device=dev)
            try:
                fn_new(c, out)
                torch.cuda.synchronize()
            except Exception as e:  # a variant that cannot launch
                print(f"  {name} {kind} cfg {c}: {e}")
                continue
            same = torch.equal(out, ref)
            t = timed(lambda: fn_new(c, out), iters)
            row[f"{kind}_cfg{c}"] = {"ms": t, "bit_identical": same, "speedup": t_ref / t, "GBps": gbps(nbytes, t), "pct_floor": floor_pct(nbytes, t)}
            print(f"  {name} {kind:7s} cfg {c:2d}: {t:7.3f} ms  installed {t_ref:7.3f}  x{t_ref / t:4.2f}  {gbps(nbytes, t):6.1f} GB/s of experts "
                  f"({floor_pct(nbytes, t)[0]:.0f}% of 242, {floor_pct(nbytes, t)[1]:.0f}% of 226)  bit-identical={same}")
            if not same:
                n, ulp = bf16_diff(out, ref)
                print(f"      {n} of {out.numel()} elements differ, up to {ulp:.2f} bf16 ulp")
                ok = False
            if kind not in best or t < best[kind][0]:
                best[kind] = (t, c)
    # through the production function, flag off then on
    kw = dict(activation="silu", is_gated=True, inplace=False, no_combine=False, apply_router_weight_on_input=False,
              routed_scaling_factor=FACTOR, swiglu_limit=10, official_routed_policy=True)
    saved = grouped._ENABLED
    grouped._ENABLED = False
    o_ref = installed.fused_experts_mxfp4(x, w13, w2, tw, ids, s13, s2, **kw)
    t_full_ref = timed(lambda: installed.fused_experts_mxfp4(x, w13, w2, tw, ids, s13, s2, **kw), iters)
    grouped._ENABLED = True
    o_new = installed.fused_experts_mxfp4(x, w13, w2, tw, ids, s13, s2, **kw)
    t_full_new = timed(lambda: installed.fused_experts_mxfp4(x, w13, w2, tw, ids, s13, s2, **kw), iters)
    grouped._ENABLED = saved
    full_same = torch.equal(o_ref, o_new)
    row["fused_experts"] = {"installed_ms": t_full_ref, "grouped_ms": t_full_new, "bit_identical": full_same}
    print(f"  {name} fused_experts_mxfp4: installed {t_full_ref:.3f} ms, grouped {t_full_new:.3f} ms, fp32 output bit-identical={full_same}")
    ok = ok and full_same
    row["ok"] = ok
    results.append(row)
    return ok


def decode_case(name, x, ids, tw, w13, s13, w2, s2, iters, results):
    m = x.shape[0]
    uniq = int(torch.unique(ids).numel())
    out = {"case": name, "rows": m, "unique_experts": uniq}
    saved = (grouped._DECODE, grouped._DECODE_MODE)
    grouped._DECODE, grouped._DECODE_MODE = False, ""
    s_ref = dec.decode_routed_experts(x, w13, w2, s13, s2, tw, ids, FACTOR, 10, stages=True)
    run_ref = lambda: dec.decode_routed_experts(x, w13, w2, s13, s2, tw, ids, FACTOR, 10)
    t_ref = graph_time(run_ref)
    grouped._DECODE, grouped._DECODE_MODE = True, "1"
    s_new = dec.decode_routed_experts(x, w13, w2, s13, s2, tw, ids, FACTOR, 10, stages=True)
    run_new = lambda: dec.decode_routed_experts(x, w13, w2, s13, s2, tw, ids, FACTOR, 10)
    t_new = graph_time(run_new)
    grouped._DECODE, grouped._DECODE_MODE = saved
    # the chain stages: a1, gu, a2, ro, out
    names = ("a1", "gu", "a2", "ro", "out")
    bad_stage = []
    for n, a, b in zip(names, s_ref, s_new):
        if not torch.equal(a, b):
            bad_stage.append((n, *bf16_diff(a, b)) if a.dtype == torch.bfloat16 else (n, int((a != b).sum()), float((a - b).abs().max())))
    bytes_all = uniq * (GU_BYTES + DN_BYTES)
    out.update(chain_installed_ms=t_ref, chain_grouped_ms=t_new, differing_stages=bad_stage,
               chain_GBps_installed=gbps(bytes_all, t_ref), chain_GBps_grouped=gbps(bytes_all, t_new))
    # the two GEMMs alone, graph replayed
    a1, a2 = s_ref[0], s_ref[2]
    gu = torch.empty(m, TOPK, 2 * I, dtype=torch.bfloat16, device=dev)
    ro = torch.empty(m, TOPK, H, dtype=torch.bfloat16, device=dev)
    pol_live = m in (2, 4, 8)
    run_gu_ref = (lambda: dec._launch_live_rows_gate_up(a1, w13, s13, ids, gu)) if pol_live else (lambda: dec.launch_gate_up_wmma(a1, w13, s13, ids, gu))
    tg_ref = graph_time(run_gu_ref)
    td_ref = graph_time(lambda: dec.launch_down_wmma(a2, w2, s2, ids, ro))
    tg, td = {}, {}
    for c in (1, 9, 11, 16, 17):
        tg[c] = graph_time(lambda: grouped._decode_gemm(a1, TOPK, w13, s13, ids, gu, 2 * I, H, c))
    for c in (13, 6, 14, 15):
        td[c] = graph_time(lambda: grouped._decode_gemm(a2, 1, w2, s2, ids, ro, H, I, c))
    out.update(gate_up_installed_ms=tg_ref, down_installed_ms=td_ref, gate_up_grouped_ms=tg, down_grouped_ms=td)
    bg, bd = min(tg.values()), min(td.values())
    cg, cd = min(tg, key=tg.get), min(td, key=td.get)
    print(f"  {name} rows={m} unique={uniq}: chain {t_ref * 1000:.0f} -> {t_new * 1000:.0f} us"
          f" | gate/up {tg_ref * 1000:.0f} -> {bg * 1000:.0f} us cfg {cg} ({gbps(uniq * GU_BYTES, bg):.0f} GB/s) | down {td_ref * 1000:.0f} -> {bd * 1000:.0f} us cfg {cd} ({gbps(uniq * DN_BYTES, bd):.0f} GB/s)"
          f" | stages differing from installed: {bad_stage or 'none'}")
    results.append(out)
    expected_diff = {"gu", "a2", "ro", "out"} if pol_live else set()
    got = {s[0] for s in bad_stage}
    return got <= expected_diff


def structured_case(ids, w13, s13, w2, s2, results):
    """Layout and dequantisation check free of summation-order effects: activations with three non-zero K positions per row make every output element a
    sum of at most three exact products, so the installed and the grouped kernel must agree bit for bit whatever order the matrix unit sums in."""
    M = ids.shape[0]
    arch = get_arch()
    installed._PREFILL_TILES = True
    g = torch.Generator(device=dev).manual_seed(5)
    ok = True
    for name, K, w, sc, N, a_div in (("gate_up", H, w13, s13, 2 * I, TOPK), ("down", I, w2, s2, H, 1)):
        rows = M if a_div == TOPK else M * TOPK
        a = torch.zeros(rows, K, dtype=torch.bfloat16, device=dev)
        pos = torch.randint(0, K, (rows, 3), device=dev, generator=g)
        val = (torch.randn(rows, 3, device=dev, generator=g) * 4).to(torch.bfloat16)
        a.scatter_(1, pos, val)
        base = get_optimal_moe_config(torch.bfloat16, use_mxfp4=True, M=M)
        cfg = installed.sparse_moe_config(base, arch, M, TOPK, E, H, I)
        gate_cfg = installed.gate_up_moe_config(cfg, arch, M, TOPK, E, H, I)
        down_cfg = installed.down_moe_config(cfg, arch, M, TOPK, E, H, I)
        st, ex, npad = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)
        wts = torch.ones(M, TOPK, device=dev, dtype=torch.float32)
        ref = torch.empty(M, TOPK, N, dtype=torch.bfloat16, device=dev)
        installed._run_mxfp4_gemm(a, w, ref, sc, wts, ids, st, ex, npad, config=gate_cfg if name == "gate_up" else down_cfg,
                                  top_k=TOPK if name == "gate_up" else 1, mul_routed_weight=False)
        routes = grouped.Routes(ids)
        for c in (grouped._GATE_UP_CFGS if name == "gate_up" else grouped._DOWN_CFGS):
            out = torch.empty(M, TOPK, N, dtype=torch.bfloat16, device=dev)
            try:
                grouped._gemm(a, a_div, w, sc, routes, out, N, K, c)
                torch.cuda.synchronize()
            except Exception:
                continue
            same = torch.equal(out, ref)
            ok &= same
            if not same:
                n, ulp = bf16_diff(out, ref)
                print(f"  structured {name} cfg {c}: {n} of {out.numel()} differ (max {ulp:.2f} ulp)  <- layout or dequantisation error")
        print(f"  structured {name}: single-product outputs bit-identical across all variants: {ok}")
    results.append({"case": "structured", "ok": ok})
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layers", default="2,20,24,36")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    w13, s13, w2, s2 = make_weights()
    results, ok = [], True
    torch.manual_seed(1)
    for layer in (int(v) for v in a.layers.split(",")):
        r = torch.load(Path(a.dump) / f"moe-L{layer:02d}.pt")
        ids = r["topk_ids"].to(dev).to(torch.int32).contiguous()
        tw = r["topk_weights"].to(dev).float().contiguous()
        M = ids.shape[0]
        x = r["hidden_states"].to(dev).to(torch.bfloat16).contiguous() if "hidden_states" in r else torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        print(f"layer {layer}: M={M} unique experts {int(torch.unique(ids).numel())}")
        if layer == int(a.layers.split(",")[0]):
            ok &= structured_case(ids[:256].contiguous(), w13, s13, w2, s2, results)
        ok &= prefill_case(f"L{layer:02d}", x, ids, tw, w13, s13, w2, s2, a.iters, results, a.sweep)
        if M == 1536:   # the merged tail chunk: more rows than a full chunk, routing cycled
            for rows in (1792, 2047):
                reps = (rows + M - 1) // M
                ok &= prefill_case(f"L{layer:02d}x{rows}", x.repeat(reps, 1)[:rows].contiguous(), ids.repeat(reps, 1)[:rows].contiguous(),
                                   tw.repeat(reps, 1)[:rows].contiguous(), w13, s13, w2, s2, a.iters, results, a.sweep)
    if a.decode:
        print("decode rows")
        r = torch.load(Path(a.dump) / "moe-L02.pt")
        ids_all = r["topk_ids"].to(dev).to(torch.int32)
        tw_all = r["topk_weights"].to(dev).float()
        x_all = r["hidden_states"].to(dev).to(torch.bfloat16)
        for m in (1, 2, 3, 4, 5, 8, 12, 16):
            for start, label in ((0, "real0"), (700, "real700")):
                sl = slice(start, start + m)
                ok &= decode_case(f"{label}", x_all[sl].contiguous(), ids_all[sl].contiguous(), tw_all[sl].contiguous(), w13, s13, w2, s2, a.iters, results)
            pop = torch.rand(E, device=dev) ** 2 + 0.05
            ids_r = torch.multinomial(pop.expand(m, E), TOPK).to(torch.int32).contiguous()
            ok &= decode_case("random", torch.randn(m, H, device=dev, dtype=torch.bfloat16), ids_r, torch.rand(m, TOPK, device=dev), w13, s13, w2, s2, a.iters, results)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1, default=str))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
