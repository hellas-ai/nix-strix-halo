"""GPU microbench of patch 0047 (row-generic native FP8 GEMV, mHC post for any row count).  gfx1151, ONE GPU, < 8 GB.

RUN ONLY AFTER the sentinel ../../GPU-FREE-FOR-MICROBENCH exists (run via run.gpu.sh / session1.sh, which also check the node is idle).
Never run this file directly on a busy node.

stages (--stage):
  gemv     for each of the ten table shapes and rows 1..8: us and effective GB/s / % of 242.253 GB/s of
             new   native_gemv_rows (C1 / C2 / C3..C8 kernels, ROWS constexpr)
             old   the pre-patch module's kernels (--baseline-module), rows 1,2,4,8 only: new == old bitwise, same timing
             gen   the generic w8a8_block_fp8_matmul_triton (what rows 3,5,6,7 used before; reference for the gain)
           bitwise: new vs old at 1,2,4,8; a token's row vs its row in the 8-row call (batch invariance 2..8, and 1 vs 2..8);
           optional C1 with 8 output rows per program (SGLANG_DSV41_NATIVE_C1_ROWS=8) vs the 2..8-row kernels.
           Timing is CUDA-graph replay over distinct weight sets (cold weights), per call, graph gaps included.
  blockk   BLOCK_K 256/512/1024 sweep of the draft shapes (+ engram) at rows 1,3,4,6,8: us, % of 242, bitwise vs BLOCK_K=512
           (a different BLOCK_K changes the reduction tree: informational for the draft-only shapes)
  hcpost   native mHC post vs the generic fallback _mhc_post_torch: rows 1..8 and M in 9,17,100,511,513,1000,1306,1537,4096:
           bitwise (random, the 65536/2^-9/-65536/2^-9 order probe, signed zeros), us per call, chunked launch (_MAX_GRID_Z=1000)
           vs the single launch
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import torch
except ImportError:  # --dry-run on a host without torch
    torch = None

STREAM_GBPS = 242.253
# (name, N, K, BLOCK_K): the seven target TP4 projections and the three DSpark draft linears (dsv41_native_rows.PROJECTION_BLOCK_K)
SHAPES = (
    ("engram_wkv", 25600, 6144, 512), ("wo_b", 5120, 2048, 512), ("wqkv_a", 1792, 5120, 512), ("wq_b", 8192, 1280, 256),
    ("shared_gate_up", 1152, 5120, 512), ("shared_down", 5120, 576, 64), ("idx_wq_b", 4096, 1280, 256),
    ("draft_wq_a", 1280, 5120, 512), ("draft_wkv", 512, 5120, 512), ("draft_main_proj", 5120, 15360, 512),
    ("draft_wkv_stacked", 1536, 5120, 512),
)
OLD_SHAPES = {s[0] for s in SHAPES[:7]}
HC_ROWS = tuple(range(1, 9)) + (9, 17, 100, 511, 513, 1000, 1306, 1537, 4096)
BASELINE_DEFAULT = ("/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308/"
                    "lib/python3.13/site-packages/sglang/kernels/ops/gemm/deepseek_v41_gemv.py")
RESULTS = os.path.join(HERE, "results")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("gemv", "blockk", "hcpost"), required=True)
    ap.add_argument("--shapes", default=None, help="comma list of shape names (default all)")
    ap.add_argument("--rows", default=None, help="comma list of row counts (default 1..8)")
    ap.add_argument("--baseline-module", default=BASELINE_DEFAULT)
    ap.add_argument("--set-bytes", type=int, default=512 << 20, help="distinct weight bytes cycled through per shape")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no GPU)")
    return ap.parse_args(argv)


def selected_shapes(args):
    if not args.shapes:
        return SHAPES
    want = set(args.shapes.split(","))
    unknown = want - {s[0] for s in SHAPES}
    if unknown:
        raise SystemExit(f"unknown shapes: {sorted(unknown)}")
    return tuple(s for s in SHAPES if s[0] in want)


def selected_rows(args, default):
    return tuple(int(r) for r in args.rows.split(",")) if args.rows else default


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def med(xs):
    return statistics.median(xs)


def weight_bytes(n, k):
    return n * k + (n // 32) * (k // 32) * 4


def pct(n, k, us):
    return 100.0 * weight_bytes(n, k) / (us * 1e-6) / 1e9 / STREAM_GBPS


def emit(rec, out_lines):
    print(json.dumps(rec), flush=True)
    out_lines.append(rec)


def operands(n, k, nsets, seed):
    import torch
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    g = torch.Generator(device="cuda").manual_seed(seed)
    sets = []
    for s in range(nsets):
        raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
        raw[(raw & 0x7F) == 0x7F] = 0x38                       # finite codes only
        ws = torch.exp2(torch.randint(-9, 1, (n // 32, k // 32), device="cuda", generator=g).float())
        sets.append((raw.view(torch.float8_e4m3fn), ws))
    x = (torch.randn(8, k, device="cuda", generator=g) * 2).to(torch.bfloat16)
    x[:, ::97] *= 8
    q, qs = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
    return sets, q.contiguous(), qs.contiguous()


def bits(t):
    return t.contiguous().view(torch.int16)


def stage_gemv(args):
    import torch
    import benchlib as B
    import sglang.kernels.ops.gemm.deepseek_v41_gemv as NEW
    from sglang.kernels.ops.gemm.dsv41_native_rows import PROJECTION_BLOCK_K
    from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton

    assert {(n, k): bk for _, n, k, bk in SHAPES} == PROJECTION_BLOCK_K, "shape table out of sync with dsv41_native_rows"
    OLD = load_module(args.baseline_module, "gemv_baseline") if os.path.exists(args.baseline_module) else None
    rows_list = selected_rows(args, tuple(range(1, 9)))
    out_lines = []
    for name, n, k, bk in selected_shapes(args):
        nsets = max(2, min(64, args.set_bytes // (n * k)))
        sets, q, qs = operands(n, k, nsets, 1000 + n + k)
        outs_new = {}
        for m in rows_list:
            qm, qsm = q[:m].contiguous(), qs[:m].contiguous()
            n_iter = nsets * 3

            def run_new(i, m=m, qm=qm, qsm=qsm):
                w, ws = sets[i % nsets]
                return NEW.native_gemv_rows(qm, qsm, w, ws)

            res = NEW.native_gemv_rows(qm, qsm, *sets[0]); torch.cuda.synchronize()
            outs_new[m] = res.clone()
            t_new, _ = B.time_graph(run_new, n_iter, reps=args.reps)
            rec = dict(stage="gemv", shape=name, N=n, K=k, rows=m, kernel="c1" if m == 1 else "c2" if m == 2 else f"c3c8_t{m}",
                       us_new=med(t_new), pct_new=pct(n, k, med(t_new)), sets=nsets)

            def run_gen(i, m=m, qm=qm, qsm=qsm):
                w, ws = sets[i % nsets]
                return w8a8_block_fp8_matmul_triton(qm, w, qsm, ws, [32, 32], output_dtype=torch.bfloat16)

            gen = run_gen(0); torch.cuda.synchronize()
            t_gen, _ = B.time_graph(run_gen, n_iter, reps=args.reps)
            rec.update(us_generic=med(t_gen), pct_generic=pct(n, k, med(t_gen)), speedup_vs_generic=med(t_gen) / med(t_new),
                       generic_ne_new_elems=int((bits(gen) != bits(res)).sum()), elems=res.numel())
            if OLD is not None and m in (1, 2, 4, 8) and name in OLD_SHAPES:
                def launch_old(i, m=m, qm=qm, qsm=qsm):
                    w, ws = sets[i % nsets]
                    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                    kw = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
                    if m == 1:
                        OLD._native_fp8_gemv_c1[(n // 4,)](qm, qsm, w, ws, out, n, k, 4, bk, **kw)
                    elif m == 2:
                        OLD._native_c2_gemv[(n // 8,)](qm, qsm, w, ws, out, n, k, 8, bk, **kw)
                    else:
                        OLD._native_c4_c8_gemv[(n // 8,)](qm, qsm, w, ws, out, n, k, m, 8, bk, **kw)
                    return out

                old = launch_old(0); torch.cuda.synchronize()
                t_old, _ = B.time_graph(launch_old, n_iter, reps=args.reps)
                rec.update(us_old=med(t_old), new_ne_old_elems=int((bits(old) != bits(res)).sum()))
            emit(rec, out_lines)
        # batch invariance on this operand set: token t of an m-row call vs the 8-row call (rows >= 2), and vs the 1-row call
        if 8 in outs_new:
            for m, res in sorted(outs_new.items()):
                inv8 = int((bits(res) != bits(outs_new[8][:m])).sum())
                inv1 = int((bits(res[:1]) != bits(outs_new[1])).sum()) if 1 in outs_new else None
                emit(dict(stage="gemv_invariance", shape=name, rows=m, differing_elems_vs_8row_call=inv8,
                          differing_elems_row0_vs_1row_call=inv1, elems=res.numel()), out_lines)
        # one output row tile of 8 for the one-token kernel (SGLANG_DSV41_NATIVE_C1_ROWS=8): bitwise the 2..8-row kernels?
        if 1 in rows_list and 2 in outs_new:
            w, ws = sets[0]
            alt = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
            NEW._native_fp8_gemv_c1[(n // 8,)](q[:1].contiguous(), qs[:1].contiguous(), w, ws, alt, n, k, 8, bk,
                                              num_warps=4, num_stages=1, enable_fp_fusion=False)
            torch.cuda.synchronize()

            def run_c1r8(i):
                w, ws = sets[i % nsets]
                o = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
                NEW._native_fp8_gemv_c1[(n // 8,)](q[:1].contiguous(), qs[:1].contiguous(), w, ws, o, n, k, 8, bk,
                                                  num_warps=4, num_stages=1, enable_fp_fusion=False)
                return o

            t_alt, _ = B.time_graph(run_c1r8, nsets * 3, reps=args.reps)
            emit(dict(stage="gemv_c1_rows8", shape=name, us=med(t_alt), pct=pct(n, k, med(t_alt)),
                      ne_vs_2row_kernel_token0=int((bits(alt) != bits(outs_new[2][:1])).sum()),
                      ne_vs_default_c1=int((bits(alt) != bits(outs_new[1])).sum()), elems=n), out_lines)
        sets = None
        torch.cuda.empty_cache()
    return out_lines


def stage_blockk(args):
    import torch
    import benchlib as B
    import sglang.kernels.ops.gemm.deepseek_v41_gemv as NEW

    names = ("draft_wq_a", "draft_wkv", "draft_main_proj", "engram_wkv")
    shapes = tuple(s for s in selected_shapes(args) if s[0] in names)
    rows_list = selected_rows(args, (1, 3, 4, 6, 8))
    out_lines = []
    for name, n, k, bk0 in shapes:
        nsets = max(2, min(64, args.set_bytes // (n * k)))
        sets, q, qs = operands(n, k, nsets, 2000 + n + k)
        for m in rows_list:
            ref = None
            for bk in (512, 256, 1024):
                if k % bk:
                    continue
                qm, qsm = q[:m].contiguous(), qs[:m].contiguous()

                def run(i, bk=bk, m=m, qm=qm, qsm=qsm):
                    w, ws = sets[i % nsets]
                    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                    NEW._launch_rows(qm, qsm, w, ws, out, n, k, m, bk)
                    return out

                res = run(0); torch.cuda.synchronize()
                if ref is None:
                    ref = res.clone()
                t, _ = B.time_graph(run, nsets * 3, reps=args.reps)
                emit(dict(stage="blockk", shape=name, rows=m, block_k=bk, us=med(t), pct=pct(n, k, med(t)),
                          ne_vs_block_k_512=int((bits(res) != bits(ref)).sum()), elems=res.numel()), out_lines)
        sets = None
        torch.cuda.empty_cache()
    return out_lines


def stage_hcpost(args):
    import torch
    import benchlib as B
    import sglang.kernels.ops.layernorm.native_hc_post as P
    from sglang.kernels.ops.layernorm.mhc import _mhc_post_torch

    H = 5120
    out_lines = []

    def inputs(rows, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = (torch.randn(rows, H, device="cuda", generator=g) * 0.7).bfloat16()
        res = (torch.randn(rows, 4, H, device="cuda", generator=g) * 1.3).bfloat16()
        post = (torch.rand(rows, 4, device="cuda", generator=g) * 2.0).contiguous()
        comb = torch.softmax(torch.randn(rows, 4, 4, device="cuda", generator=g), dim=-1).contiguous()
        return x, res, post, comb

    def probes(rows):
        yield "random", inputs(rows, 11 + rows)
        x = torch.zeros(rows, H, device="cuda", dtype=torch.bfloat16)
        r = torch.zeros(rows, 4, H, device="cuda", dtype=torch.bfloat16)
        for i, v in enumerate((65536.0, 2.0**-9, -65536.0, 2.0**-9)):
            r[:, i] = v
        yield "order_probe", (x, r, torch.zeros(rows, 4, device="cuda"), torch.ones(rows, 4, 4, device="cuda"))
        pattern = torch.arange(H, device="cuda")
        zero = torch.zeros(H, device="cuda", dtype=torch.bfloat16)
        sx = torch.where((pattern & 16) != 0, -zero, zero).expand(rows, -1).contiguous()
        sr = torch.stack([torch.where((pattern & (1 << i)) != 0, -zero, zero) for i in range(4)]).expand(rows, -1, -1).contiguous()
        yield "signed_zeros", (sx, sr, torch.ones(rows, 4, device="cuda"), torch.ones(rows, 4, 4, device="cuda"))

    for rows in selected_rows(args, HC_ROWS):
        bad = {}
        for label, (x, r, p, c) in probes(rows):
            native = P.native_hc_post_c2(x, r, p, c)
            ref = _mhc_post_torch(x, r, p.view(rows, 4, 1), c)
            bad[label] = int((bits(native) != bits(ref)).sum())
        x, r, p, c = inputs(rows, 5)
        n_iter = 8 if rows <= 512 else 3
        t_nat, _ = B.time_graph(lambda i: P.native_hc_post_c2(x, r, p, c), n_iter, reps=args.reps)
        t_ref, _ = B.time_graph(lambda i: _mhc_post_torch(x, r, p.view(rows, 4, 1), c), n_iter, reps=args.reps)
        emit(dict(stage="hcpost", rows=rows, mismatched_elems=bad, total_elems=rows * 4 * H, us_native=med(t_nat), us_torch=med(t_ref),
                  speedup=med(t_ref) / med(t_nat), per_step_ms_if_80_calls=80 * (med(t_ref) - med(t_nat)) / 1000,
                  native_bytes_per_us_gbps=(rows * (H * 2 + 4 * H * 2 + 4 * H * 2 + 16 * 4 + 16)) / (med(t_nat) * 1e-6) / 1e9), out_lines)
        del x, r, p, c
    # chunked launch (rows above the 65535 z-grid limit): force a small chunk and compare with the single launch
    x, r, p, c = inputs(2500, 77)
    single = P.native_hc_post_c2(x, r, p, c)
    keep = P._MAX_GRID_Z
    P._MAX_GRID_Z = 1000
    try:
        chunked = P.native_hc_post_c2(x, r, p, c)
    finally:
        P._MAX_GRID_Z = keep
    torch.cuda.synchronize()
    emit(dict(stage="hcpost_chunked", rows=2500, chunk=1000, differing_elems=int((bits(single) != bits(chunked)).sum())), out_lines)
    return out_lines


def main():
    args = parse()
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, shapes=[s[0] for s in selected_shapes(args)], rows=selected_rows(args, tuple(range(1, 9))),
                              hc_rows=HC_ROWS, baseline_module=args.baseline_module, baseline_exists=os.path.exists(args.baseline_module),
                              set_bytes=args.set_bytes, reps=args.reps)))
        return
    import torch
    import benchlib as B

    B.require_sentinel()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    B.check_budget(8.0)
    lines = {"gemv": stage_gemv, "blockk": stage_blockk, "hcpost": stage_hcpost}[args.stage](args)
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"{args.stage}.json")
    json.dump(lines, open(out, "w"), indent=1)
    print(f"wrote {out} (peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
