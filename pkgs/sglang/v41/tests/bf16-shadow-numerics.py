#!/usr/bin/env python3
"""Numerics (and speed) of the bf16 shadow GEMM against the generic FP8 kernel.

For the seven dense FP8 shapes of one TP4 rank, with REAL checkpoint weights
(layer 3 attention + shared expert, layer 1 Engram wkv, layer 2 indexer wq_b,
rank-0 shards) and representative activations, compare

  cur    `_w8a8_block_fp8_matmul`: what prefill runs today (M >= 16)
  shadow `Dsv41Bf16ShadowLinear` (what Fp8LinearMethod installs when
         SGLANG_DSV41_BF16_SHADOW_MIN_M is set): exact bf16 operands, torch's bf16 GEMM
  exact  float64 product of the same exact operands (a subset of rows/columns)

Pre-registered criteria (change only by recording the change in the report):
  C1  shadow weight == exact dequantized weight, bit for bit
  C2  both paths stay inside the rigorous bound |y - exact| <=
      2^-8 (|exact| + g) + g with g = K 2^-24 sum|terms| (FP32 accumulation of
      exact products, then one bf16 rounding)
  C3  rms relative error of shadow vs exact <= 1.25 x that of cur vs exact
  C4  <= 2% of outputs differ in bf16 bits between cur and shadow, and no
      output with |exact| >= 0.1 x row RMS is more than 2 bf16 steps apart (smaller
      outputs are bounded in absolute terms by C2)
  C5  row-batch invariance of the shadow path is REPORTED (fraction of rows whose
      bits change with the batch composition); cur is expected to be 100% invariant
  C6  no NaN/Inf appears that cur does not also produce

    # CPU preview (real weights, modelled accumulation; no GPU):
    sglang-python bf16-shadow-numerics.py RUNTIME --device cpu
    # GPU run (needs GPU-FREE-FOR-MICROBENCH):
    sglang-python bf16-shadow-numerics.py RUNTIME --device cuda [--blas rocblas]
Activations: --acts synthetic (default) or a torch file {shape-name: bf16 [M, K]}
captured from a real prefill.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SENTINEL = Path(
    "/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH"
)
CHECKPOINT = Path("/models/DeepSeek-V4.1-Flash-hf-dba1be0a")

parser = argparse.ArgumentParser()
parser.add_argument("runtime")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--blas", choices=("default", "rocblas"), default="default")
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--ms", default="16,64,128,256,512,1536,3072")
parser.add_argument("--shapes", default="all")
parser.add_argument("--acts", default="synthetic")
parser.add_argument("--exact-rows", type=int, default=48)
parser.add_argument("--exact-cols", type=int, default=768)
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--out", default="")
args = parser.parse_args()
runtime = Path(args.runtime).resolve()
SITE = runtime / "lib/python3.13/site-packages"

if args.device == "cpu":
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
else:
    if not SENTINEL.exists():
        raise SystemExit(f"refusing to touch a GPU: {SENTINEL} does not exist")
    if args.blas == "rocblas":
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["SGLANG_USE_AITER"] = "0"
os.environ["SGLANG_DSV41_BF16_SHADOW_MIN_M"] = "16"

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, str(SITE))
DEVICE = args.device
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.04)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    from sglang.kernels.ops.gemm.fp8_kernel import (  # noqa: E402
        dequant_block_fp8_weight_to_bf16,
        dequant_group32_activation_to_bf16,
        w8a8_block_fp8_matmul_triton,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: E402
        sglang_per_token_group_quant_fp8,
    )
    import functools  # noqa: E402

    from sglang.srt.layers.quantization.fp8_utils import (  # noqa: E402
        Dsv41Bf16ShadowLinear,
        triton_w8a8_block_fp8_linear,
    )

index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
TP = 4


def emit(event, **values):
    line = json.dumps(dict(event=event, **values))
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(line + "\n")


def tensor(name, rows=None, cols=None):
    handle = safe_open(CHECKPOINT / index[name], framework="pt")
    part = handle.get_slice(name)
    shape = part.get_shape()
    r = slice(*rows) if rows else slice(0, shape[0])
    c = slice(*cols) if cols else slice(0, shape[1])
    return part[r, c]


def pair(prefix, rows=None, cols=None):
    """(fp8 weight, fp32 power-of-two scales) for a [rows, cols] slice in element units."""
    rows_s = (rows[0] // 32, rows[1] // 32) if rows else None
    cols_s = (cols[0] // 32, cols[1] // 32) if cols else None
    weight = tensor(prefix + ".weight", rows, cols)
    scale = tensor(prefix + ".scale", rows_s, cols_s).to(torch.float32)
    assert weight.dtype == torch.float8_e4m3fn
    return weight.contiguous(), scale.contiguous()


def shard(total, rank):
    per = total // TP
    return (rank * per, (rank + 1) * per)


def build_shapes(rank):
    out = {}
    wq_a, sq_a = pair("layers.3.attn.wq_a")
    wkv, skv = pair("layers.3.attn.wkv")
    out["wqkv_a (1792,5120)"] = (torch.cat([wq_a, wkv]), torch.cat([sq_a, skv]))
    out["wq_b (8192,1280)"] = pair("layers.3.attn.wq_b", shard(32768, rank))
    out["wo_b (5120,2048)"] = pair("layers.3.attn.wo_b", None, shard(8192, rank))
    inter = shard(2304, rank)
    w1, s1 = pair("layers.3.ffn.shared_experts.w1", inter)
    w3, s3 = pair("layers.3.ffn.shared_experts.w3", inter)
    out["shared gate_up (1152,5120)"] = (torch.cat([w1, w3]), torch.cat([s1, s3]))
    out["shared down (5120,576)"] = pair("layers.3.ffn.shared_experts.w2", None, inter)
    out["engram wkv (25600,6144)"] = pair("layers.1.engram.wkv")
    out["indexer wq_b (4096,1280)"] = pair("layers.2.attn.indexer.wq_b")
    return out


def synthetic_activations(m, k, seed, kind):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=gen)
    if kind == "outlier":
        # LLM-like: lognormal channel scales and a few massive-activation channels.
        x = x * torch.exp(torch.randn(k, generator=gen) * 0.8)
        hot = torch.randperm(k, generator=gen)[: max(1, k // 200)]
        x[:, hot] *= 48.0
    return x.to(torch.bfloat16)


def quantize_cpu(x):
    """Emulate the UE8M0 row-major quantizer: scale = 2^ceil(log2(amax / 448))."""
    m, k = x.shape
    groups = x.to(torch.float32).view(m, k // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(1e-10)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (groups / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(m, k), scale


def dequant_weight_f64(w, s):
    n, k = w.shape
    out = w.to(torch.float64).view(n // 32, 32, k // 32, 32) * s.to(torch.float64).view(
        n // 32, 1, k // 32, 1
    )
    return out.view(n, k)


def emulate_cur(q, xs, w, ws):
    """CPU model of `_w8a8_block_fp8_matmul`: per 32-K tile FP32 dot, scaled, FP32 add."""
    m, k = q.shape
    n = w.shape[0]
    acc = torch.zeros(m, n, dtype=torch.float32)
    qf, wf = q.to(torch.float32), w.to(torch.float32)
    ws_n = ws.repeat_interleave(32, dim=0)  # [N, K/32]
    for t in range(k // 32):
        dot = qf[:, 32 * t : 32 * t + 32] @ wf[:, 32 * t : 32 * t + 32].t()
        acc += dot * xs[:, t : t + 1] * ws_n[:, t].unsqueeze(0)
    return acc.to(torch.bfloat16)


def bf16_ordinal(y):
    bits = y.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def timed(fn, iters):
    if DEVICE == "cpu":
        return float("nan")
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def probe_blas():
    """Which library does torch's bf16 GEMM call? Re-exec with library logging on."""
    code = (
        "import torch;x=torch.randn(256,1280,dtype=torch.bfloat16,device='cuda');"
        "w=torch.randn(4096,1280,dtype=torch.bfloat16,device='cuda');"
        "torch.nn.functional.linear(x,w);torch.cuda.synchronize();"
        "print('preferred', torch.backends.cuda.preferred_blas_library())"
    )
    env = dict(os.environ, HIPBLASLT_LOG_MASK="32", ROCBLAS_LAYER="2", HIPBLASLT_LOG_LEVEL="4")
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    text = done.stdout + done.stderr
    emit(
        "blas_probe",
        env_prefer_hipblaslt=os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
        preferred=[l for l in text.splitlines() if l.startswith("preferred")],
        hipblaslt_lines=sum("hipblaslt" in l.lower() for l in text.splitlines()),
        rocblas_lines=sum("rocblas" in l.lower() for l in text.splitlines()),
        sample=[l[:200] for l in text.splitlines() if "blas" in l.lower()][:6],
    )


results = []
failures = []
ms = [int(v) for v in args.ms.split(",")]
shapes = build_shapes(args.rank)
if args.shapes != "all":
    shapes = {k: v for k, v in shapes.items() if any(s in k for s in args.shapes.split(","))}
captured = torch.load(args.acts) if args.acts != "synthetic" else None

if DEVICE == "cuda":
    probe_blas()
emit(
    "config",
    device=DEVICE,
    blas=args.blas,
    torch=torch.__version__,
    preferred_blas=str(torch.backends.cuda.preferred_blas_library()) if DEVICE == "cuda" else None,
    rank=args.rank,
    ms=ms,
    shapes=list(shapes),
)

shadow_bytes = 0
for name, (w_cpu, s_cpu) in shapes.items():
    n, k = w_cpu.shape
    w_hat64 = dequant_weight_f64(w_cpu, s_cpu)
    # C1: the shadow weight is exactly the dequantized weight.
    if DEVICE == "cuda":
        w, s = w_cpu.to("cuda"), s_cpu.to("cuda")
        shadow_w = dequant_block_fp8_weight_to_bf16(w, s, [32, 32])
        # The per-layer callable Fp8LinearMethod installs: V4.1's UE8M0 partial underneath.
        shadow_linear = Dsv41Bf16ShadowLinear(
            functools.partial(triton_w8a8_block_fp8_linear, act_scale_ue8m0=True), shadow_w, w
        )
        shadow_ok = torch.equal(shadow_w.to(torch.float64).cpu(), w_hat64)
    else:
        w, s = w_cpu, s_cpu
        shadow_w = w_hat64.to(torch.bfloat16)
        shadow_ok = torch.equal(shadow_w.to(torch.float64), w_hat64)
    shadow_bytes += shadow_w.numel() * 2
    emit("weight", shape=name, shadow_exact=shadow_ok, shadow_mb=round(shadow_w.numel() * 2 / 2**20, 1))
    if not shadow_ok:
        failures.append((name, "C1"))

    for m in ms:
        for kind in ("gauss", "outlier"):
            x = (
                captured[name][:m].to(torch.bfloat16)
                if captured is not None
                else synthetic_activations(m, k, 1000 + m, kind)
            )
            if captured is not None and kind == "outlier":
                continue
            xq_cpu, xs_cpu = quantize_cpu(x)
            if DEVICE == "cuda":
                xq, xs = sglang_per_token_group_quant_fp8(x.cuda(), 32, scale_ue8m0=True)
                same_q = bool(torch.equal(xq.view(torch.uint8).cpu(), xq_cpu.view(torch.uint8)))
                same_s = bool(torch.equal(xs.cpu(), xs_cpu))
                cur = w8a8_block_fp8_matmul_triton(
                    xq, w, xs, s, [32, 32], output_dtype=torch.bfloat16
                ).cpu()
                shadow = shadow_linear(
                    xq, w, [32, 32], s, input_scale=xs
                ).cpu()
                t_cur = timed(
                    lambda: w8a8_block_fp8_matmul_triton(
                        xq, w, xs, s, [32, 32], output_dtype=torch.bfloat16
                    ),
                    args.iters,
                )
                t_shadow = timed(
                    lambda: shadow_linear(xq, w, [32, 32], s, input_scale=xs),
                    args.iters,
                )
                x_hat = dequant_group32_activation_to_bf16(xq, xs).cpu().to(torch.float64)
                quant_equal = (same_q, same_s)
            else:
                cur = emulate_cur(xq_cpu, xs_cpu, w_cpu, s_cpu)
                x_hat = (
                    xq_cpu.to(torch.float64).view(m, k // 32, 32) * xs_cpu.to(torch.float64).unsqueeze(-1)
                ).view(m, k)
                shadow = torch.nn.functional.linear(
                    x_hat.to(torch.float32), shadow_w.to(torch.float32)
                ).to(torch.bfloat16)
                t_cur = t_shadow = float("nan")
                quant_equal = None

            # Exact reference on a row/column subset.
            gen = torch.Generator().manual_seed(m * 7 + n)
            rows = torch.randperm(m, generator=gen)[: args.exact_rows]
            cols = torch.randperm(n, generator=gen)[: args.exact_cols]
            xr = x_hat[rows]
            wr = w_hat64[cols]
            exact = xr @ wr.t()
            l1 = xr.abs() @ wr.abs().t()
            gamma = k * 2.0**-24 * l1
            bound = 2.0**-8 * (exact.abs() + gamma) + gamma
            rms_row = exact.pow(2).mean(1, keepdim=True).sqrt()

            def stats(y):
                y64 = y[rows][:, cols].to(torch.float64)
                err = (y64 - exact).abs()
                return dict(
                    excess=float((err - bound).max()),
                    rms_rel=float(err.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()),
                    max_over_row_rms=float((err / rms_row).max()),
                )

            sc, ss = stats(cur), stats(shadow)
            ordinal_gap = (bf16_ordinal(cur) - bf16_ordinal(shadow)).abs()[rows][:, cols]
            large = exact.abs() >= 0.1 * rms_row
            max_gap_large = int(ordinal_gap[large].max()) if bool(large.any()) else 0
            differ = cur.view(torch.int16) != shadow.view(torch.int16)
            diff64 = (cur.to(torch.float64) - shadow.to(torch.float64)).abs()
            row_rms_full = cur.to(torch.float64).pow(2).mean(1, keepdim=True).sqrt()
            finite_cur = bool(torch.isfinite(cur).all())
            finite_new = bool(torch.isfinite(shadow).all())
            record = dict(
                shape=name,
                m=m,
                acts=kind,
                frac_bits_differ=float(differ.float().mean()),
                max_ordinal_steps=int((bf16_ordinal(cur) - bf16_ordinal(shadow)).abs().max()),
                max_ordinal_steps_large_outputs=max_gap_large,
                max_diff_over_row_rms=float((diff64 / row_rms_full).max()),
                cur=sc,
                shadow=ss,
                ms_cur=round(t_cur, 4),
                ms_shadow=round(t_shadow, 4),
                tflops_cur=round(2 * m * n * k / t_cur / 1e9, 2) if t_cur == t_cur else None,
                tflops_shadow=round(2 * m * n * k / t_shadow / 1e9, 2) if t_shadow == t_shadow else None,
                quantizer_matches_cpu_emulation=quant_equal,
                finite=(finite_cur, finite_new),
            )
            emit("case", **record)
            results.append(record)
            if sc["excess"] > 0 or ss["excess"] > 0:
                failures.append((name, m, kind, "C2"))
            if ss["rms_rel"] > 1.25 * sc["rms_rel"] + 1e-9:
                failures.append((name, m, kind, "C3"))
            if record["frac_bits_differ"] > 0.02 or max_gap_large > 2:
                failures.append((name, m, kind, "C4"))
            if finite_cur and not finite_new:
                failures.append((name, m, kind, "C6"))

    # C5: batch invariance. The same 16 rows inside batches of different size.
    probe_rows = 16
    base_x = synthetic_activations(max(ms), k, 99, "outlier")
    invariance = {}
    for label, fn in (("cur", None), ("shadow", None)):
        outs = []
        for m in ms:
            xb = base_x[:m]
            if m < probe_rows:
                continue
            xq_c, xs_c = quantize_cpu(xb)
            if DEVICE == "cuda":
                xq, xs = sglang_per_token_group_quant_fp8(xb.cuda(), 32, scale_ue8m0=True)
                y = (
                    w8a8_block_fp8_matmul_triton(xq, w, xs, s, [32, 32], output_dtype=torch.bfloat16)
                    if label == "cur"
                    else shadow_linear(xq, w, [32, 32], s, input_scale=xs)
                )[:probe_rows].cpu()
            else:
                if label == "cur":
                    y = emulate_cur(xq_c, xs_c, w_cpu, s_cpu)[:probe_rows]
                else:
                    xh = (xq_c.to(torch.float32).view(m, k // 32, 32) * xs_c.unsqueeze(-1)).view(m, k)
                    y = torch.nn.functional.linear(xh[:probe_rows], shadow_w.to(torch.float32)).to(torch.bfloat16)
            outs.append(y)
        ref = outs[0]
        invariance[label] = float(
            sum(int(not torch.equal(o.view(torch.int16), ref.view(torch.int16))) for o in outs[1:])
        ) / max(1, len(outs) - 1)
    emit("batch_invariance", shape=name, fraction_of_batch_sizes_with_different_bits=invariance)

emit(
    "summary",
    shadow_gb=round(shadow_bytes / 2**30, 3),
    cases=len(results),
    failures=[list(f) for f in failures],
    verdict="PASS" if not failures else "FAIL",
)
raise SystemExit(1 if failures else 0)
