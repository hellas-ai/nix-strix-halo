"""Indexer wq_b ([1280] -> [4096], block-FP8 32x32) through the native GEMV on gfx1151.

Run on a gfx1151 GPU with the runtime's sglang-python (``sglang-python
indexer-wqb-gemv.py RUNTIME [--source DIR]``).  Uses < 1 GB of GPU memory.

Asserted:
  * literal exact dyadic data (as tests/fp8.py): native_idx_wqb=True gives the exact
    result for 1..8 rows, selects the native entry point and quantizes once;
  * a row's result is bitwise independent of the batch size (1..8) on the dyadic data;
  * changed-input CUDA graph replay;
  * without the flag, or for 9+ rows, the generic matmul is used;
  * SGLANG_DSV41_NATIVE_ROWS=legacy restores the previous dispatch (5, 6, 7 generic, 3 on the padded copy).
Reported (not asserted): bf16 error of both the generic matmul and the native GEMV
against an FP64 evaluation of the same quantized operands, and how often the two
outputs differ, on a realistic weight/activation distribution.
"""

import json
import os
import sys
from unittest.mock import patch

import torch

args = sys.argv[1:]
source = None
if "--source" in args:
    i = args.index("--source")
    source = args[i + 1]
    del args[i : i + 2]
if source:
    sys.path.insert(0, source)

from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4
from sglang.kernels.ops.gemm import deepseek_v41_gemv as native
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear

torch.cuda.set_per_process_memory_fraction(0.05)
assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
assert torch.cuda.mem_get_info()[0] > 3 * 2**30, "not enough free GPU memory"
N, K = 4096, 1280


def linear(x, weight, scales, **kwargs):
    return triton_w8a8_block_fp8_linear(
        x, weight, [32, 32], scales, act_scale_ue8m0=True, **kwargs
    )


def bits(t):
    return t.view(torch.int16)


@torch.inference_mode()
def check_literal():
    rows = torch.arange(N)
    col = rows.remainder(K)
    other = (col + K // 2).remainder(K)
    codes = torch.zeros((N, K), dtype=torch.uint8, device="cuda")
    codes[rows.cuda(), col.cuda()] = 0x38  # E4M3FN +1
    codes[rows.cuda(), other.cuda()] = 0xB8  # E4M3FN -1
    weight = codes.view(torch.float8_e4m3fn)
    exponents = (torch.arange(N // 32)[:, None] + torch.arange(K // 32)) % 3 - 1
    scales_cpu = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
    scales = scales_cpu.cuda()
    torch.manual_seed(4096)
    a = (torch.randint(-4, 5, (8, K)).float() / 2).to(torch.bfloat16)
    b = a.flip(0).roll(17, dims=1).contiguous()

    def reference(x):
        x = x.cpu().float()
        return (
            x[:, col] * scales_cpu[rows // 32, col // 32]
            - x[:, other] * scales_cpu[rows // 32, other // 32]
        ).to(torch.bfloat16)

    outputs = {}
    for m in range(1, 9):
        x = a[:m].cuda()
        with patch.object(
            native, "native_idx_wqb_gemv", wraps=native.native_idx_wqb_gemv
        ) as selected, patch.object(
            fp8_utils,
            "sglang_per_token_group_quant_fp8",
            wraps=sglang_per_token_group_quant_fp8,
        ) as quantized:
            actual = linear(x, weight, scales, native_idx_wqb=True)
        assert selected.call_count == quantized.call_count == 1, m
        torch.testing.assert_close(actual.cpu(), reference(a[:m]), rtol=0, atol=0)
        q, qs = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
        prequant = linear(q, weight, scales, input_scale=qs, native_idx_wqb=True)
        assert torch.equal(bits(actual), bits(prequant))
        outputs[m] = actual.cpu().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = linear(x, weight, scales, native_idx_wqb=True)
        for host in (a[:m], b[:m], a[:m]):
            x.copy_(host)
            graph.replay()
            torch.testing.assert_close(captured.cpu(), reference(host), rtol=0, atol=0)
    assert torch.equal(bits(outputs[1]), bits(outputs[2][:1]))
    assert torch.equal(bits(outputs[2]), bits(outputs[3][:2]))
    assert torch.equal(bits(outputs[3]), bits(outputs[4][:3]))
    assert torch.equal(bits(outputs[4]), bits(outputs[8][:4]))
    for m in (5, 6, 7):
        assert torch.equal(bits(outputs[m - 1]), bits(outputs[m][: m - 1]))
        assert torch.equal(bits(outputs[m]), bits(outputs[m + 1][:m]))
    with patch.object(
        fp8_utils,
        "w8a8_block_fp8_matmul_triton",
        wraps=fp8_utils.w8a8_block_fp8_matmul_triton,
    ) as fallback:
        linear(a[:1].cuda(), weight, scales)  # no flag
        for m in (9, 12, 16):
            x = a.repeat(2, 1)[:m].cuda()
            plain = linear(x, weight, scales)
            marked = linear(x, weight, scales, native_idx_wqb=True)
            assert torch.equal(bits(plain), bits(marked))
        bias = torch.linspace(-1, 1, N, device="cuda", dtype=torch.bfloat16)
        x = a[:4].cuda()
        assert torch.equal(
            bits(linear(x, weight, scales, bias=bias)),
            bits(linear(x, weight, scales, bias=bias, native_idx_wqb=True)),
        )
        assert fallback.call_count == 1 + 6 + 2
    with patch.dict(os.environ, {"SGLANG_DSV41_NATIVE_ROWS": "legacy"}):
        # Kill switch: the previous dispatch (three rows on the padded four-row copy, five/six/seven generic).
        x = a[:3].cuda()
        with patch.object(
            native, "native_idx_wqb_gemv", wraps=native.native_idx_wqb_gemv
        ) as selected:
            actual = linear(x, weight, scales, native_idx_wqb=True)
        assert selected.call_count == 1
        torch.testing.assert_close(actual.cpu(), reference(a[:3]), rtol=0, atol=0)
        x = a[:5].cuda()
        with patch.object(
            native, "native_idx_wqb_gemv", wraps=native.native_idx_wqb_gemv
        ) as selected:
            marked = linear(x, weight, scales, native_idx_wqb=True)
        assert selected.call_count == 0
        assert torch.equal(bits(marked), bits(linear(x, weight, scales)))
    print(
        json.dumps(
            {
                "event": "indexer_wqb_literal",
                "rows": [1, 2, 3, 4, 5, 6, 7, 8],
                "exact": True,
                "row_independent_of_batch": True,
                "graph_changed_input": True,
                "fallbacks": True,
            }
        ),
        flush=True,
    )


def realistic_operands(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    wf = torch.randn(N, K, generator=g) * 0.02
    blk = wf.reshape(N // 32, 32, K // 32, 32)
    amax = blk.abs().amax(dim=(1, 3))
    ws = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    wq = (blk / ws[:, None, :, None]).reshape(N, K).to(torch.float8_e4m3fn)
    return wq, ws


@torch.inference_mode()
def report_error():
    results = []
    for seed in range(4):
        wq, ws = realistic_operands(seed)
        wdeq = (
            wq.float().reshape(N // 32, 32, K // 32, 32) * ws[:, None, :, None]
        ).reshape(N, K).double().cuda()
        wq, ws = wq.cuda(), ws.cuda()
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        # Q-lora rows after RMSNorm: unit-ish scale with a few large channels.
        x = torch.randn(8, K, generator=g) * 2
        x[:, ::97] *= 12
        x = x.to(torch.bfloat16).cuda()
        q, qs = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
        xdeq = (q.float().reshape(8, K // 32, 32) * qs[..., None]).reshape(8, K).double()
        exact = (xdeq @ wdeq.T)  # FP64 evaluation of the quantized operands
        for m in (1, 2, 3, 4, 5, 6, 7, 8):
            generic = linear(q[:m].contiguous(), wq, ws, input_scale=qs[:m].contiguous())
            nat = linear(q[:m].contiguous(), wq, ws, input_scale=qs[:m].contiguous(), native_idx_wqb=True)
            ref_bf16 = exact[:m].float().to(torch.bfloat16)
            ulp = lambda t: (t.float().abs().clamp_min(2**-126).log2().floor().exp2() * 2**-7)
            row = {
                "seed": seed,
                "rows": m,
                "generic_ne_exact_rounded": int((bits(generic) != bits(ref_bf16)).sum()),
                "native_ne_exact_rounded": int((bits(nat) != bits(ref_bf16)).sum()),
                "generic_ne_native": int((bits(generic) != bits(nat)).sum()),
                "elements": generic.numel(),
                "generic_max_abs_err_vs_fp64": float((generic.double() - exact[:m]).abs().max()),
                "native_max_abs_err_vs_fp64": float((nat.double() - exact[:m]).abs().max()),
                "generic_max_err_in_bf16_ulp": float(((generic.double() - exact[:m]).abs() / ulp(ref_bf16).double()).max()),
                "native_max_err_in_bf16_ulp": float(((nat.double() - exact[:m]).abs() / ulp(ref_bf16).double()).max()),
                "max_abs_diff_generic_vs_native": float((generic.float() - nat.float()).abs().max()),
                "max_rel_diff_generic_vs_native": float(
                    ((generic.float() - nat.float()).abs() / generic.float().abs().clamp_min(1e-6)).max()
                ),
                "output_abs_max": float(exact[:m].abs().max()),
            }
            # Consumer view: the indexer RoPEs q and rounds it to FP4 (32-element blocks) before scoring.
            freqs = torch.polar(
                torch.ones(m, 32, device="cuda"),
                torch.rand(m, 32, device="cuda", generator=None) * 6.2831853,
            )
            fq_generic = fake_quant_fp4(rope_tail(generic.view(m, 32, 128), freqs, 64))
            fq_native = fake_quant_fp4(rope_tail(nat.view(m, 32, 128), freqs, 64))
            row["fp4_q_elements_changed_by_native"] = int((fq_generic != fq_native).sum())
            row["fp4_q_elements"] = fq_generic.numel()
            results.append(row)
            print(json.dumps({"event": "indexer_wqb_error", **row}), flush=True)
    return results


if __name__ == "__main__":
    check_literal()
    report_error()
    print(json.dumps({"event": "complete"}), flush=True)
