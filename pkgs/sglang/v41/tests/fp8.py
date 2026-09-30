"""Bounded zero-row/layout and UE8M0 boundary checks for the portable quant path."""

import json
import os
import struct
import sys
from pathlib import Path
from unittest.mock import patch

os.environ["SGLANG_USE_AITER"] = "0"

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.gemm.deepseek_v41_gemv import _e4m3fn_to_fp32
from sglang.kernels.ops.quantization.fp8_kernel import (
    _run_per_token_group_quant_8bit_kernel,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear


@triton.jit
def conversion_probe(codes, old, new):
    offset = tl.arange(0, 256)
    value = tl.load(codes + offset)
    tl.store(old + offset, value.to(tl.float32))
    tl.store(new + offset, _e4m3fn_to_fp32(value))


def expected_conversion_bits(code):
    sign = (code & 128) << 24
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return sign | 0x7FC00000
    value = (
        mantissa * 2.0**-9 if exponent == 0 else (8 + mantissa) * 2.0 ** (exponent - 10)
    )
    return sign | struct.unpack("<I", struct.pack("<f", value))[0]


def check_e4m3fn_conversion():
    # Compare bits, including both NaN signs and negative zero. Floating
    # equality would either reject NaNs or hide signed-zero differences.
    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    codes = raw.cuda().view(torch.float8_e4m3fn)
    old = torch.empty(256, dtype=torch.float32, device="cuda")
    new = torch.empty_like(old)

    def run():
        conversion_probe[(1,)](codes, old, new, num_warps=4, enable_fp_fusion=False)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for case, payload in (("A", raw), ("B", raw.flip(0)), ("A_repeat", raw)):
        codes.view(torch.uint8).copy_(payload)
        graph.replay()
        torch.cuda.synchronize()
        actual = new.view(torch.int32).cpu().tolist()
        original = old.view(torch.int32).cpu().tolist()
        failures = []
        for code, candidate, baseline in zip(payload.tolist(), actual, original):
            literal = expected_conversion_bits(code)
            if len({candidate & 0xFFFFFFFF, baseline & 0xFFFFFFFF, literal}) != 1:
                failures.append(
                    {
                        "code": code,
                        "candidate_bits": candidate & 0xFFFFFFFF,
                        "baseline_bits": baseline & 0xFFFFFFFF,
                        "literal_bits": literal,
                    }
                )
        print(
            json.dumps(
                {
                    "event": "e4m3fn_conversion",
                    "case": case,
                    "codes": 256,
                    "failures": failures,
                }
            ),
            flush=True,
        )
        assert not failures, case
    return 256


@torch.inference_mode()
def check_native_gemv():
    from sglang.kernels.ops.gemm import deepseek_v41_gemv as native
    from sglang.srt.layers.quantization import fp8_utils

    # Two literal nonzero FP8 weights per output row, in different scale
    # groups. All operands and sums are exact dyadic values, so this checks
    # every output without treating the generic WMMA result as an oracle.
    shapes = (
        (25600, 6144, "engram_wkv_c1", True, "engram_gemv"),
        (5120, 2048, "native_wob_c1", True, "native_wob_gemv"),
        (1792, 5120, "native_wqkv_a_c1", True, "native_wqkv_a_gemv"),
        (8192, 1280, "native_wqb_c1", True, "native_wqb_gemv"),
        (1152, 5120, "native_v41_shared", "gate_up", "native_shared_expert_gemv"),
        (5120, 576, "native_v41_shared", "down", "native_shared_expert_gemv"),
    )
    torch.manual_seed(9041)
    for n, k, flag, value, c1_name in shapes:
        rows = torch.arange(n)
        col = rows.remainder(k)
        other = (col + k // 2).remainder(k)
        codes = torch.zeros((n, k), dtype=torch.uint8, device="cuda")
        codes[rows.cuda(), col.cuda()] = 0x38  # E4M3FN +1
        codes[rows.cuda(), other.cuda()] = 0xB8  # E4M3FN -1
        weight = codes.view(torch.float8_e4m3fn)
        exponents = (torch.arange(n // 32)[:, None] + torch.arange(k // 32)) % 3 - 1
        scales_cpu = torch.ldexp(
            torch.ones_like(exponents, dtype=torch.float32), exponents
        )
        scales = scales_cpu.cuda()
        a = (torch.randint(-4, 5, (8, k)).float() / 2).to(torch.bfloat16)
        b = a.flip(0).roll(17, dims=1).contiguous()

        def reference(x, col=col, other=other, scales_cpu=scales_cpu, rows=rows):
            x = x.cpu().float()
            return (
                x[:, col] * scales_cpu[rows // 32, col // 32]
                - x[:, other] * scales_cpu[rows // 32, other // 32]
            ).to(torch.bfloat16)

        def linear(x, weight=weight, scales=scales, **kwargs):
            return triton_w8a8_block_fp8_linear(
                x, weight, [32, 32], scales, act_scale_ue8m0=True, **kwargs
            )

        outputs = {}
        for m in (1, 2, 4, 8):
            wrapper = (
                "native_c4_c8_gemv"
                if m in (4, 8)
                else c1_name
                if m == 1 or flag == "native_v41_shared"
                else "native_c2_gemv"
            )
            x = a[:m].cuda()
            with patch.object(
                native, wrapper, wraps=getattr(native, wrapper)
            ) as selected:
                with patch.object(
                    fp8_utils,
                    "sglang_per_token_group_quant_fp8",
                    wraps=sglang_per_token_group_quant_fp8,
                ) as quantized:
                    actual = linear(x, **{flag: value})
                assert selected.call_count == quantized.call_count == 1
            torch.testing.assert_close(actual.cpu(), reference(a[:m]), rtol=0, atol=0)
            q, qs = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
            torch.testing.assert_close(
                q.float() * qs.repeat_interleave(32, dim=1), x.float(), rtol=0, atol=0
            )
            prequant = linear(q, input_scale=qs, **{flag: value})
            assert torch.equal(actual.view(torch.int16), prequant.view(torch.int16))
            outputs[m] = actual.cpu().clone()
            if m in (4, 8):
                paired = torch.cat(
                    [
                        linear(
                            q[start : start + 2],
                            input_scale=qs[start : start + 2],
                            **{flag: value},
                        )
                        .cpu()
                        .clone()
                        for start in range(0, m, 2)
                    ]
                )
                assert torch.equal(
                    outputs[m].view(torch.int16), paired.view(torch.int16)
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = linear(x, **{flag: value})
            for host in (a[:m], b[:m], a[:m]):
                x.copy_(host)
                graph.replay()
                torch.testing.assert_close(
                    captured.cpu(), reference(host), rtol=0, atol=0
                )
        assert torch.equal(
            outputs[1].view(torch.int16), outputs[2][:1].view(torch.int16)
        )

        # Equal shapes without a model opt-in and prefill sizes keep the
        # original GEMM. Compare identical generic calls, not its rounding
        # order against the specialized reduction's arithmetic reference.
        with patch.object(
            fp8_utils,
            "w8a8_block_fp8_matmul_triton",
            wraps=fp8_utils.w8a8_block_fp8_matmul_triton,
        ) as fallback:
            linear(a[:1].cuda())
            assert fallback.call_count == 1
            for m in (3, 5, 6, 7, 9):
                x = a.repeat(2, 1)[:m].cuda()
                plain = linear(x)
                marked = linear(x, **{flag: value})
                assert torch.equal(plain.view(torch.int16), marked.view(torch.int16))
            assert fallback.call_count == 11
            bias = torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)
            x = a[:4].cuda()
            plain = linear(x, bias=bias)
            marked = linear(x, bias=bias, **{flag: value})
            assert torch.equal(plain.view(torch.int16), marked.view(torch.int16))
            assert fallback.call_count == 13
        print(
            json.dumps(
                {
                    "event": "native_gemv",
                    "shape": [n, k],
                    "c1_c2_c4_c8_literal_exact": True,
                    "c4_c8_native_c2_bitwise": True,
                    "graph_changed_input": True,
                    "unmarked_and_prefill_fallback": True,
                }
            ),
            flush=True,
        )
    return len(shapes)


@torch.inference_mode()
def check_dense_native_rows():
    """Dense exact-dyadic sums exercise every K term independently of native C2.

    Every product is an integer multiple of 1/8 and has magnitude <=8.
    K=2048 bounds every intermediate sum by 16384, below the FP32 exact
    lattice limit; therefore every reduction order has the same exact sum.
    This literal finite-normal case does not establish arbitrary-input parity.
    """
    n, k = 5120, 2048
    torch.manual_seed(2048)
    codes = torch.tensor([0x30, 0x38, 0x40, 0xB8], dtype=torch.uint8)
    values = torch.tensor([0.5, 1, 2, -1], dtype=torch.float64)
    indices = torch.randint(0, 4, (n, k))
    weight = codes[indices].cuda().view(torch.float8_e4m3fn)
    exponents = (torch.arange(n // 32)[:, None] + torch.arange(k // 32)) % 3 - 1
    scales = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
    decoded = values[indices] * scales.double().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    x = (torch.randint(-4, 5, (8, k)).float() / 2).bfloat16()
    expected = (x.double() @ decoded.T).bfloat16()
    wscale = scales.cuda()
    for m in (4, 8):
        inputs = x[:m].cuda()
        q, qs = sglang_per_token_group_quant_fp8(inputs, 32, scale_ue8m0=True)
        torch.testing.assert_close(
            q.float() * qs.repeat_interleave(32, 1), inputs.float(), rtol=0, atol=0
        )
        actual = triton_w8a8_block_fp8_linear(
            inputs, weight, [32, 32], wscale, act_scale_ue8m0=True, native_wob_c1=True
        )
        owned = actual.cpu().clone()
        torch.testing.assert_close(owned, expected[:m], rtol=0, atol=0)
        paired = torch.cat(
            [
                triton_w8a8_block_fp8_linear(
                    q[start : start + 2],
                    weight,
                    [32, 32],
                    wscale,
                    input_scale=qs[start : start + 2],
                    act_scale_ue8m0=True,
                    native_wob_c1=True,
                )
                .cpu()
                .clone()
                for start in range(0, m, 2)
            ]
        )
        assert torch.equal(owned.view(torch.int16), paired.view(torch.int16))
        print(
            json.dumps(
                {
                    "event": "dense_native_exact",
                    "tokens": m,
                    "shape": [n, k],
                    "fp64_exact": True,
                    "native_c2_bitwise": True,
                }
            ),
            flush=True,
        )


def internal(x, q, s):
    return _run_per_token_group_quant_8bit_kernel(
        x,
        q,
        s,
        32,
        1e-10,
        -448.0,
        448.0,
        scale_ue8m0=True,
        fuse_silu_and_mul=False,
        masked_m=None,
    )


def main():
    import inspect

    import sglang

    runtime = Path(sys.argv[1]).resolve()
    assert (
        Path(inspect.getfile(_run_per_token_group_quant_8bit_kernel))
        .resolve()
        .is_relative_to(runtime)
    )
    assert Path(sglang.__file__).resolve().is_relative_to(runtime)
    torch.cuda.set_per_process_memory_fraction(0.03)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    x0 = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    q0, s0 = sglang_per_token_group_quant_fp8(x0, 32, scale_ue8m0=True)
    assert q0.shape == (0, 64) and s0.shape == (0, 2)
    internal(x0, q0, s0)  # raw branch also has an empty-grid guard
    x = torch.zeros((2, 64), device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1.75
    x[1, 31] = -3.5
    q, s = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
    torch.testing.assert_close(
        s.cpu(),
        torch.tensor([[2**-8, 2**-42], [2**-7, 2**-42]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert float(q[0, 0].float()) == 448.0 and float(q[1, 31].float()) == -448.0
    try:
        internal(x, torch.empty_like(q), torch.empty((2, 1), device="cuda"))
    except AssertionError:
        pass
    else:
        raise AssertionError("bad scale shape accepted")
    try:
        internal(x, torch.empty_like(q), torch.empty((2, 4), device="cuda")[:, ::2])
    except AssertionError:
        pass
    else:
        raise AssertionError("noncontiguous scale accepted")
    w = torch.ones((33, 64), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    ws = torch.ones((2, 2), device="cuda", dtype=torch.float32) * 2**-10
    y = triton_w8a8_block_fp8_linear(x, w, [32, 32], ws, act_scale_ue8m0=True)
    assert torch.isfinite(y).all() and y.shape == (2, 33)
    conversion_codes = check_e4m3fn_conversion()
    native_shapes = check_native_gemv()
    check_dense_native_rows()
    print(
        json.dumps(
            {
                "event": "complete",
                "empty_rows": True,
                "power2_boundary": True,
                "bad_layout_rejected": True,
                "native_gemv_shapes": native_shapes,
                "e4m3fn_conversion_codes": conversion_codes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
