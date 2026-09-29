#!/usr/bin/env python3
"""Check gfx1151 FP8 widening and GLM projection shapes against CPU references.

Run with this flake's sglang-python on an idle GPU. The FP64 reference uses the
actual dynamically quantized activations, separating GEMM error from quantizer
rounding. This component check does not establish full-model correctness.
"""

import argparse
import json

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.quantization.fp8_utils import widen_e4m3fn_to_fp16
from sglang.srt.layers.quantization.fp8_utils import (
    per_token_group_quant_fp8,
    triton_w8a8_block_fp8_linear,
)


@triton.jit
def widening_kernel(source, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(source + offsets)
    tl.store(output + offsets, widen_e4m3fn_to_fp16(values))


def check_widening():
    source = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    source = source.view(torch.float8_e4m3fn)
    expected = source.half()
    finite = expected.isfinite()
    device_source = source.cuda()
    output = torch.empty(256, dtype=torch.float16, device="cuda")
    widening_kernel[(1,)](device_source, output, 256)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        widening_kernel[(1,)](device_source, output, 256)
    for _ in range(4):
        graph.replay()
        actual = output.cpu()
        assert torch.equal(
            actual[finite].view(torch.int16), expected[finite].view(torch.int16)
        )
        assert torch.equal(actual.isnan(), expected.isnan())
    print(
        json.dumps({"widening": "passed", "finite_patterns": 254, "nan_patterns": 2}),
        flush=True,
    )


def check_dense(rows, n, k, seed):
    torch.manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device="cuda") * 32).to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device="cuda") * 0.001 + 0.001
    quantized, x_scales = per_token_group_quant_fp8(x, 128, column_major_scales=False)
    assert quantized.dtype == weight.dtype == torch.float8_e4m3fn
    x_reference = quantized.float().cpu().double()
    x_reference *= x_scales.cpu().double().repeat_interleave(128, -1)
    weight_reference = weight.float().cpu().double()
    weight_reference *= (
        scales.cpu().double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    )
    expected = (x_reference @ weight_reference.T).bfloat16().float()
    del x_reference, weight_reference
    actual = triton_w8a8_block_fp8_linear(x, weight, [128, 128], scales)
    error = float((actual.float().cpu() - expected).norm() / expected.norm())
    assert error < 0.001, (rows, n, k, seed, error)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = triton_w8a8_block_fp8_linear(x, weight, [128, 128], scales)
    for _ in range(4):
        graph.replay()
        torch.testing.assert_close(replay_output, actual, atol=0, rtol=0)
    print(
        json.dumps(
            {
                "shape": [rows, n, k],
                "seed": seed,
                "relative_l2_to_rounded_fp64": error,
                "graph_exact": True,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--bits-only", action="store_true")
    args = parser.parse_args()
    assert args.seeds > 0
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.025)
    check_widening()
    if not args.bits_only:
        for n, k in [
            (6144, 4096),
            (4096, 3072),
            (2048, 4096),
            (4096, 1536),
            (4096, 4096),
            (1024, 4096),
            (4096, 512),
        ]:
            for rows in (1, 17):
                for seed in range(5359, 5359 + args.seeds):
                    check_dense(rows, n, k, seed)
                    torch.cuda.empty_cache()
    print("FP8_COMPONENT_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
