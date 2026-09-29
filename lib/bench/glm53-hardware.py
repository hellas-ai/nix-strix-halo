#!/usr/bin/env python3
"""Measure local GPU and optional four-rank RCCL baselines without model weights.

Run in the SGLang ROCm Python environment. Set RANK, WORLD_SIZE, MASTER_ADDR,
MASTER_PORT and the NCCL/GLOO interface variables for the collective probe.
Results are JSON lines; failures abort rather than producing timing results.
"""

import argparse
import datetime
import json
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist


def report(kind, **data):
    print(json.dumps({"host": socket.gethostname(), "kind": kind, **data}), flush=True)


def elapsed_ms(fn, repetitions=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def local_probe():
    report("device", torch=torch.__version__, hip=torch.version.hip,
           gpu=str(torch.cuda.get_device_properties(0)), free_total=torch.cuda.mem_get_info())
    torch.manual_seed(53)
    # Check native MoE routing against independent CPU operations. In particular,
    # this exercises the host/device wave-size agreement on RDNA.
    import sgl_kernel

    for tokens in (1, 17, 1024):
        logits = torch.randn(tokens, 288, dtype=torch.float32)
        bias = torch.randn(288, dtype=torch.float32) * 0.01
        for mode in ("softmax", "sigmoid"):
            # These APIs deliberately use different bias conventions: softmax
            # adds bias to logits; sigmoid adds it only to routing scores.
            score = (logits + bias).softmax(-1) if mode == "softmax" else logits.sigmoid()
            selection = score if mode == "softmax" else score + bias
            ids_ref = selection.topk(8, dim=-1).indices.sort(-1).values
            weights_ref = score.gather(1, ids_ref)
            weights_ref /= weights_ref.sum(-1, keepdim=True)
            weights = torch.empty(tokens, 8, device="cuda", dtype=torch.float32)
            ids = torch.empty(tokens, 8, device="cuda", dtype=torch.int32)
            getattr(sgl_kernel, "topk_" + mode)(weights, ids, logits.cuda(),
                                               renormalize=True, correction_bias=bias.cuda())
            sorted_ids, order = ids.cpu().sort(-1)
            torch.testing.assert_close(sorted_ids.long(), ids_ref, rtol=0, atol=0)
            torch.testing.assert_close(weights.cpu().gather(1, order), weights_ref,
                                       rtol=2e-5, atol=2e-6)
    report("native_topk", passed=True, experts=288, top_k=8, token_counts=[1, 17, 1024])

    a, b = torch.randn(128, 128), torch.randn(128, 128)
    ref = a.bfloat16().float() @ b.bfloat16().float()
    actual = (a.cuda().bfloat16() @ b.cuda().bfloat16()).float().cpu()
    torch.testing.assert_close(actual, ref.bfloat16().float(), rtol=0.02, atol=0.125)
    report("bf16_matmul", passed=True)

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import act_and_mul_triton

    gate_up = (torch.randn(128, 1024) * 20).bfloat16()
    gate_up[0, :12] = torch.tensor(
        [-100, -20, -10, -9.9375, -1, 0, 1, 9.9375, 10, 10.0625, 20, 100],
        dtype=torch.bfloat16,
    )
    gate, up = gate_up.chunk(2, -1)
    reference = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    gpu_input = gate_up.cuda()
    for filtered in (False, True):
        result = torch.empty(128, 512, device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros(128, 1, device="cuda", dtype=torch.int32) if filtered else None
        act_and_mul_triton(gpu_input, result, {"BLOCK_SIZE_M": 16},
                          topk_ids=ids, swiglu_limit=10)
        torch.testing.assert_close(result.cpu(), reference, rtol=0.008, atol=1e-5)
    report("swiglu_clamp", passed=True, limit=10)

    # One GiB per array is much larger than the GPU caches. Count both reads
    # and writes, and report useful bytes rather than advertised DRAM speed.
    n = 2**28
    x = torch.full((n,), 1.25, device="cuda", dtype=torch.float32)
    y, z = torch.full_like(x, 2.5), torch.empty_like(x)
    ms = elapsed_ms(lambda: torch.add(x, y, out=z))
    torch.testing.assert_close(z[::65536].cpu(), torch.full_like(z[::65536].cpu(), 3.75))
    report("stream_add", bytes_per_iteration=3*n*4, median_ms=ms,
           useful_GB_s=3*n*4/(ms*1e6))
    del x, y, z
    n = 4096
    a, b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16), torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    ms = elapsed_ms(lambda: torch.mm(a, b, out=out))
    report("bf16_gemm", size=n, median_ms=ms, TFLOP_s=2*n**3/(ms*1e9))
    del a, b, out
    torch.cuda.empty_cache()


def collective_probe():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180),
                            device_id=torch.device("cuda", 0))
    try:
        for n in (1, 4096, 16384, 1048576, 16777216):
            x = torch.full((n,), rank + 1, device="cuda", dtype=torch.float32)
            dist.all_reduce(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(x.cpu(), torch.full((n,), world*(world+1)/2), rtol=0, atol=0)
            # Zero stays finite under repetition. Check nonzero data separately
            # above so overflow cannot corrupt or invalidate the benchmark.
            x.zero_()
            for _ in range(5):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            samples = []
            for _ in range(20):
                started = time.perf_counter()
                dist.all_reduce(x)
                torch.cuda.synchronize()
                samples.append((time.perf_counter()-started)*1000)
            ms = statistics.median(samples)
            report("all_reduce", world=world, bytes=n*4, median_ms=ms,
                   payload_GB_s=n*4/(ms*1e6),
                   ring_bus_GB_s=n*4/(ms*1e6)*2*(world-1)/world,
                   nccl_ib_disable=os.environ.get("NCCL_IB_DISABLE"))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collective-only", action="store_true")
    args = parser.parse_args()
    if not args.collective_only:
        local_probe()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        collective_probe()
