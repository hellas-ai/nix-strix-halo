#!/usr/bin/env python3
"""Check GLM's fused ROCm router against stable CPU selection and repeated calls.

Run in the SGLang environment with SGLANG_USE_AITER=0 and
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1. No model weights are required.
The complete quantized MoE is covered separately by glm53-moe-scaling-check.py.
"""

import json
import os
import statistics

import torch
from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu

assert torch.version.hip, "This regression exercises ROCm"
assert os.environ.get("SGLANG_USE_AITER") == "0"
assert os.environ.get("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK") == "1"
torch.set_num_threads(4)
torch.manual_seed(53029)


def check(name, logits, bias, scale_output):
    scores = logits.sigmoid()
    ref_ids = (scores + bias).argsort(dim=-1, descending=True, stable=True)[:, :8]
    ref_weights = scores.gather(1, ref_ids)
    ref_weights /= ref_weights.sum(-1, keepdim=True)
    if scale_output:
        ref_weights *= 2.5
    expected_ids, order = ref_ids.sort(-1)
    expected_weights = ref_weights.gather(1, order)
    gpu_logits, gpu_bias = logits.cuda(), bias.cuda()
    hidden = torch.empty(logits.shape[0], 4096, device="cuda", dtype=torch.bfloat16)

    def route():
        return biased_grouped_topk_gpu(
            hidden, gpu_logits, gpu_bias, 8, True, 1, 1, 0, 2.5, scale_output
        )

    first_weights, first_ids = route()
    first_weights, first_ids = first_weights.cpu(), first_ids.cpu()
    actual_ids, order = first_ids.sort(-1)
    actual_weights = first_weights.gather(1, order)
    assert torch.equal(actual_ids.long(), expected_ids), name
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-7)
    times = []
    for _ in range(20):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        weights, ids = route()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        assert torch.equal(ids.cpu(), first_ids), (name, "expert order changed")
        assert torch.equal(weights.cpu(), first_weights), (name, "weights changed")

    # Decode uses captured graphs. Their results must agree with eager routing.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = route()
    for _ in range(12):
        graph.replay()
        assert torch.equal(graph_ids.cpu(), first_ids), (name, "graph expert order")
        assert torch.equal(graph_weights.cpu(), first_weights), (name, "graph weights")
    print(
        json.dumps(
            {
                "case": name,
                "apply_scale": scale_output,
                "tokens": len(logits),
                "cpu_max_weight_delta": float(
                    (actual_weights - expected_weights).abs().max()
                ),
                "eager_repeats": 20,
                "graph_repeats": 12,
                "bitwise_repeatable": True,
                "median_ms": statistics.median(times),
            }
        ),
        flush=True,
    )


cases = [
    (f"random-{n}", torch.randn(n, 288), torch.randn(288) * 0.02)
    for n in (1, 4, 33, 1024)
]
cases += [
    ("all-ties", torch.zeros(17, 288), torch.zeros(288)),
    (
        "boundary-ties",
        torch.arange(288).remainder(6).float().expand(33, -1).contiguous(),
        torch.zeros(288),
    ),
]
for name, logits, bias in cases:
    for scale_output in (False, True):
        check(name, logits, bias, scale_output)
print(
    "PASS stable expert selection, CPU weights, scaling and eager/graph repeats",
    flush=True,
)
