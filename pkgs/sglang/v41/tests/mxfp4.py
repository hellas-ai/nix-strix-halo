#!/usr/bin/env python3
"""Check packed E2M1/E8M0 experts against a literal CPU FP64 reference."""

import json
import os
import sys
from pathlib import Path

os.environ["SGLANG_USE_AITER"] = "0"

import torch
import torch.nn.functional as F
from sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd import (
    fused_experts_mxfp4,
)


def unpack(packed, scales):
    packed, scales = packed.cpu(), scales.cpu()
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    return values[codes] * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)


def reference(x, w13, w2, scales13, scales2, weights, ids, factor, no_combine):
    active = ids.unique(sorted=True)
    mapping = {expert: i for i, expert in enumerate(active.cpu().tolist())}
    w13, w2 = unpack(w13[active], scales13[active]), unpack(w2[active], scales2[active])
    x, weights, ids = x.cpu(), weights.cpu().double(), ids.cpu()
    rows = []
    for token, routes in enumerate(ids.tolist()):
        outputs = []
        for route, expert in enumerate(routes):
            expert = mapping[expert]
            projected = (x[token].double() @ w13[expert].T).to(torch.bfloat16)
            gate, up = projected.chunk(2)
            activated = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            output = activated.double() @ w2[expert].T
            outputs.append((output * weights[token, route]).to(torch.bfloat16))
        rows.append(torch.stack(outputs))
    output = torch.stack(rows)
    if not no_combine:
        output = output.sum(1)
    if factor is not None:
        output.mul_(factor)
    return output


def main():
    import sglang
    import sgl_kernel
    import inspect
    expected_runtime = Path(sys.argv[1]).resolve()
    assert Path(sglang.__file__).resolve().is_relative_to(expected_runtime)
    assert Path(inspect.getfile(fused_experts_mxfp4)).resolve().is_relative_to(expected_runtime)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1151", props.gcnArchName
    print(json.dumps({"event": "provenance", "runtime": str(expected_runtime),
                      "sgl_kernel": sgl_kernel.__file__, "torch": torch.__version__,
                      "hip": torch.version.hip, "arch": props.gcnArchName}), flush=True)
    torch.manual_seed(734)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.03)
    for tokens, hidden, intermediate, experts in (
        (1, 5120, 576, 8),
        (2, 5120, 576, 8),
        (4, 5120, 576, 8),
        (6, 5120, 576, 8),
        (8, 5120, 576, 8),
        (33, 5120, 576, 8),
        (128, 5120, 576, 8),
        (1, 5120, 576, 384),
        (8, 5120, 576, 384),
    ):
        topk = 6
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        w13 = torch.randint(
            256,
            (experts, 2 * intermediate, hidden // 2),
            dtype=torch.uint8,
            device="cuda",
        )
        w2 = torch.randint(
            256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
        )
        s13 = torch.randint(
            117,
            122,
            (experts, 2 * intermediate, hidden // 32),
            dtype=torch.uint8,
            device="cuda",
        )
        s2 = torch.randint(
            117,
            122,
            (experts, hidden, intermediate // 32),
            dtype=torch.uint8,
            device="cuda",
        )
        choices = (torch.tensor([0, 7, 63, 127, 255, 383], device="cuda")
                   if experts == 384 else torch.arange(experts, device="cuda"))
        ids = torch.stack([
            choices[torch.randperm(len(choices), device="cuda")[:topk]]
            for _ in range(tokens)
        ]).int()
        weights = torch.rand(tokens, topk, device="cuda")
        weights /= weights.sum(-1, keepdim=True)
        for factor, no_combine in ((None, False), (1.5, False), (1.5, True)):
            actual = fused_experts_mxfp4(
                x,
                w13,
                w2,
                weights,
                ids,
                s13,
                s2,
                activation="silu",
                is_gated=True,
                inplace=False,
                no_combine=no_combine,
                apply_router_weight_on_input=False,
                routed_scaling_factor=factor,
                swiglu_limit=10,
            )
            if tokens <= 8 and factor is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = fused_experts_mxfp4(
                        x,
                        w13,
                        w2,
                        weights,
                        ids,
                        s13,
                        s2,
                        activation="silu",
                        is_gated=True,
                        inplace=False,
                        no_combine=False,
                        apply_router_weight_on_input=False,
                        routed_scaling_factor=None,
                        swiglu_limit=10,
                    )
                for _ in range(3):
                    graph.replay()
                    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
            expected = reference(x, w13, w2, s13, s2, weights, ids, factor, no_combine)
            error = (
                actual.cpu().float() - expected.float()
            ).norm() / expected.float().norm()
            assert torch.isfinite(actual).all()
            # FP32 reduction can land on the adjacent BF16 value at a tie.
            # Bound the complete two-GEMM error, including routed scaling.
            assert error < 0.002, float(error)
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=0.03, atol=0.002 * (factor or 1)
            )
            if factor is None:
                unscaled = actual.clone()
            elif not no_combine:
                torch.testing.assert_close(actual, unscaled * factor, rtol=0, atol=0)
            print(
                json.dumps(
                    {
                        "experts": experts,
                        "tokens": tokens,
                        "hidden": hidden,
                        "intermediate": intermediate,
                        "factor": factor,
                        "no_combine": no_combine,
                        "relative_l2": float(error),
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "cases": 27,
                      "max_memory_allocated": torch.cuda.max_memory_allocated()}), flush=True)


if __name__ == "__main__":
    main()
