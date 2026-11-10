#!/usr/bin/env python3
"""Check packed E2M1/E8M0 experts against a literal CPU FP64 reference."""

import json
import os

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
    w13, w2 = unpack(w13, scales13), unpack(w2, scales2)
    x, weights, ids = x.cpu(), weights.cpu().double(), ids.cpu()
    rows = []
    for token, routes in enumerate(ids.tolist()):
        outputs = []
        for route, expert in enumerate(routes):
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
    torch.manual_seed(734)
    torch.set_num_threads(4)
    for tokens, hidden, intermediate in (
        (1, 256, 256),
        (5, 256, 256),
        (64, 256, 256),
        (1, 4096, 512),
        (5, 4096, 512),
    ):
        experts, topk = 8, 6
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
        ids = torch.stack(
            [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
        ).int()
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
            if tokens in (1, 5) and factor is None:
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


if __name__ == "__main__":
    main()
