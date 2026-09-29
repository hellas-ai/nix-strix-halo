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
import sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd as moe


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


def bf16_bits_equal(left, right):
    return torch.equal(
        left.contiguous().view(torch.int16), right.contiguous().view(torch.int16)
    )


def verify_stage1_ties(x, w13, w2, s13, s2, weights, ids, actual_routes, bad):
    """Independently prove each route exception comes from a BF16 midpoint.

    FP32 accumulation can choose the neighboring BF16 value at a near tie.
    The exception is accepted only when the exact FP64 stage-1 dot lies within
    0.001 BF16 steps of that midpoint, and the remaining computation agrees
    bitwise with CPU arithmetic from the GPU's actual BF16 stage-1 result.
    """
    tokens, topk = ids.shape
    intermediate = w13.shape[1] // 2
    captured = []
    original = moe._run_mxfp4_gemm

    def spy(*args, **kwargs):
        if captured:
            captured.append(args[0].detach().cpu().clone().reshape(
                tokens, topk, intermediate
            ))
        original(*args, **kwargs)
        captured.append(args[2].detach().cpu().clone())

    moe._run_mxfp4_gemm = spy
    try:
        repeated = fused_experts_mxfp4(
            x, w13, w2, weights, ids, s13, s2,
            activation="silu", is_gated=True, inplace=False, no_combine=True,
            apply_router_weight_on_input=False, routed_scaling_factor=None,
            swiglu_limit=10,
        )
    finally:
        moe._run_mxfp4_gemm = original
    assert len(captured) == 3
    projected, activated, routed = captured
    assert bf16_bits_equal(repeated.cpu(), actual_routes)
    assert bf16_bits_equal(routed, actual_routes)

    x_cpu, ids_cpu, weights_cpu = x.cpu(), ids.cpu(), weights.cpu()
    bad_by_route = {}
    for token, route, feature in bad.tolist():
        bad_by_route.setdefault((token, route), []).append(feature)
    for (token, route), features in bad_by_route.items():
        expert = int(ids_cpu[token, route])
        decoded13 = unpack(w13[expert:expert + 1], s13[expert:expert + 1])[0]
        exact_projection = x_cpu[token].double() @ decoded13.T
        reference_projection = exact_projection.to(torch.bfloat16)
        gpu_projection = projected[token, route]
        different = (gpu_projection != reference_projection).nonzero().flatten()
        assert len(different), (token, route, "no stage-1 difference")
        for column in different.tolist():
            gpu_value = float(gpu_projection[column])
            reference_value = float(reference_projection[column])
            bit_gap = abs(int(gpu_projection[column].view(torch.int16)) -
                          int(reference_projection[column].view(torch.int16)))
            midpoint = (gpu_value + reference_value) / 2
            distance_in_steps = abs(float(exact_projection[column]) - midpoint) / abs(
                gpu_value - reference_value
            )
            assert bit_gap == 1 and distance_in_steps <= 0.001, (
                token, route, column, bit_gap, distance_in_steps
            )
        gate, up = gpu_projection.chunk(2)
        cpu_activation = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        assert bf16_bits_equal(cpu_activation, activated[token, route])
        decoded2 = unpack(w2[expert:expert + 1], s2[expert:expert + 1])[0]
        for feature in features:
            exact_output = (activated[token, route].double() @ decoded2[feature]) * (
                weights_cpu[token, route].double()
            )
            expected_from_gpu_stage1 = exact_output.to(torch.bfloat16)
            assert bf16_bits_equal(expected_from_gpu_stage1,
                                   routed[token, route, feature]), (
                token, route, feature, "stage-2 mismatch"
            )
    return set(bad_by_route)


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
        (255, 5120, 576, 8),
        (256, 5120, 576, 8),
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
        # Dequantize and run the literal FP64 oracle once per fixture. The
        # other expected outputs differ only by BF16 route sum or scaling.
        expected_routes = reference(
            x, w13, w2, s13, s2, weights, ids, None, True
        )
        expected_combined = expected_routes.sum(1)
        actuals = {}
        stage1_tie_routes = set()
        for factor, no_combine in (
            (None, False), (None, True), (1.5, False), (1.5, True)
        ):
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
                        no_combine=no_combine,
                        apply_router_weight_on_input=False,
                        routed_scaling_factor=None,
                        swiglu_limit=10,
                    )
                for _ in range(3):
                    graph.replay()
                    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
            expected = (
                expected_routes if no_combine else expected_combined
            ).clone()
            if factor is not None:
                expected.mul_(factor)
            error = (
                actual.cpu().float() - expected.float()
            ).norm() / expected.float().norm()
            assert torch.isfinite(actual).all()
            # FP32 accumulation can round one stage-1 BF16 value across a
            # midpoint relative to the FP64 oracle. A single changed route
            # can then exceed the combined per-element tolerance when other
            # routes cancel. Bound the full output error globally and each
            # route individually at the existing tolerance instead.
            assert error < 0.002, float(error)
            if no_combine:
                bad = (~torch.isclose(
                    actual.cpu(), expected, rtol=0.03,
                    atol=0.002 * (factor or 1),
                )).nonzero()
                if len(bad):
                    if factor is None:
                        stage1_tie_routes = verify_stage1_ties(
                            x, w13, w2, s13, s2, weights, ids,
                            actual.cpu(), bad,
                        )
                    else:
                        # Prove even scaled-only features against the
                        # unscaled routed output, then check scaling exactly
                        # below. Reusing only the route ID would miss these.
                        scaled_tie_routes = verify_stage1_ties(
                            x, w13, w2, s13, s2, weights, ids,
                            actuals[(None, True)].cpu(), bad,
                        )
                        assert scaled_tie_routes <= stage1_tie_routes
            actuals[(factor, no_combine)] = actual
            if factor is None and no_combine:
                # This also checks that unscaled combine has no hidden
                # accumulation or route selection difference.
                assert torch.equal(
                    actuals[(None, False)].contiguous().view(torch.int16),
                    actual.sum(1).contiguous().view(torch.int16),
                ), "Unscaled combine differs bitwise from routed BF16 sum"
            elif factor is not None:
                torch.testing.assert_close(
                    actual, actuals[(None, no_combine)] * factor,
                    rtol=0, atol=0,
                )
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
                        "explained_tie_elements": len(bad) if no_combine else 0,
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "cases": 44,
                      "max_memory_allocated": torch.cuda.max_memory_allocated()}), flush=True)


if __name__ == "__main__":
    main()
