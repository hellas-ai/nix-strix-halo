"""C1/C2/C4 routed composition: exact policy steps, declared GEMM accuracy budget.

Native TP4 geometry and independent two-term FP64 projection references. The
FP32-scale projection budget is an engineering acceptance policy, not a proved
WMMA error bound. Existing dense MXFP4 tests remain complementary coverage.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["SGLANG_USE_AITER"] = "0"
# This test spies on the installed sequence's two GEMM calls and two quantiser calls. The five-kernel decode
# chain (dsv41_mxfp4_decode) computes the same values; mxfp4-decode.py checks it kernel by kernel against this
# sequence, so here the installed sequence stays selected.
os.environ["SGLANG_DSV41_MXFP4_DECODE"] = "0"

import torch
import torch.nn.functional as F


def emit(name, passed, **details):
    print(
        json.dumps(dict(event="gate", name=name, passed=bool(passed), **details)),
        flush=True,
    )
    assert passed, name


def bits(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def digest(x):
    return hashlib.sha256(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def projection_acceptance(actual, exact, absolute, k):
    """Require the observed BF16 rounding-cell closure to meet an FP32-scale budget.

    gamma_k * sum(abs(products)) uses the conventional sequential FP32-dot
    error scale as a declared accuracy budget. RDNA WMMA is not established to
    obey that scalar model, so this is NOT a hardware theorem or exactness claim.
    Full K is conservative for this two-nonzero fixture. Cell closures include
    midpoint ties conservatively; this is not a tie-parity-exact cell test.
    No absolute tolerance is inferred from measured hardware residuals.
    """
    assert actual.device.type == exact.device.type == absolute.device.type == "cpu"
    assert (
        actual.dtype == torch.bfloat16
        and exact.dtype == absolute.dtype == torch.float64
    )
    assert actual.shape == exact.shape == absolute.shape and 0 < k < 2**24
    if not (
        torch.isfinite(actual).all()
        and torch.isfinite(exact).all()
        and torch.isfinite(absolute).all()
        and (absolute >= exact.abs()).all()
    ):
        return {
            "passed": False,
            "reason": "nonfinite output/reference or invalid L1 premise",
        }
    # The sparse fixture proves exact FP64 products/sums. Round this budget
    # upward, then compare exact FP64 midpoints of adjacent BF16 values.
    u = 2.0**-24
    gamma = math.nextafter((k * u) / (1 - k * u), math.inf)
    budget = torch.nextafter(absolute * gamma, torch.full_like(absolute, math.inf))
    budget = torch.where(absolute == 0, torch.zeros_like(budget), budget)
    previous = torch.nextafter(actual, torch.full_like(actual, -math.inf)).double()
    following = torch.nextafter(actual, torch.full_like(actual, math.inf)).double()
    lower = (previous + actual.double()) / 2
    upper = (following + actual.double()) / 2
    distance = torch.maximum(
        torch.maximum(lower - exact, exact - upper), torch.zeros_like(exact)
    )
    accepted = distance <= budget
    return {
        "passed": bool(accepted.all()),
        "projection_accuracy_policy": "BF16 cell closure intersects FP64 reference +/- gamma_K*L1",
        "proved_hardware_bound": False,
        "k": k,
        "gamma_K": gamma,
        "max_reference_abs": float(exact.abs().max()),
        "max_native_L1": float(absolute.max()),
        "violations": int((~accepted).sum()),
        "max_abs_error": float((actual.double() - exact).abs().max()),
        "max_cell_distance": float(distance.max()),
        "max_fp32_scale_budget": float(budget.max()),
        "max_cell_distance_budget_ratio": float(
            torch.where(
                budget > 0, distance / budget, torch.where(distance == 0, 0.0, math.inf)
            ).max()
        ),
    }


def check_projection_contract():
    def accepts(observed, exact, l1, k=5120):
        return projection_acceptance(
            torch.tensor([observed], dtype=torch.bfloat16),
            torch.tensor([exact], dtype=torch.float64),
            torch.tensor([l1], dtype=torch.float64),
            k,
        )["passed"]

    u = 2.0**-24
    short_budget = 16 * u / (1 - 16 * u)
    controls = {
        "exact_zero": accepts(0, 0, 0),
        "zero_L1_rejects_nonzero": not accepts(2.0**-133, 0, 0),
        "exact_one": accepts(1, 1, 1),
        "negative_cell": accepts(-1, -1, 1),
        "negative_cell_rejects_wrong_value": not accepts(-1 - 2.0**-7, -1, 1),
        "subnormal_cell": accepts(2.0**-133, 2.0**-133, 2.0**-133),
        "subnormal_cell_rejects_wrong_value": not accepts(
            2.0**-132, 2.0**-133, 2.0**-133
        ),
        "rejects_missing_scale": not accepts(1.5, 1, 1),
        "rejects_wrong_BF16_value": not accepts(1 + 2.0**-7, 1, 1),
        "rejects_cancellation_error": not accepts(0.125, 0, 20),
        "inside_declared_budget": accepts(short_budget / 2, 0, 1, 16),
        "outside_declared_budget": not accepts(short_budget * 2, 0, 1, 16),
        "rejects_nonfinite": not accepts(math.nan, 0, 1),
        "rejects_invalid_L1": not accepts(1, 1, 0),
    }
    x = torch.ones((1, 5120), dtype=torch.bfloat16)
    expected, l1 = sparse_projection_reference(
        x, torch.tensor([[2]], dtype=torch.int32), 8, True
    )
    wrong_route, _ = sparse_projection_reference(
        x, torch.tensor([[7]], dtype=torch.int32), 8, True
    )
    controls["rejects_wrong_expert_route"] = not projection_acceptance(
        wrong_route.bfloat16(), expected, l1, 5120
    )["passed"]
    emit(
        "projection_accuracy_contract_controls",
        all(controls.values()),
        controls=controls,
    )


def quant_reference(x):
    """Finite BF16 group32 E4M3FN RNE, enumerated codes; FP32 power-of-two scales."""
    assert x.device.type == "cpu" and x.dtype == torch.bfloat16
    v = x.double().reshape(*x.shape[:-1], -1, 32)
    assert torch.isfinite(v).all()
    raw = v.abs().amax(-1).clamp_min(1e-4) / 448
    exp = torch.tensor(
        [math.ceil(math.log2(a)) for a in raw.flatten().tolist()]
    ).reshape(raw.shape)
    scale = torch.exp2(exp.double())
    level = torch.tensor(
        [
            (i & 7) * 2.0**-9
            if (i >> 3) == 0
            else (1 + (i & 7) / 8) * 2.0 ** ((i >> 3) - 7)
            for i in range(127)
        ],
        dtype=torch.float64,
    )
    normalized = v / scale.unsqueeze(-1)
    a = normalized.abs().contiguous()
    hi = torch.searchsorted(level, a).clamp(0, 126)
    lo = (hi - 1).clamp_min(0)
    dl, dh = a - level[lo], level[hi] - a
    code = torch.where((dh < dl) | ((dh == dl) & ((hi & 1) == 0)), hi, lo)
    payload = (code | (torch.signbit(normalized).long() << 7)).byte().reshape(x.shape)
    decoded = torch.where(torch.signbit(normalized), -level[code], level[code])
    expanded = (decoded * scale.unsqueeze(-1)).reshape(x.shape)
    assert torch.equal(expanded.bfloat16().double(), expanded), (
        "exact operand expansion"
    )
    return payload, scale.float(), expanded.bfloat16()


def sparse_parameters(expert, outputs, k, gate_up):
    row = torch.arange(outputs)
    first = (row * 13 + expert * 7) % k
    second = (first + k // 2 + 1) % k
    power = (expert % 3 - 1) if gate_up else (expert % 4 - 3)
    return first, second, power


def fill_expert(weight, scales, expert, gate_up):
    n, packed_k = weight.shape[1:]
    first, second, power = sparse_parameters(expert, n, packed_k * 2, gate_up)
    row = torch.arange(n)
    packed = torch.zeros((n, packed_k), dtype=torch.uint8)
    # +1 and -2 for gate/up; +1 and -1 for down, in native E2M1 codes.
    packed[row, first // 2] |= (2 << ((first % 2) * 4)).byte()
    packed[row, second // 2] |= ((12 if gate_up else 10) << ((second % 2) * 4)).byte()
    weight[expert].copy_(packed)
    scales[expert].fill_(127 + power)
    assert bits(weight[expert].cpu(), packed)


def sparse_projection_reference(x, ids, outputs, gate_up):
    rows, absolutes = [], []
    for token, routes in enumerate(ids.tolist()):
        projected, l1 = [], []
        for slot, expert in enumerate(routes):
            a = x[token] if gate_up else x[token, slot]
            first, second, power = sparse_parameters(
                expert, outputs, a.numel(), gate_up
            )
            # Two dyadic BF16 terms: reference products and sums fit FP64
            # exactly. This does not imply exact WMMA cancellation.
            exact = (
                a.double()[first] - (2 if gate_up else 1) * a.double()[second]
            ) * 2.0**power
            assert torch.equal(exact.float().double(), exact)
            absolute = (
                a.double()[first].abs()
                + (2 if gate_up else 1) * a.double()[second].abs()
            ) * 2.0**power
            assert torch.equal(absolute.float().double(), absolute)
            projected.append(exact)
            l1.append(absolute)
        rows.append(torch.stack(projected))
        absolutes.append(torch.stack(l1))
    return torch.stack(rows), torch.stack(absolutes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime")
    args = parser.parse_args()
    runtime = Path(args.runtime).resolve()
    check_projection_contract()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.03, 0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    import sglang.kernels.ops.quantization.fp8_kernel as quant
    import sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd as moe

    assert Path(moe.__file__).resolve().is_relative_to(runtime)
    w13 = torch.zeros((384, 1152, 2560), dtype=torch.uint8, device="cuda")
    w2 = torch.zeros((384, 5120, 288), dtype=torch.uint8, device="cuda")
    s13 = torch.full((384, 1152, 160), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((384, 5120, 18), 127, dtype=torch.uint8, device="cuda")
    patterns = [
        [2, 7, 11, 19, 23, 31],
        [31, 23, 19, 11, 7, 2],
        [3, 5, 13, 17, 29, 37],
        [2, 2, 7, 7, 11, 11],
    ]
    for expert in sorted({e for row in patterns for e in row}):
        fill_expert(w13, s13, expert, True)
        fill_expert(w2, s2, expert, False)
    original_weights = [digest(t) for t in (w13, w2, s13, s2)]
    kw = {
        "activation": "silu",
        "is_gated": True,
        "inplace": False,
        "no_combine": False,
        "apply_router_weight_on_input": False,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10,
        "official_routed_policy": True,
    }
    for m in (1, 2, 4):
        x_cpu = (
            ((torch.arange(m * 5120).reshape(m, 5120) * 7) % 97 - 48) / 8
        ).bfloat16()
        x_cpu[:, :2] = torch.tensor([16, -16], dtype=torch.bfloat16)
        ids_cpu = torch.tensor(patterns[:m], dtype=torch.int32)
        weights_cpu = (
            torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
            .expand(m, -1)
            .clone()
            / 21
        )
        x, ids, weights = x_cpu.cuda(), ids_cpu.cuda(), weights_cpu.cuda()

        def run(x=x, weights=weights, ids=ids):
            return moe.fused_experts_mxfp4(x, w13, w2, weights, ids, s13, s2, **kw)

        def checked(label, x=x, ids=ids, weights=weights, m=m, run=run):
            projections, quants = [], []
            gemm = moe._run_mxfp4_gemm
            quantizer = quant.sglang_per_token_group_quant_fp8

            def spy_gemm(*gemm_args, **k):
                gemm(*gemm_args, **k)
                projections.append(
                    (
                        gemm_args[0].detach().cpu().clone(),
                        gemm_args[2].detach().cpu().clone(),
                        k["mul_routed_weight"],
                    )
                )

            def spy_quant(operand, *args, **k):
                q, s = quantizer(operand, *args, **k)
                quants.append(
                    (
                        operand.detach().cpu().clone(),
                        q.detach().cpu().clone(),
                        s.detach().cpu().clone(),
                    )
                )
                return q, s

            before = [digest(t) for t in (x, ids, weights)]
            moe._run_mxfp4_gemm, quant.sglang_per_token_group_quant_fp8 = (
                spy_gemm,
                spy_quant,
            )
            try:
                actual = run().detach().cpu().clone()
            finally:
                moe._run_mxfp4_gemm, quant.sglang_per_token_group_quant_fp8 = (
                    gemm,
                    quantizer,
                )
            emit(
                label + ":dispatch",
                len(projections) == len(quants) == 2
                and not any(p[2] for p in projections),
            )
            for i, (operand, q, s) in enumerate(quants):
                payload, scale, expanded = quant_reference(operand)
                emit(
                    label + f":quant{i}",
                    bits(q.view(torch.uint8), payload)
                    and bits(s, scale)
                    and bits(projections[i][0], expanded),
                )
            projected, projection_l1 = sparse_projection_reference(
                projections[0][0], ids.cpu(), 1152, True
            )
            verdict = projection_acceptance(
                projections[0][1], projected, projection_l1, 5120
            )
            emit(label + ":gate_up_fp64_accuracy", verdict.pop("passed"), **verdict)
            # FP32 activation contract from the independently accepted ACTUAL projection;
            # same-device Torch SiLU avoids claiming a CPU transcendental oracle.
            g, u = projections[0][1].cuda().float().chunk(2, -1)
            # Preserve official parenthesization: weight * (silu(gate) * up).
            activation = (
                (
                    (weights * 1.5).unsqueeze(-1)
                    * (F.silu(g.clamp(max=10)) * u.clamp(-10, 10))
                )
                .bfloat16()
                .cpu()
            )
            emit(
                label + ":weighted_activation",
                bits(quants[1][0], activation.reshape(m * 6, 576)),
            )
            down, down_l1 = sparse_projection_reference(
                projections[1][0].reshape(m, 6, 576), ids.cpu(), 5120, False
            )
            verdict = projection_acceptance(projections[1][1], down, down_l1, 576)
            emit(label + ":down_fp64_accuracy", verdict.pop("passed"), **verdict)
            expected = torch.zeros((m, 5120), dtype=torch.float32)
            for row in range(m):
                for slot in sorted(range(6), key=lambda s: int(ids[row, s])):
                    expected[row] += projections[1][1][row, slot].float()
            emit(
                label + ":logical_sum_scale_once",
                actual.dtype == torch.float32 and bits(actual, expected),
            )
            emit(
                label + ":inputs_unchanged",
                before == [digest(t) for t in (x, ids, weights)],
            )
            return actual

        a = checked(f"C{m}:A")
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        owned_a = captured.cpu().clone()
        emit(f"C{m}:graph_A", bits(a, owned_a))
        x.copy_(-x_cpu)
        ids.copy_(ids_cpu.flip(1))
        weights.copy_(weights_cpu.flip(1))
        b = checked(f"C{m}:B")
        graph.replay()
        torch.cuda.synchronize()
        owned_b = captured.cpu().clone()
        emit(f"C{m}:graph_B", bits(b, owned_b) and not bits(owned_a, owned_b))
        x.copy_(x_cpu)
        ids.copy_(ids_cpu)
        weights.copy_(weights_cpu)
        graph.replay()
        torch.cuda.synchronize()
        emit(f"C{m}:graph_A_return", bits(captured.cpu(), owned_a) and bits(a, owned_a))
        del graph, captured
    emit(
        "native_weight_scale_unchanged",
        original_weights == [digest(t) for t in (w13, w2, s13, s2)],
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "passed": True,
                "full_model_acceptance": False,
                "request_rows": [1, 2, 4],
                "gpu_allocator_fraction": 0.03,
                "peak_gpu_allocated": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
