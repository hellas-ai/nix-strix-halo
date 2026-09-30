#!/usr/bin/env python3
"""Check packed E2M1/E8M0 experts against a literal CPU FP64 reference."""

import ast
import gc
import hashlib
import inspect
import json
import math
import os
import struct
import sys
import types
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
    print(json.dumps({"event": "reference_complete", "cases": 44,
                      "max_memory_allocated": torch.cuda.max_memory_allocated()}), flush=True)


def direct_bf16_scalar(value):
    """Exact nearest-even conversion of a finite binary64 to BF16 bits."""
    assert math.isfinite(value)
    raw = struct.unpack("<Q", struct.pack("<d", value))[0]
    sign, exponent, fraction = (
        (raw >> 48) & 32768,
        (raw >> 52) & 2047,
        raw & (2**52 - 1),
    )
    mantissa = fraction if exponent == 0 else 2**52 + fraction
    power = -1074 if exponent == 0 else exponent - 1023 - 52
    return round_dyadic_bf16(
        -mantissa if sign else mantissa, power, negative_zero=bool(sign)
    )


def round_dyadic_bf16(integer, power, negative_zero=False):
    sign = 32768 if integer < 0 or (integer == 0 and negative_zero) else 0
    integer = abs(integer)
    if not integer:
        return sign
    exponent = integer.bit_length() - 1 + power
    if exponent > 127:
        return sign | 0x7F80
    spacing = max(exponent - 7, -133)
    shift = spacing - power
    if shift > 0:
        quotient, remainder = divmod(integer, 1 << shift)
        half = 1 << (shift - 1)
        quotient += remainder > half or (remainder == half and quotient & 1)
    else:
        quotient = integer << -shift
    if exponent < -126:
        bits = quotient
    else:
        if quotient == 256:
            quotient, exponent = 128, exponent + 1
        bits = 0x7F80 if exponent > 127 else ((exponent + 127) << 7) | (quotient - 128)
    return sign | bits


def direct_bf16(values):
    bits = [direct_bf16_scalar(x) for x in values.reshape(-1).tolist()]
    return (
        torch.tensor(bits, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(values.shape)
    )


def bf16_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def cell_intersects(bits, lower, upper):
    """Closed interval versus exact RNE cell, including midpoint parity."""
    value = bf16_value(bits)
    assert (
        math.isfinite(value)
        and math.isfinite(lower)
        and math.isfinite(upper)
        and lower <= upper
    )
    if value == 0:
        previous, following = -(2.0**-133), 2.0**-133
    elif bits & 32768:
        previous, following = bf16_value(bits + 1), bf16_value(bits - 1)
    else:
        previous, following = bf16_value(bits - 1), bf16_value(bits + 1)
    assert math.isfinite(previous) and math.isfinite(following), (
        "BF16 overflow edge outside fixture"
    )
    low_mid, high_mid = (previous + value) / 2, (following + value) / 2
    even = bits & 1 == 0
    return (upper > low_mid or (upper == low_mid and even)) and (
        lower < high_mid or (lower == high_mid and even)
    )


def upward(value):
    assert torch.isfinite(value).all() and (value >= 0).all()
    return torch.where(
        value == 0, value, torch.nextafter(value, torch.full_like(value, math.inf))
    )


def gamma(n):
    u = 2.0**-24
    return math.nextafter(n * u / (1 - n * u), math.inf)


def lattice(values):
    """Exact least-set-bit exponent for each finite normal BF16, zero sentinel."""
    assert values.device.type == "cpu" and values.dtype == torch.bfloat16
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 65535
    exponent, mantissa = (bits >> 7) & 255, bits & 127
    zero = (exponent == 0) & (mantissa == 0)
    assert ((exponent > 0) & (exponent < 255) | zero).all(), (
        "nonfinite/subnormal BF16 operand"
    )
    significant = mantissa | 128
    trailing = torch.zeros_like(significant)
    for _ in range(7):
        even = significant & 1 == 0
        trailing += even
        significant = torch.where(even, significant >> 1, significant)
    return torch.where(zero, 1024, exponent - 134 + trailing)


def decode_native(packed, scales):
    assert packed.device.type == scales.device.type == "cpu"
    assert packed.dtype == scales.dtype == torch.uint8
    assert ((scales >= 1) & (scales <= 254)).all(), (
        "finite-normal reference excludes reserved scale bytes0/255"
    )
    exact = unpack(packed, scales)
    expanded = exact.to(torch.bfloat16)
    assert torch.equal(expanded.double(), exact), (
        "native BF16 weight expansion is not exact"
    )
    return expanded


def exact_dot(a, w, envelope=False):
    """Prove every FP64 partial exact, then sum; never fall back on a looser proof."""
    assert a.ndim == 1 and w.ndim == 2 and a.numel() == w.shape[1]
    aq, wq = lattice(a), lattice(w)
    active_a = aq != 1024
    q = aq.min() + wq.min(dim=1).values
    active = (w != 0).any(dim=1) & active_a.any()
    q = torch.where(active, q, 0)
    quantum = torch.ldexp(torch.ones(w.shape[0], dtype=torch.float64), q)
    maximum = a.double().abs().max() * w.double().abs().amax(dim=1)
    worst_l1 = upward(maximum * a.numel())
    assert ((q >= -126) | ~active).all(), (
        "FP32 nonzero intermediate normality not proved"
    )
    assert (worst_l1 < 2.0**126).all(), "FP32 overflow exclusion failed"
    units = upward(worst_l1 / quantum)
    assert (units < 2.0**52).all(), "exact FP64 dyadic headroom not proved"
    product = w.double() * a.double()[None, :]
    # BF16 products fit<=16 significant bits; lattice+L1 prove all partial sums exact.
    exact = product.sum(dim=1)
    proof = {
        "max_lattice_units": float(units.max()),
        "minimum_quantum_exponent": int(q.min()),
        "K": a.numel(),
    }
    if not envelope:
        return exact, proof
    assert a.numel() == 5120
    blocks = product.reshape(w.shape[0], 10, 512)
    s, l1 = blocks.sum(dim=2), blocks.abs().sum(dim=2)
    b = upward(l1 * gamma(9))
    error, partial = b[:, 0], s[:, 0]
    for j in range(1, 10):
        partial = partial + s[:, j]
        propagated = upward(upward(error + b[:, j]) * (1 + 2.0**-24))
        fresh = upward(partial.abs() * 2.0**-24)
        error = upward(propagated + fresh)
    conservative = upward(l1.sum(dim=1) * gamma(18))
    assert (error <= conservative).all(), "tight recurrence exceeds conservative bound"
    return exact, error, proof


# Separate globals preserve an N256 differential control without changing the
# installed module or embedding another implementation. The independent FP64
# reference above remains the acceptance oracle.
COUNTS = {"bitwise": 0, "oracle": 0, "graphs": 0}


def n256_control(candidate):
    baseline = types.ModuleType("mxfp4_n256_control")
    baseline.__dict__.update(vars(candidate))
    baseline.gate_up_moe_config = lambda config, *args: config
    fn = candidate.fused_experts_mxfp4
    baseline.fused_experts_mxfp4 = types.FunctionType(
        fn.__code__, baseline.__dict__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    baseline.fused_experts_mxfp4.__kwdefaults__ = fn.__kwdefaults__
    assert baseline.fused_experts_mxfp4.__globals__ is baseline.__dict__
    assert candidate.fused_experts_mxfp4.__globals__ is vars(candidate)
    return baseline


def n128_control(candidate):
    """Isolate the installed fallback; remove only its asserted live-row guard."""
    baseline = types.ModuleType("mxfp4_n128_control")
    baseline.__dict__.update(vars(candidate))
    tree = ast.parse(inspect.getsource(candidate._run_mxfp4_gemm))
    fn = tree.body[0]
    guard = fn.body[0]
    assert isinstance(guard, ast.If)
    assert (
        ast.unparse(guard.test)
        == "a.shape == (2, 5120) and top_k == 6 and (not mul_routed_weight)"
    )
    assert isinstance(guard.body[0], ast.ImportFrom)
    assert guard.body[0].module == "sglang.kernels.ops.gemm.dsv41_mxfp4_gate_up"
    assert guard.body[0].names[0].name == "try_gate_up_live_rows"
    assert isinstance(guard.body[1], ast.If)
    assert isinstance(guard.body[1].body[0], ast.Return)
    fn.body.pop(0)
    exec(compile(tree, "<installed-n128-fallback>", "exec"), baseline.__dict__)  # noqa: S102 -- asserted installed function
    fused = candidate.fused_experts_mxfp4
    baseline.fused_experts_mxfp4 = types.FunctionType(
        fused.__code__,
        baseline.__dict__,
        fused.__name__,
        fused.__defaults__,
        fused.__closure__,
    )
    baseline.fused_experts_mxfp4.__kwdefaults__ = fused.__kwdefaults__
    return baseline


def check_gate_up_guards(candidate):
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }
    before = config.copy()
    shape = [2, 6, 384, 5120, 576]
    assert candidate.gate_up_moe_config(config, "gfx1151", *shape) == (
        config | {"BLOCK_SIZE_N": 128}
    )
    assert config == before
    for arch in ("gfx950", "gfx1100"):
        assert candidate.gate_up_moe_config(config, arch, *shape) is config
    for dim in range(len(shape)):
        other = shape.copy()
        other[dim] += 1
        assert candidate.gate_up_moe_config(config, "gfx1151", *other) is config
    for key, value in config.items():
        other = config | {key: value + 1}
        assert candidate.gate_up_moe_config(other, "gfx1151", *shape) is other
    emit(event="gate_up_guards", controls=17)


def emit(**row):
    print(json.dumps(row, sort_keys=True), flush=True)


def same_bits(a, b, case):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        emit(
            event="bitwise",
            case=case,
            shape_dtype_failure=True,
            candidate_shape=list(a.shape),
            baseline_shape=list(b.shape),
            candidate_dtype=str(a.dtype),
            baseline_dtype=str(b.dtype),
        )
        raise AssertionError(case)
    raw_a, raw_b = a.view(torch.uint8).reshape(-1), b.view(torch.uint8).reshape(-1)
    bad = (raw_a != raw_b).nonzero().flatten()
    values = [
        {
            "byte": int(i),
            "element": int(i) // a.element_size(),
            "candidate_byte": int(raw_a[i]),
            "baseline_byte": int(raw_b[i]),
            "candidate_value": float(a.reshape(-1)[int(i) // a.element_size()]),
            "baseline_value": float(b.reshape(-1)[int(i) // b.element_size()]),
        }
        for i in bad[:16]
    ]
    emit(
        event="bitwise",
        case=case,
        shape=list(a.shape),
        dtype=str(a.dtype),
        unequal_bytes=bad.numel(),
        first=values,
    )
    assert not bad.numel(), case
    COUNTS["bitwise"] += 1


def tensor_sha(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 64 * 1024**2):
        owned = raw[start : start + 64 * 1024**2].cpu()
        digest.update(memoryview(owned.numpy()))
    return digest.hexdigest()


def call(
    module,
    x,
    w13,
    w2,
    s13,
    s2,
    weights,
    ids,
    *,
    factor=None,
    no_combine=True,
    stages=False,
    router_on_input=False,
):
    calls, snapshots = [], []
    original = module._run_mxfp4_gemm

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        if snapshots:
            snapshots.append(args[0].reshape(x.shape[0], 6, 576).clone())
        args[2].fill_(float("nan"))
        original(*args, **kwargs)
        snapshots.append(args[2].clone())

    if stages:
        module._run_mxfp4_gemm = spy
    try:
        out = module.fused_experts_mxfp4(
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
            apply_router_weight_on_input=router_on_input,
            routed_scaling_factor=factor,
            swiglu_limit=10,
        )
    finally:
        module._run_mxfp4_gemm = original
    return out, snapshots, calls


def oracle(
    ref,
    x,
    w13,
    w2,
    s13,
    s2,
    weights,
    ids,
    actual,
    factor,
    no_combine,
    case,
    unscaled=None,
    stage1_tie_routes=None,
):
    expected = ref.reference(x, w13, w2, s13, s2, weights, ids, factor, no_combine)
    actual_cpu = actual.detach().cpu().clone()
    error = float(
        (actual_cpu.float() - expected.float()).norm() / expected.float().norm()
    )
    bad = (
        (
            ~torch.isclose(actual_cpu, expected, rtol=0.03, atol=0.002 * (factor or 1))
        ).nonzero()
        if no_combine
        else torch.empty((0, 3), dtype=torch.int64)
    )
    emit(
        event="oracle",
        case=case,
        relative_l2=error,
        finite=bool(torch.isfinite(actual_cpu).all()),
        per_route_bad=len(bad),
        first_bad=bad[:16].tolist(),
        ids=ids.cpu().tolist(),
    )
    assert torch.isfinite(actual_cpu).all() and error < 0.002, case
    ties = set()
    if len(bad):
        ties = ref.verify_stage1_ties(
            x,
            w13,
            w2,
            s13,
            s2,
            weights,
            ids,
            actual_cpu if factor is None else unscaled,
            bad,
        )
        if factor is not None:
            emit(
                event="scaled_ties",
                case=case,
                ties=sorted(ties),
                prior=sorted(stage1_tie_routes),
            )
            assert ties <= stage1_tie_routes
    COUNTS["oracle"] += 1
    return ties


def snapshot_result(result):
    out, stages, _ = result
    return out.detach().cpu().clone(), [
        value.detach().cpu().clone() for value in stages
    ]


def compare_snapshots(candidate, baseline, case):
    same_bits(candidate[0], baseline[0], case + ":output")
    assert len(candidate[1]) == len(baseline[1]) == 3
    for i in range(3):
        same_bits(candidate[1][i], baseline[1][i], case + ":stage" + str(i))


def check_reference_contract():
    """Literal rounding, cancellation and fail-closed premise controls."""
    value = 1 + 2.0**-8 + 2.0**-40
    assert direct_bf16_scalar(value) == 0x3F81
    assert (
        torch.tensor(value, dtype=torch.float32).bfloat16().view(torch.int16).item()
        == 0x3F80
    )
    assert direct_bf16_scalar(1 + 2.0**-8) == 0x3F80
    assert direct_bf16_scalar(-0.0) == 0x8000
    assert cell_intersects(0x3F80, 1, 1)
    assert not cell_intersects(0x3F81, 1, 1)
    assert not cell_intersects(0x3F81, 1 + 2.0**-8, 1 + 2.0**-8)
    a = torch.zeros(5120, dtype=torch.bfloat16)
    a[0], a[512] = 1, -1
    w = torch.zeros((1, 5120), dtype=torch.bfloat16)
    w[0, 0], w[0, 512] = 1, 1
    exact, radius, _ = exact_dot(a, w, True)
    assert exact.item() == 0 and 0 < radius.item() < 0.001
    assert cell_intersects(0, -radius.item(), radius.item())
    assert not cell_intersects(0x3F80, -radius.item(), radius.item())
    a[0] = 2.0**-133
    try:
        exact_dot(a, w, True)
    except AssertionError as error:
        assert str(error) == "nonfinite/subnormal BF16 operand"
    else:
        raise AssertionError("Subnormal fixture premise was accepted")
    a.fill_(1)
    w[0, 0], w[0, 512] = 2.0**100, 2.0**-100
    try:
        exact_dot(a, w, True)
    except AssertionError as error:
        assert str(error) == "exact FP64 dyadic headroom not proved"
    else:
        raise AssertionError("Unproved exact reference was accepted")
    emit(event="reference_contract", passed=True)


def live_reference(buffers, actual, baseline, case):
    """Gate/up accuracy contract; the existing oracle checks the downstream path."""
    x, w13, _w2, s13, _s2, _weights, ids = buffers
    x, ids = x.cpu(), ids.cpu()
    projection, activation, _routed = actual[1]
    assert all(
        torch.isfinite(v).all()
        for v in (actual[0], *actual[1], baseline[0], *baseline[1])
    )
    violations, count, better, worse, direct_mismatches = [], 0, 0, 0, 0
    for expert in ids.unique(sorted=True).tolist():
        if expert == -1:
            continue
        decoded13 = decode_native(w13[expert].cpu(), s13[expert].cpu())
        for token in range(2):
            routes = (ids[token] == expert).nonzero().flatten().tolist()
            if not routes:
                continue
            first = routes[0]
            for route in routes[1:]:
                same_bits(
                    projection[token, first],
                    projection[token, route],
                    case + ":duplicate",
                )
                same_bits(
                    activation[token, first],
                    activation[token, route],
                    case + ":duplicate-activation",
                )
            for start in range(0, 1152, 128):
                exact, bound, _ = exact_dot(
                    x[token], decoded13[start : start + 128], True
                )
                for route in routes:
                    observed = projection[token, route, start : start + 128]
                    old = baseline[1][0][token, route, start : start + 128]
                    ce, be = (
                        (observed.double() - exact).abs(),
                        (old.double() - exact).abs(),
                    )
                    better += int((ce < be).sum())
                    worse += int((ce > be).sum())
                    count += observed.numel()
                    rounded = direct_bf16(exact)
                    direct_mismatches += int(
                        (observed.view(torch.int16) != rounded.view(torch.int16)).sum()
                    )
                    codes = (
                        observed.view(torch.int16).to(torch.int32) & 65535
                    ).tolist()
                    for offset, (code, value, radius) in enumerate(
                        zip(codes, exact.tolist(), bound.tolist())
                    ):
                        if not cell_intersects(
                            code,
                            math.nextafter(value - radius, -math.inf),
                            math.nextafter(value + radius, math.inf),
                        ):
                            violations.append(
                                (
                                    token,
                                    route,
                                    start + offset,
                                    float(observed[offset]),
                                    value,
                                    radius,
                                )
                            )
    emit(
        event="live_reference",
        case=case,
        values=count,
        candidate_closer=better,
        baseline_closer=worse,
        equal=count - better - worse,
        interval_violations=len(violations),
        first_violations=violations[:16],
        direct_round_mismatches=direct_mismatches,
        baseline_scope="Common projection reference only; downstream gate is the existing full-pipeline oracle",
    )
    assert not violations, case + ": operation-derived projection envelope"
    COUNTS["oracle"] += 1


def live_call(candidate, buffers, **kwargs):
    """Observe the real dispatch with restoration even on failure."""
    from sglang.kernels.ops.gemm import dsv41_mxfp4_gate_up as helper

    original = helper.try_gate_up_live_rows
    selections = []

    def observe(*args, **kw):
        selected = original(*args, **kw)
        selections.append(selected)
        return selected

    helper.try_gate_up_live_rows = observe
    try:
        result = call(candidate, *buffers, **kwargs)
    finally:
        helper.try_gate_up_live_rows = original
    assert selections == [True], selections
    return result


def correctness_case(candidate, baseline, ref, buffers, name, ids_value, swap=False):
    x, w13, w2, s13, s2, weights, ids = buffers
    ids.copy_(torch.tensor(ids_value, device=ids.device, dtype=ids.dtype))
    if swap:
        x.copy_(x.flip(0))
        weights.copy_(weights.flip(0))
        ids.copy_(ids.flip(0))
    before = {
        key: tensor_sha(t) for key, t in [("x", x), ("weights", weights), ("ids", ids)]
    }
    emit(
        event="case_inputs", case=name, ids=ids.cpu().tolist(), before=before, seed=734
    )
    retained = {}
    stage1_tie_routes = set()
    for factor, no_combine in [(None, True), (None, False), (1.5, True), (1.5, False)]:
        label = name + f":factor{factor}:routes{no_combine}"
        new = live_call(
            candidate, buffers, factor=factor, no_combine=no_combine, stages=True
        )
        new_snapshot = snapshot_result(new)  # Own bytes before baseline/reference work.
        old = call(
            baseline, *buffers, factor=factor, no_combine=no_combine, stages=True
        )
        old_snapshot = snapshot_result(old)
        configs_new = [record[1]["config"] for record in new[2]]
        configs_old = [record[1]["config"] for record in old[2]]
        emit(event="dispatch", case=label, candidate=configs_new, baseline=configs_old)
        if factor is None and no_combine:
            live_reference(buffers, new_snapshot, old_snapshot, label)
            if name in ("shared", "disjoint", "partial", "row_swap"):
                n256 = snapshot_result(
                    call(n256_control(baseline), *buffers, stages=True)
                )
                compare_snapshots(old_snapshot, n256, label + ":retained-n128-n256")
        assert (
            configs_new[0] == configs_old[0] and configs_new[0]["BLOCK_SIZE_N"] == 128
        )
        assert (
            configs_new[1] == configs_old[1] and configs_new[1]["BLOCK_SIZE_N"] == 256
        )
        ties = oracle(
            ref,
            *buffers,
            new_snapshot[0],
            factor,
            no_combine,
            label,
            unscaled=retained.get((None, True)),
            stage1_tie_routes=stage1_tie_routes,
        )
        if factor is None and no_combine:
            stage1_tie_routes = ties
        retained[(factor, no_combine)] = new_snapshot[0]
        if factor is None and not no_combine:
            same_bits(
                new_snapshot[0], retained[(None, True)].sum(1), label + ":route_sum"
            )
        elif factor is not None:
            same_bits(
                new_snapshot[0],
                retained[(None, no_combine)] * factor,
                label + ":scale_once",
            )
    after = {
        key: tensor_sha(t) for key, t in [("x", x), ("weights", weights), ("ids", ids)]
    }
    emit(event="inputs", case=name, before=before, after=after)
    assert before == after


def fallback_cases(candidate, baseline, buffers):
    x, w13, w2, s13, s2, weights, ids = buffers
    for name, rows, dtype, router in [
        ("C1", 1, torch.bfloat16, False),
        ("C3", 3, torch.bfloat16, False),
        ("FP16", 2, torch.float16, False),
        ("router_on_input", 2, torch.bfloat16, True),
    ]:
        xx = torch.randn((rows, 5120), device=x.device, dtype=dtype)
        ww = torch.full((rows, 6), 1 / 6, device=x.device, dtype=torch.float32)
        ii = torch.tensor(
            [[0, 7, 63, 127, 255, 383]] * rows, device=x.device, dtype=torch.int32
        )
        args = (xx, w13, w2, s13, s2, ww, ii)
        new = call(candidate, *args, stages=True, router_on_input=router)
        snap = snapshot_result(new)
        old = call(baseline, *args, stages=True, router_on_input=router)
        compare_snapshots(snap, snapshot_result(old), "fallback:" + name)
        assert [v[1]["config"] for v in new[2]] == [v[1]["config"] for v in old[2]]
        emit(event="fallback", case=name, config=[v[1]["config"] for v in new[2]])


def graph_cases(candidate, baseline, ref, buffers, routes_a, routes_b):
    x, w13, w2, s13, s2, weights, ids = buffers
    x_a, weights_a = x.clone(), weights.clone()
    ids.copy_(torch.tensor(routes_a, dtype=ids.dtype, device=ids.device))
    graphs, outputs = [], []
    for module in (candidate, baseline):
        call(module, *buffers, stages=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = call(module, *buffers, stages=True)
        graphs.append(graph)
        outputs.append(result)
    for label, use_b in [("A", False), ("B", True), ("A-again", False)]:
        x.copy_(x_a.flip(0) * 0.75 if use_b else x_a)
        weights.copy_(weights_a.flip(0) if use_b else weights_a)
        ids.copy_(
            torch.tensor(
                routes_b if use_b else routes_a, dtype=ids.dtype, device=ids.device
            )
        )
        emit(
            event="graph_inputs",
            case=label,
            ids=ids.cpu().tolist(),
            hashes={
                key: tensor_sha(t)
                for key, t in [("x", x), ("weights", weights), ("ids", ids)]
            },
        )
        graphs[0].replay()
        new = snapshot_result(outputs[0])
        graphs[1].replay()
        old = snapshot_result(outputs[1])
        live_reference(buffers, new, old, "graph:" + label)
        eager = snapshot_result(call(candidate, *buffers, stages=True))
        compare_snapshots(new, eager, "graph:eager:" + label)
        oracle(ref, *buffers, new[0], None, True, "graph:" + label)
        COUNTS["graphs"] += 1
    del graphs, outputs


def expansion_controls(candidate, baseline, buffers):
    """Exact basis/classification probes of the installed BF16 arithmetic."""
    x, w13, _w2, s13, _s2, _weights, ids = buffers
    saved_x, saved_ids = x.clone(), ids.clone()
    ids.copy_(torch.tensor([[0, 7, 63, 127, 255, 383]] * 2, device=ids.device))
    active = ids[0].long()
    saved_w, saved_s = w13[active].clone(), s13[active].clone()
    # Retain actual aligner allocations and invoke gate/up alone: activation
    # and down projection must not hide the exceptional classifications.
    _, _, records = call(baseline, *buffers, stages=True)
    args, kwargs = records[0]
    outputs = [torch.empty_like(args[2]), torch.empty_like(args[2])]
    failures = []
    for name in (
        "subnormal-weight",
        "scale-zero",
        "subnormal-activation",
        "scale255-zero",
        "scale255-nonzero",
    ):
        packed = torch.zeros((1152, 2560), dtype=torch.uint8, device=x.device)
        scales = torch.full((1152, 160), 127, dtype=torch.uint8, device=x.device)
        x.zero_()
        expected = None
        if name == "subnormal-weight":
            packed[:, 0] = 1
            scales[:, 0] = 1
            x[0, 0], x[1, 0] = 2.0**126, -(2.0**126)
            expected = "zero"
        elif name == "scale-zero":
            packed[:, :16] = 0x22
            scales[:, 0] = 0
            x[0, :32], x[1, :32] = 1, -1
            expected = "zero"
        elif name == "subnormal-activation":
            packed[:, 0] = 2
            scales[:, 0] = 253
            x[0, 0], x[1, 0] = 2.0**-127, -(2.0**-127)
        else:
            scales[:, 0] = 255
            x[0, :32], x[1, :32] = 1, -1
            if name == "scale255-nonzero":
                # All32 lanes in the Inf-scale block must be nonzero;
                # otherwise0*Inf creates NaNs and conceals the Inf case.
                packed[:, :16] = 0x22
                expected = "infinity"
            else:
                expected = "nan"
        for expert in active.tolist():
            w13[expert].copy_(packed)
            s13[expert].copy_(scales)
        input_codes = (x[:, 0].cpu().view(torch.int16).to(torch.int32) & 65535).tolist()
        if name == "subnormal-activation":
            assert input_codes == [0x40, 0x8040], input_codes
        emit(
            event="expansion_input",
            case=name,
            activation_codes=input_codes,
            packed_first_block=packed[0, :16].cpu().tolist(),
            scale=int(scales[0, 0]),
        )
        snapshots = []
        for module, output in zip((candidate, baseline), outputs):
            output.fill_(123)  # A finite sentinel also detects unwritten NaNs.
            operands = list(args)
            operands[2] = output
            module._run_mxfp4_gemm(*operands, **kwargs)
            snapshots.append(output.cpu().clone())
        new, old = snapshots
        emit(
            event="expansion_control",
            case=name,
            candidate_first=[float(new[t, 0, 0]) for t in range(2)],
            baseline_first=[float(old[t, 0, 0]) for t in range(2)],
            candidate_nan=int(new.isnan().sum()),
            baseline_nan=int(old.isnan().sum()),
            candidate_inf=int(new.isinf().sum()),
            baseline_inf=int(old.isinf().sum()),
            activation_literal_basis=[0.5, -0.5]
            if name == "subnormal-activation"
            else None,
        )
        checks = {
            "nan_mask": torch.equal(new.isnan(), old.isnan()),
            "positive_inf_mask": torch.equal(new.isposinf(), old.isposinf()),
            "negative_inf_mask": torch.equal(new.isneginf(), old.isneginf()),
            "finite_bits": bf16_bits_equal(new[old.isfinite()], old[old.isfinite()]),
        }
        if expected == "zero":
            checks["expected_zero_bits"] = bf16_bits_equal(new, torch.zeros_like(new))
        elif expected == "nan":
            checks["expected_nan"] = bool(new.isnan().all())  # No NaN-payload gate.
        elif expected == "infinity":
            checks["expected_infinity"] = bool(
                new[0].isposinf().all() and new[1].isneginf().all()
            )
        failed = [key for key, passed in checks.items() if not passed]
        emit(event="expansion_verdict", case=name, checks=checks, failed=failed)
        if failed:
            failures.append({"case": name, "failed": failed})
    w13.index_copy_(0, active, saved_w)
    s13.index_copy_(0, active, saved_s)
    x.copy_(saved_x)
    ids.copy_(saved_ids)
    emit(event="expansion_summary", failures=failures, passed=not failures)
    assert not failures, failures


def padded_route_cases(candidate, baseline, buffers):
    """Filtered EP routes are +0, including dynamic graph transitions."""
    ids = buffers[-1]
    saved = ids.clone()
    mixed = [[-1, 0, -1, 7, 63, -1], [7, -1, 0, -1, -1, 255]]
    live_call(candidate, buffers, stages=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = live_call(candidate, buffers, stages=True)
    for label, pattern in (
        ("mixed", mixed),
        ("all-filtered", [[-1] * 6] * 2),
        ("mixed-return", mixed),
    ):
        ids.copy_(torch.tensor(pattern, dtype=ids.dtype, device=ids.device))
        new = snapshot_result(live_call(candidate, buffers, stages=True))
        old = snapshot_result(call(baseline, *buffers, stages=True))
        mask = ids.cpu() == -1
        for implementation, snapshot in (("candidate", new), ("n128", old)):
            for index, tensor in enumerate((snapshot[0], *snapshot[1])):
                zero = torch.zeros_like(tensor[mask])
                same_bits(tensor[mask], zero, f"{label}:{implementation}:zero{index}")
        # Normal valid routes retain the independent gate/up envelope; the
        # unchanged downstream oracle is exercised by the ordinary route cases.
        live_reference(buffers, new, old, "padding:" + label)
        graph.replay()
        compare_snapshots(snapshot_result(captured), new, "padding-graph:" + label)
        COUNTS["graphs"] += 1
    ids.copy_(saved)


def check_gate_up():
    candidate = moe
    baseline = n128_control(candidate)
    ref = sys.modules[__name__]
    check_gate_up_guards(candidate)
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(734)
    x = torch.randn((2, 5120), device="cuda", dtype=torch.bfloat16)
    w13 = torch.randint(256, (384, 1152, 2560), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(256, (384, 5120, 288), dtype=torch.uint8, device="cuda")
    s13 = torch.randint(117, 122, (384, 1152, 160), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(117, 122, (384, 5120, 18), dtype=torch.uint8, device="cuda")
    weights = torch.rand((2, 6), device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    ids = torch.empty((2, 6), dtype=torch.int32, device="cuda")
    buffers = (x, w13, w2, s13, s2, weights, ids)
    before = {
        name: tensor_sha(t)
        for name, t in [("w13", w13), ("w2", w2), ("s13", s13), ("s2", s2)]
    }
    emit(
        event="native_input_initial",
        hashes=before,
        seed=734,
        shape=[2, 384, 5120, 576],
        topk=6,
    )
    patterns = {
        "shared": [[0, 7, 63, 127, 255, 383]] * 2,
        "disjoint": [[0, 1, 7, 63, 127, 255], [2, 8, 64, 128, 256, 383]],
        "partial": [[0, 7, 63, 127, 255, 383], [0, 7, 63, 128, 256, 382]],
        "duplicate": [[0, 0, 7, 7, 63, 63], [63, 63, 127, 127, 255, 255]],
    }
    for name, values in patterns.items():
        correctness_case(candidate, baseline, ref, buffers, name, values)
    correctness_case(
        candidate, baseline, ref, buffers, "row_swap", patterns["disjoint"], swap=True
    )
    saved_x, saved_weights = x.clone(), weights.clone()
    active = torch.tensor(patterns["shared"][0], device=x.device)
    saved_w, saved_s = w13[active].clone(), s13[active].clone()
    for scale in (1, 254):
        packed = torch.full((1152, 2560), 0x88, dtype=torch.uint8, device=x.device)
        packed[:, 0] = 0x82
        packed[::2, 256] = 0x82
        for expert in patterns["shared"][0]:
            w13[expert].copy_(packed)
            s13[expert].fill_(scale)
        power, ratio = (2.0**126, 0.5) if scale == 1 else (2.0**-126, 2)
        x.zero_()
        x[0, 0], x[0, 512] = power, -power
        x[1, 0], x[1, 512] = power * ratio, -power * ratio
        ids.copy_(torch.tensor(patterns["shared"], device=ids.device))
        observed = snapshot_result(live_call(candidate, buffers, stages=True))
        expected = torch.zeros((2, 6, 1152), dtype=torch.bfloat16)
        expected[0, :, 1::2] = 1 if scale == 1 else 2
        expected[1, :, 1::2] = expected[0, :, 1::2] * ratio
        same_bits(observed[1][0], expected, "normal-cancellation:" + str(scale))
        correctness_case(
            candidate, baseline, ref, buffers, "scale" + str(scale), patterns["shared"]
        )
    w13.index_copy_(0, active, saved_w)
    s13.index_copy_(0, active, saved_s)
    x.copy_(saved_x)
    weights.copy_(saved_weights)
    del active, saved_w, saved_s
    fallback_cases(candidate, baseline, buffers)
    graph_cases(
        candidate, baseline, ref, buffers, patterns["shared"], patterns["disjoint"]
    )
    padded_route_cases(candidate, baseline, buffers)
    expansion_controls(candidate, baseline, buffers)
    after = {
        name: tensor_sha(t)
        for name, t in [("w13", w13), ("w2", w2), ("s13", s13), ("s2", s2)]
    }
    emit(event="native_inputs", before=before, after=after)
    assert before == after
    emit(
        event="complete",
        original_reference_cases=44,
        **COUNTS,
        max_memory_allocated=torch.cuda.max_memory_allocated(),
    )


if __name__ == "__main__":
    main()
    check_reference_contract()
    check_gate_up()
