#!/usr/bin/env python3
"""Check packed E2M1/E8M0 experts against a literal CPU FP64 reference."""

import gc
import hashlib
import json
import os
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
    assert len(candidate[1]) == len(baseline[1]) == 2
    for i in range(2):
        same_bits(candidate[1][i], baseline[1][i], case + ":stage" + str(i))


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
        new = call(
            candidate, *buffers, factor=factor, no_combine=no_combine, stages=True
        )
        new_snapshot = snapshot_result(new)  # Own bytes before baseline/reference work.
        old = call(
            baseline, *buffers, factor=factor, no_combine=no_combine, stages=True
        )
        old_snapshot = snapshot_result(old)
        configs_new = [record[1]["config"] for record in new[2]]
        configs_old = [record[1]["config"] for record in old[2]]
        emit(event="dispatch", case=label, candidate=configs_new, baseline=configs_old)
        compare_snapshots(new_snapshot, old_snapshot, label)
        assert configs_new[0] == configs_old[0] | {"BLOCK_SIZE_N": 128}
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
        compare_snapshots(new, old, "graph:" + label)
        eager = snapshot_result(call(candidate, *buffers, stages=True))
        compare_snapshots(new, eager, "graph:eager:" + label)
        oracle(ref, *buffers, new[0], None, True, "graph:" + label)
        COUNTS["graphs"] += 1
    del graphs, outputs


def check_gate_up():
    candidate = moe
    baseline = n256_control(candidate)
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
    }
    for name, values in patterns.items():
        correctness_case(candidate, baseline, ref, buffers, name, values)
    correctness_case(
        candidate, baseline, ref, buffers, "row_swap", patterns["disjoint"], swap=True
    )
    fallback_cases(candidate, baseline, buffers)
    graph_cases(
        candidate, baseline, ref, buffers, patterns["shared"], patterns["disjoint"]
    )
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
    check_gate_up()
