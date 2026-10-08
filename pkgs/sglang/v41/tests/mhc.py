"""V4.1 portable mHC: literal FP64 reference and changing-data graph replay."""

import inspect
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["SGLANG_USE_AITER"] = "0"
os.environ["SGLANG_OPT_USE_FLASHINFER_MHC"] = "0"
os.environ["SGLANG_OPT_USE_TILELANG_MHC_PRE"] = "0"
os.environ["SGLANG_OPT_USE_TILELANG_MHC_POST"] = "0"

import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.kernels.ops.layernorm import native_hc_post as native_post
from sglang.kernels.ops.layernorm import native_hc_projection as native
from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer as Layer


def projection_reference(x, weight):
    # FP32 products are exact in FP64. The reduction has one multiply,
    # nine lane additions, nine tree levels and two final levels: gamma21.
    up = lambda v: math.nextafter(v, math.inf)
    u = 2.0**-24
    gamma = up(21 * u / (1 - 21 * u))
    underflow = up(40959 * 2.0**-150 * up((1 + u) ** 21))
    weights = weight.cpu().tolist()
    refs, bounds = [], []
    for row in x.cpu().tolist():
        values, errors = [], []
        for w in weights:
            terms = [a * b for a, b in zip(row, w, strict=True)]
            total, absolute = math.fsum(terms), up(math.fsum(map(abs, terms)))
            assert math.isfinite(total) and absolute < 2.0**120
            values.append(total)
            errors.append(up(up(up(gamma * absolute) + underflow) + math.ulp(total)))
        refs.append(values)
        bounds.append(errors)
    return torch.tensor(refs, dtype=torch.float64), torch.tensor(
        bounds, dtype=torch.float64
    )


def check_projection():
    weight = torch.randn(24, 20480, device="cuda", dtype=torch.float32) * 0.01
    x = torch.randn(2, 20480, device="cuda", dtype=torch.bfloat16).float()
    expected, bounds = projection_reference(x, weight)
    c2 = native.native_hc_projection_c2(x, weight)
    c1 = torch.cat([native.native_hc_projection(row[None], weight) for row in x])
    assert torch.equal(c1.view(torch.int32), c2.view(torch.int32))
    errors = (c2.cpu().double() - expected).abs()
    wrong = (~torch.isfinite(c2.cpu())) | (errors > bounds)
    print(
        json.dumps(
            {
                "event": "hc_projection",
                "violations": int(wrong.sum()),
                "max_abs_error": float(errors.max()),
                "max_error_bound": float(bounds.max()),
                "failed_values": [
                    [
                        int(i),
                        int(j),
                        float(c2[i, j]),
                        float(expected[i, j]),
                        float(bounds[i, j]),
                    ]
                    for i, j in wrong.nonzero().tolist()
                ],
            }
        ),
        flush=True,
    )
    assert not wrong.any()


def coeff_reference(x, weight, scale, bias, rms_eps):
    x, weight, scale, bias = [t.cpu().double() for t in (x, weight, scale, bias)]
    flat = x.flatten(1)
    mix = (flat @ weight.T) / (flat.square().mean(-1, keepdim=True) + rms_eps).sqrt()
    pre = (scale[0] * mix[:, :4] + bias[:4]).sigmoid() + 1e-6
    post = 2 * (scale[1] * mix[:, 4:8] + bias[4:8]).sigmoid()
    logits = (scale[2] * mix[:, 8:] + bias[8:]).reshape(-1, 4, 4)
    comb = (logits - logits.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + 1e-6
    for step in range(20):
        if step:
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return pre, post, comb


def norm_reference(x, weight, eps):
    # The BF16 combine is observable before the caller's RMSNorm.
    x = x.to(torch.bfloat16).float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (x * weight.cpu().float()).to(torch.bfloat16)


def post_check(actual, args, case, literal=None):
    owned = actual.detach().cpu().clone()
    baseline = mhc.hc_post(*args, 4).view_as(args[1]).cpu()
    bad = (owned.view(torch.int16) != baseline.view(torch.int16)).nonzero()
    print(
        json.dumps(
            {
                "event": "hc_post_bitwise",
                "case": case,
                "unequal": len(bad),
                "first": [
                    [index, float(owned[tuple(index)]), float(baseline[tuple(index)])]
                    for index in bad[:16].tolist()
                ],
            }
        ),
        flush=True,
    )
    assert not len(bad), case
    x, residual, post, comb = [t.detach().cpu().double() for t in args]
    expected = post.unsqueeze(-1) * x.unsqueeze(1)
    expected += torch.einsum("sij,sih->sjh", comb, residual)
    torch.testing.assert_close(owned.double(), expected, rtol=0.01, atol=0.03)
    if literal is not None:
        assert torch.equal(owned.view(torch.int16), literal.cpu().view(torch.int16))
    return owned


def check_post(layer, rows):
    def inputs(rows=rows, dtype=torch.bfloat16):
        return (
            torch.randn(rows, 5120, device="cuda", dtype=dtype),
            torch.randn(rows, 4, 5120, device="cuda", dtype=dtype),
            torch.rand(rows, 4, device="cuda") * 2,
            torch.rand(rows, 4, 4, device="cuda") / 4,
        )

    def check(args, case, literal=None):
        before = [t.cpu().clone() for t in args]
        with patch.object(
            native_post, "native_hc_post_c2", wraps=native_post.native_hc_post_c2
        ) as spy:
            actual = Layer.hc_post(layer, *args)
            assert spy.call_count == 1
        result = post_check(actual, args, f"C{rows}:" + case, literal)
        for old, current in zip(before, args, strict=True):
            assert torch.equal(old.view(torch.uint8), current.cpu().view(torch.uint8))
        return result

    x, residual, post, comb = inputs()
    pattern = torch.arange(5120)
    zero = torch.zeros(5120, dtype=torch.bfloat16)
    x.copy_(torch.where((pattern & 16) != 0, -zero, zero).expand(rows, -1))
    residual.copy_(
        torch.stack(
            [torch.where((pattern & (1 << i)) != 0, -zero, zero) for i in range(4)]
        ).expand(rows, -1, -1)
    )
    post.fill_(1)
    comb.fill_(1)
    args = (x, residual, post, comb)
    check(args, "all32_signed_zero_patterns", torch.zeros_like(residual))
    x.zero_()
    post.zero_()
    for i, value in enumerate([65536.0, 2**-9, -65536.0, 2**-9]):
        residual[:, i].fill_(value)
    check(args, "sequential_reduction", torch.full_like(residual, 2**-9))
    residual.zero_()
    comb.fill_(0.5)
    residual[:, 0].fill_(2**-133)
    residual[:, 1].fill_(2**-133)
    check(args, "gradual_underflow", torch.full_like(residual, 2**-133))
    x.fill_(1)
    residual.zero_()
    post.fill_(1 + 2**-8)
    comb.zero_()
    check(args, "BF16_even_midpoint", torch.ones_like(residual))
    original = inputs()
    expected = check(original, "random")
    swapped = check(tuple(t.flip(0).contiguous() for t in original), "row_swap")
    assert torch.equal(swapped.view(torch.int16), expected.flip(0).view(torch.int16))

    for name, values in (
        ("C0", inputs(0)),
        ("FP16", inputs(dtype=torch.float16)),
    ):
        with patch.object(
            native_post, "native_hc_post_c2", wraps=native_post.native_hc_post_c2
        ) as spy:
            actual = Layer.hc_post(layer, *values)
            assert spy.call_count == 0, name
        expected = mhc.hc_post(*values, 4).view_as(values[1])
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name
    for name in ("strided", "raw_text_model", "other_model", "previous_false"):
        values = list(inputs())
        old_type, old_previous = (
            layer.config.model_type,
            layer.hc_pre_from_prev_sublayer,
        )
        try:
            if name == "strided":
                storage = torch.zeros(rows, 10240, device="cuda", dtype=torch.bfloat16)
                storage[:, ::2].copy_(values[0])
                values[0] = storage[:, ::2]
            if name == "raw_text_model":
                layer.config.model_type = "deepseek_v41_text"
            if name == "other_model":
                layer.config.model_type = "deepseek_v4"
            if name == "previous_false":
                layer.hc_pre_from_prev_sublayer = False
            with patch.object(
                native_post, "native_hc_post_c2", wraps=native_post.native_hc_post_c2
            ) as spy:
                actual = Layer.hc_post(layer, *values)
                assert spy.call_count == 0, name
            post_check(actual, values, f"C{rows}:fallback:" + name)
        finally:
            layer.config.model_type, layer.hc_pre_from_prev_sublayer = (
                old_type,
                old_previous,
            )

    buffers = tuple(t.clone() for t in original)
    changed = inputs()
    graphs = []
    for call in (
        lambda *v: mhc.hc_post(*v, 4).view_as(v[1]),
        lambda *v: Layer.hc_post(layer, *v),
    ):
        call(*buffers)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call(*buffers)
        graphs.append((graph, output))
    for name, values in (("A", original), ("B", changed), ("A_again", original)):
        for target, value in zip(buffers, values, strict=True):
            target.copy_(value)
        graphs[0][0].replay()
        old = graphs[0][1].cpu().clone()
        graphs[1][0].replay()
        actual = post_check(graphs[1][1], buffers, f"C{rows}:graph:" + name)
        assert torch.equal(actual.view(torch.int16), old.view(torch.int16))
        eager = Layer.hc_post(layer, *buffers).cpu()
        assert torch.equal(actual.view(torch.int16), eager.view(torch.int16))
    # Each new row is also compared to the existing exact C2 geometry.
    parts = []
    for row in range(rows):
        pair = tuple(
            t[row : row + 1].repeat((2,) + (1,) * (t.ndim - 1)) for t in original
        )
        parts.append(native_post.native_hc_post_c2(*pair)[:1].cpu().clone())
    whole = Layer.hc_post(layer, *original).cpu().clone()
    assert torch.equal(whole.view(torch.int16), torch.cat(parts).view(torch.int16))
    if rows > 1:
        separate = tuple(t.clone() for t in original)
        for tensor in separate:
            tensor[0].add_(0.125)
        own = post_check(
            Layer.hc_post(layer, *separate), separate, f"C{rows}:one_row_change"
        )
        assert torch.equal(own[1:].view(torch.int16), whole[1:].view(torch.int16))
    print(
        json.dumps(
            {
                "event": "post_rows_complete",
                "rows": rows,
                "bitwise_C2_rows": True,
                "unchanged_FP64_gate": True,
                "fallbacks": 6,
                "graph_replays": 3,
            }
        ),
        flush=True,
    )


def projection_bits(actual, expected, case):
    a, b = [t.detach().cpu().contiguous() for t in (actual, expected)]
    assert a.shape == b.shape and a.dtype == b.dtype, case
    wrong = a.view(torch.uint8) != b.view(torch.uint8)
    print(
        json.dumps(
            {
                "event": "hc_rows_bitwise",
                "case": case,
                "unequal_bytes": int(wrong.sum()),
            }
        ),
        flush=True,
    )
    assert not wrong.any(), case


def check_projection_rows(layer):
    weight = torch.randn(24, 20480, device="cuda") * 0.01
    a = torch.randn(8, 4, 5120, device="cuda", dtype=torch.bfloat16)
    b = (a.flip(0).roll(17, 2) * 0.5).contiguous()
    scale = torch.tensor([0.1, 0.2, 0.125], device="cuda")
    bias = torch.randn(24, device="cuda")
    originals = [t.cpu().clone() for t in (a, b, weight, scale, bias)]
    references, row_controls = [], []
    for x in (a, b):
        flat = x.flatten(1).float()
        references.append(projection_reference(flat, weight))
        c1 = torch.cat([native.native_hc_projection(row[None], weight) for row in flat])
        c2 = torch.cat(
            [
                native.native_hc_projection_c2(flat[i : i + 2], weight)
                for i in range(0, 8, 2)
            ]
        )
        projection_bits(c2, c1, "original_C1_C2")
        row_controls.append(c1)
    records = []
    for rows in range(3, 9):
        expected = []
        for i, x in enumerate((a[:rows], b[:rows])):
            flat = x.flatten(1).float()
            actual = native.native_hc_projection_rows(flat, weight)
            projection_bits(actual, row_controls[i][:rows], f"C{rows}:projection:{i}")
            ref, bound = (v[:rows] for v in references[i])
            errors = (actual.cpu().double() - ref).abs()
            wrong = (~torch.isfinite(actual.cpu())) | (errors > bound)
            print(
                json.dumps(
                    {
                        "event": "hc_rows_reference",
                        "rows": rows,
                        "pattern": i,
                        "violations": int(wrong.sum()),
                        "max_abs_error": float(errors.max()),
                        "max_error_bound": float(bound.max()),
                    }
                ),
                flush=True,
            )
            assert not wrong.any()
            with patch.object(
                native,
                "native_hc_projection_rows",
                wraps=native.native_hc_projection_rows,
            ) as spy:
                stats = Layer._hc_mix_stats(layer, x, weight, scale, bias)
                assert spy.call_count == 1
            with patch.object(
                native, "native_hc_projection_rows", return_value=row_controls[i][:rows]
            ):
                controls = Layer._hc_mix_stats(layer, x, weight, scale, bias)
            ideal = coeff_reference(x, weight, scale, bias, layer.rms_norm_eps)
            for actual_stat, control, reference in zip(
                stats, controls, ideal, strict=True
            ):
                projection_bits(actual_stat, control, f"C{rows}:stats:{i}")
                torch.testing.assert_close(
                    actual_stat.cpu().double(), reference, rtol=1e-4, atol=2e-6
                )
            expected.append(tuple(v.cpu().clone() for v in stats))
        flat = a[:rows].flatten(1).float()
        swapped = native.native_hc_projection_rows(flat.flip(0).contiguous(), weight)
        projection_bits(swapped, row_controls[0][:rows].flip(0), f"C{rows}:swapped")
        flat[-1].copy_(b[rows - 1].flatten().float())
        changed = native.native_hc_projection_rows(flat, weight)
        projection_bits(
            changed[:-1], row_controls[0][: rows - 1], f"C{rows}:other_rows"
        )
        buffer = a[:rows].clone()
        Layer._hc_mix_stats(layer, buffer, weight, scale, bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = Layer._hc_mix_stats(layer, buffer, weight, scale, bias)
        for pattern in (0, 1, 0):
            buffer.copy_((a, b)[pattern][:rows])
            graph.replay()
            for result, control in zip(output, expected[pattern], strict=True):
                projection_bits(result, control, f"C{rows}:graph:{pattern}")
        records.append((graph, buffer, output, expected[0]))
    # Later capture allocations must not replace an earlier graph's live owners.
    for graph, buffer, output, expected in reversed(records):
        projection_bits(buffer, a[: len(buffer)], "retained_input")
        graph.replay()
        for actual, control in zip(output, expected, strict=True):
            projection_bits(actual, control, "retained_graph")
    for current, original in zip((a, b, weight, scale, bias), originals, strict=True):
        projection_bits(current, original, "immutable_inputs")

    storage = torch.empty(3, 4, 10240, device="cuda", dtype=torch.float32)
    storage[:, :, ::2].copy_(a[:3])
    for case, x, w, model in (
        ("C0", a[:0], weight, "deepseek_v41"),
        ("C9", torch.cat((a, a[:1])), weight, "deepseek_v41"),
        ("strided_input", storage[:, :, ::2], weight, "deepseek_v41"),
        ("strided_weight", a[:3], weight.t().contiguous().t(), "deepseek_v41"),
        ("other_model", a[:3], weight, "deepseek_v4"),
    ):
        with (
            patch.object(layer.config, "model_type", model),
            patch.object(
                native,
                "native_hc_projection_rows",
                wraps=native.native_hc_projection_rows,
            ) as spy,
        ):
            actual = Layer._hc_mix_stats(layer, x, w, scale, bias)
            assert spy.call_count == 0, case
        with patch.object(layer.config, "model_type", "deepseek_v4"):
            expected = Layer._hc_mix_stats(layer, x, w, scale, bias)
        for result, control in zip(actual, expected, strict=True):
            projection_bits(result, control, case)
    for x, w in (
        (a[:2].flatten(1).float(), weight),
        (a.flatten(1).half(), weight),
        (a.flatten(1).float(), weight.t().contiguous().t()),
    ):
        try:
            native.native_hc_projection_rows(x, w)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid direct HC operands accepted")


def check_projection_literals():
    weight = torch.zeros(24, 20480, device="cuda")
    x = torch.zeros(8, 20480, device="cuda")
    weight[:, [0, 5120, 10240, 15360]] = torch.tensor([1, -1, 2, 0.5], device="cuda")
    for i in range(8):
        x[i, [0, 5120, 10240, 15360]] = torch.tensor(
            [i + 1, i + 1, i, 2], device="cuda", dtype=torch.float32
        )
    expected = (2 * torch.arange(8, device="cuda") + 1).float()[:, None].expand(-1, 24)
    for rows in range(3, 9):
        projection_bits(
            native.native_hc_projection_rows(x[:rows], weight),
            expected[:rows],
            f"C{rows}:split_cancellation",
        )
    x.zero_()
    weight.zero_()
    x[:, 0] = 2**-126
    weight[:, 0] = 0.5
    projection_bits(
        native.native_hc_projection_rows(x, weight),
        torch.full((8, 24), 2**-127, device="cuda"),
        "projection_underflow",
    )
    x.zero_()
    x[1::2, 0] = -0.0
    weight.fill_(1)
    c1 = torch.cat([native.native_hc_projection(row[None], weight) for row in x])
    projection_bits(
        native.native_hc_projection_rows(x, weight), c1, "projection_signed_zero"
    )
    x.zero_()
    weight.zero_()
    weight[:, 0] = 1
    x[0, 0], x[1, 0], x[2, 0] = float("inf"), -float("inf"), float("nan")
    c1 = torch.cat([native.native_hc_projection(row[None], weight) for row in x])
    for rows in range(3, 9):
        actual = native.native_hc_projection_rows(x[:rows], weight)
        expected = c1[:rows]
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isinf(actual), torch.isinf(expected))
        assert torch.equal(
            torch.signbit(actual)[torch.isinf(actual)],
            torch.signbit(expected)[torch.isinf(expected)],
        )
        assert torch.equal(
            actual[torch.isfinite(expected)], expected[torch.isfinite(expected)]
        )


def main():
    runtime = Path(sys.argv[1]).resolve()
    assert Path(inspect.getfile(Layer)).resolve().is_relative_to(runtime)
    assert "aiter.dist.device_communicators.custom_all_reduce" not in sys.modules
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    torch.cuda.set_per_process_memory_fraction(0.03)
    torch.set_num_threads(4)
    torch.manual_seed(431)
    # Exercise the installed normalization, including the nested text type.
    config = DeepseekV41Config(
        text_config={
            "model_type": "deepseek_v41_text",
            "hidden_size": 5120,
            "hc_mult": 4,
        }
    )
    assert config.model_type == "deepseek_v41"
    layer = SimpleNamespace(
        config=config,
        hidden_size=5120,
        hc_mult=4,
        rms_norm_eps=1e-20,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        hc_pre_from_prev_sublayer=True,
    )
    check_projection()
    print(
        json.dumps(
            {"event": "provenance", "runtime": str(runtime), "torch": torch.__version__}
        ),
        flush=True,
    )
    cases = 0
    maximum = 0.0
    for tokens in (0, 1, 2, 5, 33, 128):
        for eps in (1e-6, 1e-20):
            layer.rms_norm_eps = eps
            for bias_scale in (1, 10):
                x = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
                if tokens > 1:
                    x[0].zero_()
                fn = torch.randn(24, 20480, device="cuda") * 0.01
                scale = torch.tensor([0.1, 0.2, 0.125], device="cuda")
                bias = torch.randn(24, device="cuda") * bias_scale
                norm = RMSNorm(5120, eps=eps).to(device="cuda", dtype=torch.bfloat16)
                norm.weight.data.uniform_(0.5, 1.5)
                actual = Layer.hc_pre(layer, x, fn, scale, bias, norm=norm)
                assert actual[3] is False
                pre, post, comb = coeff_reference(x, fn, scale, bias, eps)
                expected = (pre.unsqueeze(-1) * x.cpu().double()).sum(1)
                for a, b in zip(actual[:3], (expected, post, comb), strict=True):
                    torch.testing.assert_close(
                        a.cpu().double(),
                        b,
                        rtol=0.01 if a.dtype == torch.bfloat16 else 1e-4,
                        atol=0.03 if a.dtype == torch.bfloat16 else 2e-6,
                    )
                with (
                    patch.object(
                        native,
                        "native_hc_projection",
                        wraps=native.native_hc_projection,
                    ) as c1,
                    patch.object(
                        native,
                        "native_hc_projection_c2",
                        wraps=native.native_hc_projection_c2,
                    ) as c2,
                ):
                    stats = Layer._hc_mix_stats(layer, x, fn, scale, bias)
                    assert c1.call_count == int(tokens == 1)
                    assert c2.call_count == int(tokens == 2)
                for a, b in zip(stats, (pre, post, comb), strict=True):
                    torch.testing.assert_close(
                        a.cpu().double(), b, rtol=1e-4, atol=2e-6
                    )
                    if a.numel():
                        maximum = max(
                            maximum, float((a.cpu().double() - b).abs().max())
                        )
                residual = torch.randn_like(x)
                y, p, c, _ = actual
                result = Layer.hc_post(layer, y, residual, p, c)
                ref_post = p.cpu().double().unsqueeze(-1) * y.cpu().double().unsqueeze(
                    1
                )
                ref_post += torch.einsum(
                    "sij,sih->sjh", c.cpu().double(), residual.cpu().double()
                )
                torch.testing.assert_close(
                    result.cpu().double(), ref_post, rtol=0.01, atol=0.03
                )
                if tokens:
                    # V4.1 consumes the PREVIOUS sublayer's pre coefficients.
                    previous_pre = torch.rand(tokens, 4, device="cuda")
                    combined = Layer._hc_combine(layer, x, previous_pre, norm)
                    previous_ref = (
                        previous_pre.cpu().double().unsqueeze(-1) * x.cpu().double()
                    ).sum(1)
                    torch.testing.assert_close(
                        combined.cpu(),
                        norm_reference(previous_ref, norm.weight, eps),
                        rtol=0.015,
                        atol=0.03,
                    )
                    initial = Layer._hc_combine(layer, x, None, norm)
                    torch.testing.assert_close(
                        initial.cpu(),
                        norm_reference(x[:, 0].cpu(), norm.weight, eps),
                        rtol=0.015,
                        atol=0.03,
                    )
                if tokens in (1, 2, 5) and eps == 1e-20 and bias_scale == 1:

                    def forward(
                        x=x, fn=fn, scale=scale, bias=bias, y=y, residual=residual
                    ):
                        a = Layer._hc_mix_stats(layer, x, fn, scale, bias)
                        b = Layer.hc_post(layer, y, residual, a[1], a[2])
                        return (*a, b)

                    forward()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = forward()
                    for change in (0.25, -0.75, 1.5):
                        x.mul_(change)
                        residual.add_(change)
                        bias.add_(change)
                        eager = forward()
                        graph.replay()
                        for a, b in zip(captured, eager, strict=True):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                cases += 1
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "eps": eps,
                            "bias_scale": bias_scale,
                            "passed": True,
                        }
                    ),
                    flush=True,
                )
    # decode rows 1..8 and prefill tails of any size (patch 0047 removed the (1, 2, 4, 512, 1306, 1536) whitelist)
    for rows in (*range(1, 9), 9, 17, 100, 513):
        check_post(layer, rows)
    check_projection_rows(layer)
    check_projection_literals()
    print(
        json.dumps(
            {
                "event": "complete",
                "cases": cases,
                "max_coefficient_abs_error": maximum,
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
