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


def check_post(layer):
    def inputs(rows=2, dtype=torch.bfloat16):
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
        result = post_check(actual, args, case, literal)
        for old, current in zip(before, args, strict=True):
            assert torch.equal(old.view(torch.uint8), current.cpu().view(torch.uint8))
        return result

    x, residual, post, comb = inputs()
    pattern = torch.arange(5120)
    zero = torch.zeros(5120, dtype=torch.bfloat16)
    x.copy_(torch.where((pattern & 16) != 0, -zero, zero).expand(2, -1))
    residual.copy_(
        torch.stack(
            [torch.where((pattern & (1 << i)) != 0, -zero, zero) for i in range(4)]
        ).expand(2, -1, -1)
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
        ("C1", inputs(1)),
        ("C3", inputs(3)),
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
                storage = torch.zeros(2, 10240, device="cuda", dtype=torch.bfloat16)
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
            post_check(actual, values, "fallback:" + name)
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
        actual = post_check(graphs[1][1], buffers, "graph:" + name)
        assert torch.equal(actual.view(torch.int16), old.view(torch.int16))
        eager = Layer.hc_post(layer, *buffers).cpu()
        assert torch.equal(actual.view(torch.int16), eager.view(torch.int16))


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
    check_post(layer)
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
