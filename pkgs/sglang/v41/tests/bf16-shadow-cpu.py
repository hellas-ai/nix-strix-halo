#!/usr/bin/env python3
"""CPU checks of the opt-in bf16 shadow path for the dense FP8 prefill GEMMs.

No GPU: the Triton dequantizer runs in Triton's interpreter, and the installed
helper functions are extracted from source (importing fp8_kernel needs a device).

  * the activation dequantizer decodes all 256 E4M3FN codes exactly, with
    arbitrary scale strides, and NaN codes stay NaN;
  * the weight expansion is exact (value * 2^k, bf16) and refuses non-power-of-two
    scales;
  * the gate in `triton_w8a8_block_fp8_linear` is off by default, takes the shadow
    only at M >= threshold with a built shadow, and never during graph capture or
    with a bias / non-bf16 output / non-UE8M0 activations;
  * dequantize -> bf16 GEMM stays within bf16 rounding of the exact product.

    sglang-python bf16-shadow-cpu.py /nix/store/...-sglang-v41-rocm-gfx1151
"""

import ast
import functools
import math
import os
import sys
from pathlib import Path
from unittest import mock

os.environ["TRITON_INTERPRET"] = "1"
os.environ.pop("SGLANG_DSV41_BF16_SHADOW_MIN_M", None)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
sys.path.insert(0, str(SITE))
from sglang.kernels.ops.gemm.deepseek_v41_gemv import _e4m3fn_to_fp32  # noqa: E402

assert not torch.cuda.is_initialized()


def extract(relative, names, namespace):
    """Exec the named top-level defs/assignments of an installed file, in order."""
    path = SITE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        target = (
            node.name
            if isinstance(node, ast.FunctionDef)
            else node.targets[0].id
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None
        )
        if target in names:
            body.append(node)
    found = {
        n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in body
    }
    assert found == set(names), (set(names) - found, relative)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *body], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 -- installed helpers, no device
    return namespace


ns = {
    "torch": torch,
    "triton": triton,
    "tl": tl,
    "os": os,
    "functools": functools,
    "List": list,
    "_e4m3fn_to_fp32": _e4m3fn_to_fp32,
}
extract(
    "sglang/kernels/ops/gemm/fp8_kernel.py",
    [
        "GFX1151_V41_DENSE_SHAPES",
        "dsv41_bf16_shadow_min_m",
        "dequant_block_fp8_weight_to_bf16",
        "block_fp8_scale_is_power_of_two",
        "_dequant_e4m3fn_group32_bf16",
        "dequant_group32_activation_to_bf16",
    ],
    ns,
)
# The shape set must equal the literal in the prefill-cast predicate (see fp8_kernel.py).
_src = (SITE / "sglang/kernels/ops/gemm/fp8_kernel.py").read_text()
_fn = next(
    n
    for n in ast.parse(_src).body
    if isinstance(n, ast.FunctionDef) and n.name == "_use_gfx1151_fp8_prefill_cast"
)
_literals = [n for n in ast.walk(_fn) if isinstance(n, ast.Set)]
assert len(_literals) == 1
assert set(ast.literal_eval(_literals[0])) == set(ns["GFX1151_V41_DENSE_SHAPES"])

dequant_act = ns["dequant_group32_activation_to_bf16"]
dequant_weight = ns["dequant_block_fp8_weight_to_bf16"]
is_pow2 = ns["block_fp8_scale_is_power_of_two"]
min_m = ns["dsv41_bf16_shadow_min_m"]

# ---- activation dequantizer: every code, every scale, odd strides -----------------------
codes = torch.arange(256, dtype=torch.int16).to(torch.uint8)
for exponent in (-20, -9, 0, 7, 30):
    scale = 2.0**exponent
    q = codes.repeat(4)[:1024].view(1, 1024).view(torch.float8_e4m3fn)  # 32 groups of 32
    s = torch.full((1, 32), scale, dtype=torch.float32)
    got = dequant_act(q.contiguous(), s)
    ref_f32 = q.to(torch.float32)  # torch's own decode, CPU
    expected = (ref_f32 * scale).to(torch.bfloat16)
    nan = torch.isnan(ref_f32)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got[~nan].view(torch.int16), expected[~nan].view(torch.int16)), exponent
    # Negative zero keeps its sign through the exact decoder.
    assert got[0, 128].item() == 0.0 and math.copysign(1.0, got[0, 128].item()) == -1.0

# Scale stored transposed (column-major) and a row stride larger than K.
m, k = 5, 160
base = torch.randint(0, 256, (m, k + 64), dtype=torch.int16).to(torch.uint8)
base[base == 0x7F] = 0x3F
base[base == 0xFF] = 0xBF
q = base[:, :k].view(torch.float8_e4m3fn)  # row stride k + 64
sc_t = 2.0 ** torch.randint(-12, 12, (k // 32, m)).to(torch.float32)
sc = sc_t.t()  # (m, k // 32) with stride (1, m)
out = dequant_act(q, sc)
expected = (
    q.to(torch.float32).view(m, k // 32, 32) * sc.unsqueeze(-1)
).view(m, k).to(torch.bfloat16)
assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
assert dequant_act(q[:0], sc[:0]).shape == (0, k)

# ---- weight expansion -------------------------------------------------------------------
gen = torch.Generator().manual_seed(3)
n, k = 96, 320
raw = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.int16).to(torch.uint8)
raw[raw == 0x7F] = 0x01
raw[raw == 0xFF] = 0x81
w = raw.view(torch.float8_e4m3fn)
exp = torch.randint(-24, 12, (n // 32, k // 32), generator=gen)
scale = (2.0 ** exp.to(torch.float64)).to(torch.float32)
shadow = dequant_weight(w, scale, [32, 32])
exact = w.to(torch.float64).view(n // 32, 32, k // 32, 32) * scale.to(
    torch.float64
).view(n // 32, 1, k // 32, 1)
exact = exact.view(n, k)
assert shadow.dtype == torch.bfloat16 and shadow.shape == (n, k)
assert torch.equal(shadow.to(torch.float64), exact), "shadow weight is not exact"
assert is_pow2(scale)
for bad in (1.5, 0.0, float("inf"), float("nan"), 3.0, -2.0 * 1.0000001):
    s2 = scale.clone()
    s2[1, 2] = bad
    assert not is_pow2(s2), bad
assert is_pow2(-scale), "negative powers of two keep a zero mantissa"  # sign is irrelevant here

# ---- dequantize -> bf16 GEMM vs exact ---------------------------------------------------
m, k, n = 48, 512, 128
gen = torch.Generator().manual_seed(11)
x = torch.randn(m, k, generator=gen) * (torch.rand(k, generator=gen) * 3).exp()
groups = x.view(m, k // 32, 32)
amax = groups.abs().amax(-1).clamp_min(1e-10)
act_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))  # UE8M0 round-up rule
q = (groups / act_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
raw = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.int16).to(torch.uint8)
raw[(raw & 0x7F) == 0x7F] = 0x30
wq = raw.view(torch.float8_e4m3fn)
wscale = 2.0 ** torch.randint(-12, -4, (n // 32, k // 32), generator=gen).to(torch.float32)
x_hat = dequant_act(q, act_scale)
w_hat = dequant_weight(wq, wscale, [32, 32])
# CPU bf16 matmul may accumulate in bf16; the GPU BLAS accumulates in FP32 (checked by
# bf16-shadow-numerics.py). Model that with an FP32 GEMM of the exact operands.
y = torch.nn.functional.linear(x_hat.float(), w_hat.float()).to(torch.bfloat16)
exact = x_hat.to(torch.float64) @ w_hat.to(torch.float64).t()
# Worst-case bound: FP32 accumulation of K exact products (gamma_K * sum|terms|),
# then one round-to-nearest to bf16 (unit roundoff 2^-8: 8 significand bits).
l1 = x_hat.abs().to(torch.float64) @ w_hat.abs().to(torch.float64).t()
gamma = k * 2.0**-24 * l1
bound = 2.0**-8 * (exact.abs() + gamma) + gamma
excess = ((y.to(torch.float64) - exact).abs() - bound).max().item()
assert excess <= 0.0, f"bf16 shadow GEMM leaves its rounding bound by {excess}"
rms = exact.pow(2).mean(1, keepdim=True).sqrt()
assert ((y.to(torch.float64) - exact).abs() / rms).max().item() <= 2.0**-5

# ---- dispatch gate and per-layer wrapper -------------------------------------------------
quantizer_calls = []


def fake_quantizer(x, group_size, scale_ue8m0=False):
    """CPU stand-in for sglang_per_token_group_quant_fp8 (UE8M0 round-up rule)."""
    assert group_size == 32 and scale_ue8m0
    quantizer_calls.append(tuple(x.shape))
    rows, cols = x.shape
    groups = x.to(torch.float32).view(rows, cols // 32, 32)
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-10) / 448.0)))
    q = (groups / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(rows, cols), scale


gate_ns = {
    "torch": torch,
    "dsv41_bf16_shadow_min_m": min_m,
    "dequant_group32_activation_to_bf16": dequant_act,
    "sglang_per_token_group_quant_fp8": fake_quantizer,
}
# Class + gate come from the installed file; the gate is a plain function.
path = SITE / "sglang/srt/layers/quantization/fp8_utils.py"
tree = ast.parse(path.read_text(), filename=str(path))
wanted = {"dsv41_bf16_shadow_applies", "Dsv41Bf16ShadowLinear"}
nodes = [n for n in tree.body if getattr(n, "name", None) in wanted]
assert {n.name for n in nodes} == wanted
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(  # noqa: S102 -- installed helpers with stubs, no device
    compile(
        ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
        str(path),
        "exec",
    ),
    gate_ns,
)
gate = gate_ns["dsv41_bf16_shadow_applies"]
Wrapper = gate_ns["Dsv41Bf16ShadowLinear"]
fp8 = torch.float8_e4m3fn


def gate_args(rows=256, k_dim=512, n_dim=128, **overrides):
    args = dict(
        shadow=torch.zeros(n_dim, k_dim, dtype=torch.bfloat16),
        q_input=torch.zeros(rows, k_dim, dtype=fp8),
        x_scale=torch.ones(rows, k_dim // 32, dtype=torch.float32),
        block_size=[32, 32],
        output_dtype=torch.bfloat16,
        bias=None,
    )
    args.update(overrides)
    return args


with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
    min_m.cache_clear()
    assert min_m() == 0 and not gate(**gate_args()), "must be off unless the env var is set"
    for text, expected in (("", 0), ("0", 0), ("-7", 0), ("256", 256)):
        os.environ["SGLANG_DSV41_BF16_SHADOW_MIN_M"] = text
        min_m.cache_clear()
        assert min_m() == expected, (text, min_m())
    os.environ["SGLANG_DSV41_BF16_SHADOW_MIN_M"] = "128"
    min_m.cache_clear()
    assert gate(**gate_args(128)) and gate(**gate_args(1536))
    assert not gate(**gate_args(127)) and not gate(**gate_args(16))
    assert not gate(**gate_args(shadow=None)), "no shadow built"
    assert not gate(**gate_args(block_size=[128, 128]))
    assert not gate(**gate_args(bias=torch.zeros(1)))
    assert not gate(**gate_args(output_dtype=torch.float16))
    assert not gate(**gate_args(x_scale=torch.ones(256, 4, dtype=torch.float32))), "group-128"
    assert not gate(**gate_args(x_scale=torch.ones(256, 16, dtype=torch.float16)))
    assert not gate(**gate_args(shadow=torch.zeros(8, 64, dtype=torch.bfloat16))), "stale K"

    # The wrapper: fallback below the threshold (same kwargs), shadow above it.
    calls = []

    def fallback(**kwargs):
        calls.append(kwargs)
        return torch.full((kwargs["input"].numel() // kwargs["input"].shape[-1], 128), -1.0, dtype=torch.bfloat16)

    gen = torch.Generator().manual_seed(5)
    n_dim, k_dim = 128, 512
    raw = torch.randint(0, 256, (n_dim, k_dim), generator=gen, dtype=torch.int16).to(torch.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x30
    wq = raw.view(fp8)
    wscale = 2.0 ** torch.randint(-12, -4, (n_dim // 32, k_dim // 32), generator=gen).to(torch.float32)
    shadow_w = dequant_weight(wq, wscale, [32, 32])
    wrapper = Wrapper(fallback, shadow_w, wq)
    kw = dict(weight=wq, block_size=[32, 32], weight_scale=wscale)

    small = torch.randn(64, k_dim, generator=gen).to(torch.bfloat16)
    out = wrapper(small, **kw)
    assert calls and calls[-1]["input"] is small and calls[-1]["input_scale"] is None
    assert float(out[0, 0]) == -1.0 and not quantizer_calls, "below threshold: fallback, no quantizer"

    big = (torch.randn(200, k_dim, generator=gen) * 5).to(torch.bfloat16)
    calls.clear()
    out = wrapper(big, **kw)
    assert not calls and out.shape == (200, n_dim) and out.dtype == torch.bfloat16
    q_ref, s_ref = fake_quantizer(big, 32, True)
    ref = torch.nn.functional.linear(
        dequant_act(q_ref, s_ref).float(), shadow_w.float()
    ).to(torch.bfloat16)
    err = (out.double() - ref.double()).abs().max().item()
    assert err <= 2.0**-6 * ref.abs().max().item(), err  # same operands, FP32 order only

    out3 = wrapper(big.view(8, 25, k_dim), **kw)  # batched input keeps leading dims
    assert out3.shape == (8, 25, n_dim)

    pre_q, pre_s = fake_quantizer(big, 32, True)
    quantizer_calls.clear()
    outp = wrapper(pre_q, **kw, input_scale=pre_s)  # pre-quantized input skips the quantizer
    assert not quantizer_calls and torch.equal(outp.view(torch.int16), out.view(torch.int16))

    calls.clear()
    wrapper(big.to(torch.float16), **kw)  # non-bf16 activations stay on the generic kernel
    assert calls
    # A different weight (the DSpark draft passes a stacked wkv through the same
    # callable) must never get this layer's shadow: same shape, other storage ...
    calls.clear()
    other = wq.clone()
    wrapper(big, weight=other, block_size=[32, 32], weight_scale=wscale)
    assert calls and calls[-1]["weight"] is other
    # ... and another shape.
    calls.clear()
    wrapper(big, weight=wq[:64], block_size=[32, 32], weight_scale=wscale[:2])
    assert calls
    calls.clear()
    wrapper(big, bias=torch.zeros(n_dim), **kw)
    assert calls and calls[-1]["bias"] is not None
with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
    calls.clear()
    quantizer_calls.clear()
    wrapper(big, **kw)
    assert len(calls) == 1 and calls[0]["input"] is big, "never during CUDA graph capture"
    assert calls[0]["input_scale"] is None and calls[0]["weight"] is wq
    assert not quantizer_calls, "capture must not quantize before the fallback quantizes again"

    calls.clear()
    wrapper(pre_q, **kw, input_scale=pre_s)
    assert len(calls) == 1 and calls[0]["input"] is pre_q
    assert calls[0]["input_scale"] is pre_s and calls[0]["weight"] is wq
    assert not quantizer_calls, "pre-quantized capture must reach the fallback unchanged"

# ---- build step: which layers get a shadow ---------------------------------------------
import functools as _ft  # noqa: E402
import logging  # noqa: E402
import types  # noqa: E402


def stub_block_linear(**kwargs):
    raise AssertionError("never called")


# `func is triton_w8a8_block_fp8_linear`: use a stand-in with the same identity role.
build_ns = {
    "torch": torch,
    "logger": logging.getLogger("bf16-shadow-test"),
    "_is_hip": True,
    "dsv41_bf16_shadow_min_m": min_m,
    "GFX1151_V41_DENSE_SHAPES": ns["GFX1151_V41_DENSE_SHAPES"],
    "block_fp8_scale_is_power_of_two": is_pow2,
    "dequant_block_fp8_weight_to_bf16": dequant_weight,
    "Dsv41Bf16ShadowLinear": Wrapper,
    "triton_w8a8_block_fp8_linear": stub_block_linear,
    "_DSV41_SHADOW_BUILT": {"layers": 0, "bytes": 0, "shapes": set()},
}
build_nodes = [
    n for n in tree.body if getattr(n, "name", None) == "maybe_build_dsv41_bf16_shadow"
]
assert len(build_nodes) == 1
exec(  # noqa: S102 -- installed helper with stubs, no device
    compile(
        ast.fix_missing_locations(ast.Module(body=[future, *build_nodes], type_ignores=[])),
        str(path),
        "exec",
    ),
    build_ns,
)
build = build_ns["maybe_build_dsv41_bf16_shadow"]


class FakeCudaWeight:
    """A CPU weight that claims to live on the gfx1151 GPU (build step only reads metadata)."""

    is_cuda = True
    device = "cuda:0"

    def __init__(self, tensor):
        self.data = tensor
        self.dtype, self.shape = tensor.dtype, tensor.shape

    def dim(self):
        return self.data.dim()

    def data_ptr(self):
        return self.data.data_ptr()


def make_layer(n_dim=5120, k_dim=576, power_of_two=True):
    gen = torch.Generator().manual_seed(n_dim + k_dim)
    raw = torch.randint(0, 256, (n_dim, k_dim), generator=gen, dtype=torch.int16).to(torch.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x30
    exp = torch.randint(-12, -4, (n_dim // 32, k_dim // 32), generator=gen)
    scale = 2.0 ** exp.to(torch.float32)
    if not power_of_two:
        scale[0, 0] = 1.5
    layer = types.SimpleNamespace(
        weight=FakeCudaWeight(raw.view(fp8)),
        weight_scale_inv=types.SimpleNamespace(data=scale, dtype=scale.dtype, shape=scale.shape),
    )
    return layer


def make_method(**overrides):
    partial = _ft.partial(stub_block_linear, act_scale_ue8m0=True)
    method = types.SimpleNamespace(
        w8a8_block_fp8_linear=partial,
        weight_block_size=[32, 32],
        use_mxfp8=False,
        block_fp8_as_mxfp8=False,
    )
    # `getattr(block_linear, "func")` must be the (stand-in) triton function.
    for key, value in overrides.items():
        setattr(method, key, value)
    return method, partial


props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc+:xnack-")
with mock.patch.object(torch.cuda, "get_device_properties", return_value=props):
    os.environ.pop("SGLANG_DSV41_BF16_SHADOW_MIN_M", None)
    min_m.cache_clear()
    layer, (method, original) = make_layer(), (None, None)
    method, original = make_method()
    build(layer, method)
    assert method.w8a8_block_fp8_linear is original, "off by default"

    os.environ["SGLANG_DSV41_BF16_SHADOW_MIN_M"] = "128"
    min_m.cache_clear()
    build(layer, method)
    wrapped = method.w8a8_block_fp8_linear
    assert isinstance(wrapped, Wrapper) and wrapped.fallback is original
    assert layer.dsv41_bf16_shadow is wrapped.shadow and wrapped.shadow.shape == (5120, 576)
    assert torch.equal(
        wrapped.shadow.to(torch.float64),
        dequant_weight(layer.weight.data, layer.weight_scale_inv.data, [32, 32]).to(torch.float64),
    )
    assert build_ns["_DSV41_SHADOW_BUILT"]["layers"] == 1
    build(layer, method)  # weight reload: unwrap, rebuild, one wrapper deep
    assert isinstance(method.w8a8_block_fp8_linear, Wrapper)
    assert method.w8a8_block_fp8_linear.fallback is original
    assert build_ns["_DSV41_SHADOW_BUILT"]["layers"] == 2

    def skipped(layer_, method_):
        before = method_.w8a8_block_fp8_linear
        build(layer_, method_)
        return method_.w8a8_block_fp8_linear is before

    m, _ = make_method()
    assert skipped(make_layer(power_of_two=False), m), "non-power-of-two scales"
    assert skipped(make_layer(5120, 576), make_method(weight_block_size=[128, 128])[0])
    assert skipped(make_layer(5120, 576), make_method(use_mxfp8=True)[0])
    assert skipped(make_layer(5120, 576), make_method(block_fp8_as_mxfp8=True)[0])
    assert skipped(make_layer(64, 128), make_method()[0]), "not one of the seven shapes"
    wrong, _ = make_method()
    wrong.w8a8_block_fp8_linear = _ft.partial(lambda **k: None, act_scale_ue8m0=True)
    assert skipped(make_layer(), wrong), "not the V4.1 triton block linear"
    no_ue8m0, _ = make_method()
    no_ue8m0.w8a8_block_fp8_linear = _ft.partial(stub_block_linear)
    assert skipped(make_layer(), no_ue8m0), "activation scales not UE8M0"
    plain = make_layer()
    plain.keep_plain_weight_layout = True
    assert skipped(plain, make_method()[0]), "wo_a keeps its plain layout"
    cpu_layer = make_layer()
    cpu_layer.weight.is_cuda = False
    assert skipped(cpu_layer, make_method()[0])
with mock.patch.object(
    torch.cuda, "get_device_properties", return_value=types.SimpleNamespace(gcnArchName="gfx1030")
):
    assert skipped(make_layer(), make_method()[0]), "only gfx1151"

os.environ.pop("SGLANG_DSV41_BF16_SHADOW_MIN_M")
min_m.cache_clear()
print("bf16-shadow-cpu OK: exact decode, exact weights, GEMM within bf16 rounding, gate, wrapper, build step")
