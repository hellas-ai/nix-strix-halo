#!/usr/bin/env python3
"""The bf16 shadow wrapper composes with the native-kernel marking of the block-FP8 callables (CPU, no GPU).

Integration check for the exact-rows series (native marks: `native_gemv_rows`, `native_v41_shared`, `native_idx_wqb`, ...)
and the prefill series (`Dsv41Bf16ShadowLinear`, built after weight loading). Both rewrite the same attribute,
`quant_method.w8a8_block_fp8_linear`, so the order and the shape/pointer guards matter:

  * every mark is made in a constructor, the shadow is built at the end of `process_weights_after_loading_block_quant`,
    so the wrapper always wraps the already-marked partial (checked on the installed source order);
  * the real `DSparkV4Stage._enable_native_projections` and `DeepseekV41Indexer._mark_native_wq_b` run on stand-in layers;
    the real `maybe_build_dsv41_bf16_shadow` then qualifies exactly the eligible marked layers, keeps the marked partial as
    the fallback (flags intact) and leaves ineligible shapes (draft wq_a / wkv / main_proj) on the marked partial itself;
  * below the threshold the wrapper calls the marked partial with the caller's keyword arguments; at or above it the
    shadow GEMM runs; a weight of another shape or storage (the draft's stacked wkv) always reaches the marked partial;
  * the only code that inspects `block_linear.func` / `.keywords` is constructor-time marking (a fixed list of files and
    functions), so nothing sees the wrapper where it expected a partial.

    sglang-python shadow-native-compose-cpu.py /nix/store/...-sglang-v41-rocm-gfx1151
"""

import ast
import functools
import logging
import os
import sys
import types
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
fp8 = torch.float8_e4m3fn
FUTURE = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)


def parse(relative):
    path = SITE / relative
    return path, ast.parse(path.read_text(), filename=str(path))


def run(path, nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[FUTURE, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 -- installed code with stand-in modules, no device
    return namespace


def top(tree, *names):
    found = [n for n in tree.body if getattr(n, "name", None) in names or (
        isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names)]
    assert len(found) == len(names), names
    return found


def method_of(tree, cls, name):
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name), klass


# ---- 1. source order: marks in constructors, the shadow at the end of weight processing ------------------------------
_, fp8_tree = parse("sglang/srt/layers/quantization/fp8.py")
process, _ = method_of(fp8_tree, "Fp8LinearMethod", "process_weights_after_loading_block_quant")
assert "maybe_build_dsv41_bf16_shadow(layer, self)" in ast.unparse(process.body[-1]), "shadow built last"
assert sum("maybe_build_dsv41_bf16_shadow" in ast.unparse(n) for n in ast.walk(process) if isinstance(n, ast.Expr)) == 1

_, dspark_tree = parse("sglang/srt/models/deepseek_v4_dspark.py")
stage_init, _ = method_of(dspark_tree, "DSparkV4Stage", "__init__")
assert "_enable_native_projections" in ast.unparse(stage_init.body[-1]), "draft marks are made in the constructor"

sparse_path, sparse_tree = parse("sglang/srt/layers/attention/dsv4/dsv41_sparse.py")
indexer_init, _ = method_of(sparse_tree, "DeepseekV41Indexer", "__init__")
assert "self._mark_native_wq_b(config, quant_config)" in ast.unparse(indexer_init), "indexer mark is made in the constructor"

# Code that inspects the callable's func / keywords: every occurrence must sit in a constructor-time marker. A new reader
# elsewhere (for example a post-load hook) would see the shadow wrapper instead of a partial and must be reviewed.
CONSTRUCTOR_TIME = {"__init__", "_mark_native_wq_b", "_enable_native_projections"}
readers = {}
for path in (SITE / "sglang").rglob("*.py"):
    text = path.read_text(errors="ignore")
    if "block_linear.func" not in text and "block_linear.keywords" not in text:
        continue
    module = ast.parse(text, filename=str(path))
    scopes = set()
    for outer in ast.walk(module):
        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(outer):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in ("func", "keywords")
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "block_linear"
                ):
                    scopes.add(outer.name)
    readers[str(path.relative_to(SITE))] = scopes
expected_readers = {
    "sglang/srt/models/deepseek_v2.py",
    "sglang/srt/models/deepseek_v4.py",
    "sglang/srt/models/deepseek_v4_dspark.py",
    "sglang/srt/layers/attention/dsv4/dsv41_sparse.py",
    "sglang/srt/layers/engram.py",
}
assert set(readers) == expected_readers, set(readers) ^ expected_readers
for relative, scopes in readers.items():
    # nested helpers (block_linear_of, mark) are reached through their constructor-time owner
    allowed = CONSTRUCTOR_TIME | {"block_linear_of", "mark"}
    assert scopes <= allowed, (relative, scopes - allowed)

# ---- 2. stand-in modules for the real marking code ----------------------------------------------------------------
logging.getLogger().setLevel(logging.ERROR)


class Fp8Config:
    def __init__(self):
        self.is_checkpoint_fp8_serialized = True
        self.activation_scheme = "dynamic"
        self.weight_block_size = [32, 32]
        self.scale_fmt = "ue8m0"


calls = []


def triton_w8a8_block_fp8_linear(**kwargs):
    """Stand-in with the identity role of the real function: records what the marked partial receives."""
    calls.append(kwargs)
    return "fallback-output"


class Fp8LinearMethod:
    def __init__(self):
        self.use_mxfp8 = False
        self.block_fp8_as_mxfp8 = False
        self.weight_block_size = [32, 32]
        self.w8a8_block_fp8_linear = functools.partial(triton_w8a8_block_fp8_linear, act_scale_ue8m0=True)


class ReplicatedLinear:
    def __init__(self, in_size, out_size):
        self.input_size, self.output_size, self.bias = in_size, out_size, None
        self.quant_method = Fp8LinearMethod()


class MergedColumnParallelLinear(ReplicatedLinear):
    def __init__(self):
        super().__init__(5120, 2304 * 2)
        self.output_sizes, self.output_size_per_partition, self.tp_size = [2304, 2304], 1152, 4


class RowParallelLinear(ReplicatedLinear):
    def __init__(self):
        super().__init__(2304, 5120)
        self.input_size_per_partition, self.tp_size = 576, 4


for name, attrs in (
    ("sglang.srt.layers.quantization.fp8", dict(Fp8Config=Fp8Config, Fp8LinearMethod=Fp8LinearMethod)),
    ("sglang.srt.layers.quantization.fp8_utils", dict(triton_w8a8_block_fp8_linear=triton_w8a8_block_fp8_linear)),
    ("sglang.srt.layers.linear", dict(MergedColumnParallelLinear=MergedColumnParallelLinear, RowParallelLinear=RowParallelLinear)),
    ("sglang.srt.utils", dict(is_hip=lambda: True)),
):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod

# The real draft marker.
enable_node, _ = method_of(dspark_tree, "DSparkV4Stage", "_enable_native_projections")
dspark_path = SITE / "sglang/srt/models/deepseek_v4_dspark.py"
ns_draft = dict(
    functools=functools, os=os, ReplicatedLinear=ReplicatedLinear, Optional=__import__("typing").Optional,
    get_parallel=lambda: types.SimpleNamespace(tp_size=4, attn_tp_size=4), DeepSeekV4Config=object, QuantizationConfig=object,
)
run(dspark_path, [*top(dspark_tree, "_NATIVE_DRAFT_OFF"), enable_node], ns_draft)
enable_draft = ns_draft["_enable_native_projections"]

# The real indexer marker.
mark_node, _ = method_of(sparse_tree, "DeepseekV41Indexer", "_mark_native_wq_b")
ns_idx = dict(functools=functools)
run(sparse_path, [mark_node], ns_idx)
mark_indexer = ns_idx["_mark_native_wq_b"]

# The real shadow classes and builder (installed fp8_utils.py), with the real helper functions of fp8_kernel.py.
kernel_path, kernel_tree = parse("sglang/kernels/ops/gemm/fp8_kernel.py")
ns_kernel = dict(torch=torch, triton=triton, tl=tl, os=os, functools=functools, List=list, _e4m3fn_to_fp32=_e4m3fn_to_fp32)
run(
    kernel_path,
    top(
        kernel_tree, "GFX1151_V41_DENSE_SHAPES", "dsv41_bf16_shadow_min_m", "dequant_block_fp8_weight_to_bf16",
        "block_fp8_scale_is_power_of_two", "_dequant_e4m3fn_group32_bf16", "dequant_group32_activation_to_bf16",
    ),
    ns_kernel,
)
min_m = ns_kernel["dsv41_bf16_shadow_min_m"]
SHAPES = ns_kernel["GFX1151_V41_DENSE_SHAPES"]


def fake_quantizer(x, group_size, scale_ue8m0=False):
    assert group_size == 32 and scale_ue8m0
    rows, cols = x.shape
    groups = x.to(torch.float32).view(rows, cols // 32, 32)
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-10) / 448.0)))
    return (groups / scale.unsqueeze(-1)).clamp(-448, 448).to(fp8).view(rows, cols), scale


utils_path, utils_tree = parse("sglang/srt/layers/quantization/fp8_utils.py")
ns_shadow = dict(
    torch=torch, logger=logging.getLogger("compose-test"), _is_hip=True,
    dsv41_bf16_shadow_min_m=min_m, GFX1151_V41_DENSE_SHAPES=SHAPES,
    block_fp8_scale_is_power_of_two=ns_kernel["block_fp8_scale_is_power_of_two"],
    dequant_block_fp8_weight_to_bf16=ns_kernel["dequant_block_fp8_weight_to_bf16"],
    dequant_group32_activation_to_bf16=ns_kernel["dequant_group32_activation_to_bf16"],
    sglang_per_token_group_quant_fp8=fake_quantizer,
    triton_w8a8_block_fp8_linear=triton_w8a8_block_fp8_linear,
    _DSV41_SHADOW_BUILT={"layers": 0, "bytes": 0, "shapes": set()},
)
run(utils_path, top(utils_tree, "dsv41_bf16_shadow_applies", "Dsv41Bf16ShadowLinear"), ns_shadow)
run(utils_path, top(utils_tree, "maybe_build_dsv41_bf16_shadow"), ns_shadow)
Wrapper = ns_shadow["Dsv41Bf16ShadowLinear"]
build = ns_shadow["maybe_build_dsv41_bf16_shadow"]


class FakeCudaWeight:
    is_cuda = True
    device = "cpu"

    def __init__(self, tensor):
        self.data = tensor
        self.dtype, self.shape = tensor.dtype, tensor.shape

    def dim(self):
        return self.data.dim()

    def data_ptr(self):
        return self.data.data_ptr()


def layer_of(n_dim, k_dim, seed=0):
    gen = torch.Generator().manual_seed(n_dim * 7 + k_dim + seed)
    raw = torch.randint(0, 256, (n_dim, k_dim), generator=gen, dtype=torch.int16).to(torch.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x30
    scale = 2.0 ** torch.randint(-12, -4, (n_dim // 32, k_dim // 32), generator=gen).to(torch.float32)
    return types.SimpleNamespace(
        weight=FakeCudaWeight(raw.view(fp8)),
        weight_scale_inv=types.SimpleNamespace(data=scale, dtype=scale.dtype, shape=scale.shape),
    )


# ---- 3. the draft stage: mark in the constructor, build after loading -----------------------------------------------
THRESHOLD = 16
props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc+:xnack-")
cfg = types.SimpleNamespace(
    model_type="deepseek_v41", hidden_size=5120, q_lora_rank=1280, head_dim=512,
    moe_intermediate_size=2304, n_shared_experts=1, swiglu_limit=10.0,
)
stage = types.SimpleNamespace(
    stage_id=0,
    self_attn=types.SimpleNamespace(wq_a=ReplicatedLinear(5120, 1280), wkv=ReplicatedLinear(5120, 512)),
    main_proj=ReplicatedLinear(15360, 5120),
    mlp=types.SimpleNamespace(
        tp_size=4, moe_ep_size=1, _shared_expert_tp1=False,
        shared_experts=types.SimpleNamespace(gate_up_proj=MergedColumnParallelLinear(), down_proj=RowParallelLinear()),
    ),
)
enable_draft(stage, cfg, Fp8Config())
plain = {"act_scale_ue8m0": True}
draft_linears = {  # name -> (linear, per-rank weight shape, expected flag)
    "wq_a": (stage.self_attn.wq_a, (1280, 5120), {"native_gemv_rows": True}),
    "wkv": (stage.self_attn.wkv, (512, 5120), {"native_gemv_rows": True}),
    "main_proj": (stage.main_proj, (5120, 15360), {"native_gemv_rows": True}),
    "gate_up": (stage.mlp.shared_experts.gate_up_proj, (1152, 5120), {"native_v41_shared": "gate_up"}),
    "down": (stage.mlp.shared_experts.down_proj, (5120, 576), {"native_v41_shared": "down"}),
}
marked_partials = {}
for name, (linear, shape, flag) in draft_linears.items():
    callable_ = linear.quant_method.w8a8_block_fp8_linear
    assert callable_.keywords == {**plain, **flag}, (name, callable_.keywords)
    marked_partials[name] = callable_

with mock.patch.object(torch.cuda, "get_device_properties", return_value=props), mock.patch.object(
    torch.cuda, "is_current_stream_capturing", return_value=False
):
    os.environ["SGLANG_DSV41_BF16_SHADOW_MIN_M"] = str(THRESHOLD)
    min_m.cache_clear()
    layers = {}
    for name, (linear, shape, flag) in draft_linears.items():
        layers[name] = layer_of(*shape)
        build(layers[name], linear.quant_method)
    for name, (linear, shape, flag) in draft_linears.items():
        callable_ = linear.quant_method.w8a8_block_fp8_linear
        if shape in SHAPES:
            # eligible: wrapped, the marked partial is the fallback with its flag intact
            assert isinstance(callable_, Wrapper), name
            assert callable_.fallback is marked_partials[name], name
            assert callable_.fallback.keywords == {**plain, **draft_linears[name][2]}
            assert callable_.shadow.shape == shape
        else:
            # the draft's wq_a / wkv / main_proj are not among the seven dense shapes: untouched
            assert callable_ is marked_partials[name], name
    assert {n for n, (_, s, _) in draft_linears.items() if s in SHAPES} == {"gate_up", "down"}

    # below the threshold the wrapper hands the caller's keyword arguments to the marked partial
    gate_up = draft_linears["gate_up"][0].quant_method.w8a8_block_fp8_linear
    weight = layers["gate_up"].weight.data
    scale = layers["gate_up"].weight_scale_inv.data
    for rows in (1, 3, 6, 8, THRESHOLD - 1):
        calls.clear()
        x = torch.randn(rows, 5120).to(torch.bfloat16)
        out = gate_up(input=x, weight=weight, block_size=[32, 32], weight_scale=scale, input_scale=None, bias=None)
        assert out == "fallback-output" and len(calls) == 1, rows
        got = calls[0]
        assert got["act_scale_ue8m0"] is True and got["native_v41_shared"] == "gate_up", got.keys()
        assert got["weight"] is weight and got["input"] is x and got["weight_scale"] is scale

    # at or above it the shadow GEMM runs (and the marked partial is not touched)
    for rows in (THRESHOLD, 100):
        calls.clear()
        x = torch.randn(rows, 5120).to(torch.bfloat16)
        out = gate_up(input=x, weight=weight, block_size=[32, 32], weight_scale=scale, input_scale=None, bias=None)
        assert not calls and out.shape == (rows, 1152) and out.dtype == torch.bfloat16, rows

    # the stacked weight of the commit path goes through a *different* callable and shape; through this one it must
    # still reach the marked partial untouched, whatever the row count (foreign shape, then foreign storage)
    stacked = torch.zeros(3 * 512, 5120).to(fp8)
    stacked_scale = torch.ones(3 * 512 // 32, 5120 // 32)
    calls.clear()
    x = torch.randn(4 * THRESHOLD, 5120).to(torch.bfloat16)
    out = gate_up(input=x, weight=stacked, block_size=[32, 32], weight_scale=stacked_scale, input_scale=None, bias=None)
    assert out == "fallback-output" and calls[0]["weight"] is stacked and calls[0]["native_v41_shared"] == "gate_up"
    clone = weight.clone()
    assert clone.shape == weight.shape and clone.data_ptr() != weight.data_ptr()
    calls.clear()
    gate_up(input=x, weight=clone, block_size=[32, 32], weight_scale=scale, input_scale=None, bias=None)
    assert calls and calls[0]["weight"] is clone, "same shape, other storage"

    # the draft's own stacked wkv: wkv itself is not shadow-eligible, so its callable is the plain marked partial
    wkv = draft_linears["wkv"][0].quant_method.w8a8_block_fp8_linear
    assert wkv is marked_partials["wkv"] and not isinstance(wkv, Wrapper)
    calls.clear()
    wkv(input=x, weight=stacked, block_size=[32, 32], weight_scale=stacked_scale, input_scale=None, bias=None)
    assert calls[0]["native_gemv_rows"] is True and calls[0]["weight"] is stacked

    # ---- 4. the indexer wq_b: marked in the constructor, shadow-eligible (4096, 1280) --------------------------------
    wq_b = ReplicatedLinear(1280, 4096)
    indexer = types.SimpleNamespace(wq_b=wq_b, n_heads=32, index_head_dim=128, _mark_native_wq_b=None)
    mark_indexer(indexer, types.SimpleNamespace(model_type="deepseek_v41"), Fp8Config())
    marked = wq_b.quant_method.w8a8_block_fp8_linear
    assert marked.keywords == {**plain, "native_idx_wqb": True}, marked.keywords
    layer = layer_of(4096, 1280)
    build(layer, wq_b.quant_method)
    wrapped = wq_b.quant_method.w8a8_block_fp8_linear
    assert isinstance(wrapped, Wrapper) and wrapped.fallback is marked
    calls.clear()
    x = torch.randn(4, 1280).to(torch.bfloat16)
    wrapped(input=x, weight=layer.weight.data, block_size=[32, 32], weight_scale=layer.weight_scale_inv.data,
            input_scale=None, bias=None)
    assert calls[0]["native_idx_wqb"] is True

    # ---- 5. the target's flags (set in constructors by the existing model code) keep composing ------------------------
    for flag, shape in (
        ({"native_wqkv_a_c1": True}, (1792, 5120)),
        ({"native_wqb_c1": True}, (8192, 1280)),
        ({"native_v41_shared": "gate_up"}, (1152, 5120)),
        ({"engram_wkv_c1": True}, (25600, 6144)),
    ):
        linear = ReplicatedLinear(shape[1], shape[0])
        linear.quant_method.w8a8_block_fp8_linear = functools.partial(
            triton_w8a8_block_fp8_linear, **plain, **flag
        )
        original = linear.quant_method.w8a8_block_fp8_linear
        layer = layer_of(*shape)
        build(layer, linear.quant_method)
        wrapped = linear.quant_method.w8a8_block_fp8_linear
        assert isinstance(wrapped, Wrapper) and wrapped.fallback is original and wrapped.fallback.keywords == {**plain, **flag}
        calls.clear()
        x = torch.randn(1, shape[1]).to(torch.bfloat16)
        wrapped(input=x, weight=layer.weight.data, block_size=[32, 32], weight_scale=layer.weight_scale_inv.data,
                input_scale=None, bias=None)
        assert all(calls[0][k] == v for k, v in flag.items())

    # reload (weights loaded twice): one wrapper deep, marked partial still the fallback
    linear = draft_linears["down"][0]
    before = linear.quant_method.w8a8_block_fp8_linear
    build(layers["down"], linear.quant_method)
    after = linear.quant_method.w8a8_block_fp8_linear
    assert isinstance(after, Wrapper) and after.fallback is marked_partials["down"] and after is not before

os.environ.pop("SGLANG_DSV41_BF16_SHADOW_MIN_M", None)
min_m.cache_clear()
print(
    "shadow/native compose OK: marks precede the shadow build; wrapper keeps the marked partial as fallback; "
    "draft wq_a/wkv/main_proj untouched; foreign weights reach the marked partial"
)
