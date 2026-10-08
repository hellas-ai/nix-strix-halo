"""Actual routed-policy caller/constructor contracts with CPU tensors and explicit device stubs."""

from __future__ import annotations

import ast
import contextlib
import json
import sys
import types
from pathlib import Path

import torch

base = Path(sys.argv[1]) / "lib/python3.13/site-packages"
source = base


def load(names, path, ns, cls=None):
    body = ast.parse(path.read_text()).body
    if cls:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)  # noqa: S102 -- actual installed methods with explicit CPU stubs
    return ns


def module(name, **kw):
    value = types.ModuleType(name)
    value.__dict__.update(kw)
    sys.modules[name] = value


def bits(a):
    return a.contiguous().view(torch.uint8)


class Stream:
    def wait_stream(self, other):
        pass


class TorchCPU:
    cuda = types.SimpleNamespace(
        current_stream=lambda: Stream(), stream=lambda s: contextlib.nullcontext()
    )

    def __getattr__(self, n):
        return getattr(torch, n)


class TopK(tuple):
    format = "standard"


for mod, cls in [
    ("expert_pack", "ExpertPackMoEMethod"),
    ("mxfp4_flashinfer_cutlass_moe", "Mxfp4FlashinferCutlassMoEMethod"),
    ("mxfp4_marlin_moe", "Mxfp4MarlinMoEMethod"),
]:
    module("sglang.srt.layers.quantization." + mod, **{cls: type(cls, (), {})})
finalize = load(
    ["maybe_fuse_routed_scale_and_shared_add"],
    base / "sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py",
    {"torch": torch, "Mxfp4FlashinferTrtllmMoEMethod": type("Flashinfer", (), {})},
)["maybe_fuse_routed_scale_and_shared_add"]
ns = {
    "nullcontext": contextlib.nullcontext,
    "torch": TorchCPU(),
    "_is_cuda": False,
    "_is_musa": False,
    "_is_xpu": False,
    "_use_aiter": False,
    "use_intel_amx_backend": lambda *a: False,
    "get_exec": lambda: types.SimpleNamespace(
        moe=types.SimpleNamespace(enable_eplb=False)
    ),
    "KTEPWrapperMethod": type("KT", (), {}),
    "maybe_fuse_routed_scale_and_shared_add": finalize,
    "should_skip_post_experts_all_reduce": lambda **k: False,
    "post_experts_all_reduce": lambda x: x,
    "should_add_replicated_moe_output": lambda: True,
    "get_forward": lambda: types.SimpleNamespace(flashinfer_trtllm_bypass=False),
    "TopKOutputFormat": types.SimpleNamespace(BYPASSED="bypassed"),
}
load(
    [
        "forward_normal",
        "forward_normal_dual_stream",
        "_experts_apply_routed_scaling_factor",
    ],
    source / "sglang/srt/models/deepseek_v2.py",
    ns,
    "DeepseekV2MoE",
)
records = []
for branch in ("forward_normal", "forward_normal_dual_stream"):
    for replicated in (False, True):
        trace = []
        hidden = torch.zeros((1, 8), dtype=torch.bfloat16)

        class Experts:
            _dsv41_official_routed_policy = True
            moe_runner_config = types.SimpleNamespace(inplace=False)
            quant_method = types.SimpleNamespace(
                applies_routed_scaling_factor_on_output=True
            )

            def __call__(self, *a, trace=trace, hidden=hidden, **kw):
                trace.append("routed")
                return torch.full_like(hidden, 257, dtype=torch.float32)

        def shared(*a, trace=trace, hidden=hidden, **kw):
            trace.append("shared")
            return torch.ones_like(hidden)

        def reduce(value, replicated=replicated, trace=trace):
            assert value.dtype == torch.float32
            assert torch.all(value == (257 if replicated else 258))
            trace.append("tp_fp32")
            return value * 2

        fake = types.SimpleNamespace(
            experts=Experts(),
            layer_id=24,
            is_nextn=False,
            is_hash=False,
            _fuse_shared_experts_inside_sbo=False,
            _shared_expert_tp1=replicated,
            is_deepseek_v4=True,
            tp_size=4,
            routed_scaling_factor=1.5,
            _maybe_quant_moe_input_once=lambda x: None,
            _forward_shared_experts=shared,
            _forward_gate=lambda *a, **kw: (None, None),
            topk=lambda *a, **kw: TopK((None, None, None)),
            _should_quant_routed_input_mxfp8=lambda x: False,
            _fuse_finalize_all_reduce=False,
            num_fused_shared_experts=0,
            _all_reduce_output=reduce,
            alt_stream=Stream() if branch.endswith("dual_stream") else None,
        )
        fake._experts_apply_routed_scaling_factor = types.MethodType(
            ns["_experts_apply_routed_scaling_factor"], fake
        )
        actual = ns[branch](fake, hidden)
        expected = torch.full_like(
            hidden, (257 * 2 + 1) if replicated else (257 + 1) * 2, dtype=torch.float32
        ).bfloat16()
        assert actual.dtype == torch.bfloat16 and torch.equal(
            bits(actual), bits(expected)
        )
        records.append(
            {
                "branch": branch,
                "replicated_shared": replicated,
                "trace": list(trace),
                "final_value": float(actual[0, 0]),
                "final_dtype": str(actual.dtype),
            }
        )
# Actual StandardDispatcher combine returns the same FP32 tensor under DP1.
combine = load(
    ["combine"],
    base / "sglang/srt/layers/moe/token_dispatcher/standard.py",
    {"torch": torch, "should_use_flashinfer_cutlass_moe_fp4_allgather": lambda: False},
    "StandardDispatcher",
)["combine"]
probe = torch.tensor([[257]], dtype=torch.float32)
assert combine(None, (probe,)) is probe
# Execute the exact constructor opt-in statement with explicit metadata mocks.
model = ast.parse((source / "sglang/srt/models/deepseek_v4.py").read_text())
constructor = next(
    n
    for c in model.body
    if isinstance(c, ast.ClassDef) and c.name == "DeepseekV4DecoderLayer"
    for n in c.body
    if getattr(n, "name", None) == "__init__"
)
guards = [
    n
    for n in constructor.body
    if isinstance(n, ast.If)
    and any(
        isinstance(x, ast.Attribute) and x.attr == "_dsv41_official_routed_policy"
        for x in ast.walk(n)
    )
]
assert len(guards) == 1
controls = []
for change in (
    {},
    {"model": "deepseek_v41_text"},
    {"arch": "gfx950"},
    {"tp": 2},
    {"dp": 2},
    {"cp": 2},
    {"ep": 2},
    {"aiter": True},
    {"sbo": True},
    {"capability": False},
    {"normalized": False},
    {"hidden": 4096},
    {"prescaled": True},
    {"a2a": True},
    {"fused": 1},
):
    cfg = {
        "model": "deepseek_v41",
        "arch": "gfx1151",
        "tp": 4,
        "dp": 1,
        "cp": 1,
        "ep": 1,
        "aiter": False,
        "sbo": False,
        "capability": True,
        "normalized": True,
        "hidden": 5120,
        "prescaled": False,
        "a2a": False,
        "fused": 0,
    }
    cfg.update(change)
    experts = types.SimpleNamespace(
        moe_runner_config=types.SimpleNamespace(
            inplace=True, apply_router_weight_on_input=False, no_combine=False
        ),
        should_fuse_routed_scaling_factor_in_topk=cfg["prescaled"],
        quant_method=types.SimpleNamespace(
            applies_routed_scaling_factor_on_output=cfg["capability"]
        ),
    )
    mlp = types.SimpleNamespace(
        experts=experts,
        moe_ep_size=cfg["ep"],
        _enable_a2a_moe=cfg["a2a"],
        num_fused_shared_experts=cfg["fused"],
    )
    parallel = types.SimpleNamespace(
        tp_size=cfg["tp"],
        attn_tp_size=cfg["tp"],
        dp_size=cfg["dp"],
        attn_dp_size=cfg["dp"],
        moe_dp_size=cfg["dp"],
        attn_cp_size=cfg["cp"],
        attn_dcp_size=1,
    )
    meta = {
        "self": types.SimpleNamespace(mlp=mlp, layer_id=0),
        "logger": types.SimpleNamespace(info=lambda *a: None),
        "config": types.SimpleNamespace(
            model_type=cfg["model"],
            hidden_size=cfg["hidden"],
            moe_intermediate_size=2304,
            n_routed_experts=384,
            num_experts_per_tok=6,
            norm_topk_prob=cfg["normalized"],
        ),
        "_is_hip": True,
        "_use_aiter": cfg["aiter"],
        "get_parallel": lambda parallel=parallel: parallel,
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                current_device=lambda: 0,
                get_device_properties=lambda d, cfg=cfg: types.SimpleNamespace(
                    gcnArchName=cfg["arch"]
                ),
            )
        ),
        "deepseek_v2": types.SimpleNamespace(
            is_sbo_enabled=lambda cfg=cfg: cfg["sbo"], is_tbo_enabled=lambda: False
        ),
    }
    exec(  # noqa: S102 -- exact installed constructor opt-in with metadata stubs
        compile(
            ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])),
            "constructor-metadata",
            "exec",
        ),
        meta,
    )
    selected = getattr(experts, "_dsv41_official_routed_policy", False)
    assert selected == (not change)
    assert experts.moe_runner_config.inplace == (not selected)
    controls.append({"change": change, "selected": selected})
# Deferred TP ownership must be rejected instead of casting before a later sum.
ns["should_skip_post_experts_all_reduce"] = lambda **kw: True
try:
    ns["forward_normal"](fake, hidden)
except AssertionError:
    deferred_reduction_rejected = True
else:
    raise AssertionError("candidate accepted an owed deferred TP reduction")
ns["should_skip_post_experts_all_reduce"] = lambda **kw: False
# Execute the actual Fp8.apply body to bind marker and once-only scale arguments.
seen = []


def helper_spy(*args, **kwargs):
    seen.append(kwargs)
    return args[0].float()


module(
    "sglang.srt.layers.moe.token_dispatcher",
    StandardCombineInput=lambda **kw: types.SimpleNamespace(**kw),
)
module(
    "sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd",
    fused_experts_mxfp4=helper_spy,
)
apply = load(
    ["apply"],
    source / "sglang/srt/layers/quantization/fp8.py",
    {"torch": torch},
    "Fp8MoEMethod",
)["apply"]
apply_cfg = types.SimpleNamespace(
    activation="silu",
    is_gated=True,
    inplace=False,
    no_combine=False,
    apply_router_weight_on_input=False,
    routed_scaling_factor=1.5,
    swiglu_limit=10,
)
method = types.SimpleNamespace(
    applies_routed_scaling_factor_on_output=True, moe_runner_config=apply_cfg
)
layer = types.SimpleNamespace(
    w13_weight=None, w2_weight=None, w13_weight_scale_inv=None, w2_weight_scale_inv=None
)
for marked, prescaled in ((True, False), (True, True), (False, False)):
    layer._dsv41_official_routed_policy = marked
    layer.should_fuse_routed_scaling_factor_in_topk = prescaled
    output = apply(
        method,
        layer,
        types.SimpleNamespace(
            hidden_states=hidden,
            topk_output=(torch.ones((1, 6)), torch.arange(6).reshape(1, 6), None),
        ),
    )
    assert seen[-1]["official_routed_policy"] == marked
    assert seen[-1]["routed_scaling_factor"] == (None if prescaled else 1.5)
    assert output.hidden_states.dtype == torch.float32
result = {
    "passed": True,
    "caller_records": records,
    "standard_dispatcher_fp32_identity": True,
    "constructor_metadata_controls": controls,
    "architecture_queries_mocked_for_metadata_only": True,
    "actual_fp8_apply_controls": len(seen),
    "deferred_reduction_rejected": deferred_reduction_rejected,
    "cuda_initialized": torch.cuda.is_initialized(),
}
assert not result["cuda_initialized"]
# Actual M32 dispatch guard. Native GPU numerical coverage is separate.
moe_path = base / "sglang/srt/layers/moe/moe_runner/triton_utils/mxfp4_moe_amd.py"
tree = ast.parse(moe_path.read_text())
tile = next(
    n.value
    for n in tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "_SPARSE_MOE_TILE" for t in n.targets)
)
config_fn = load(
    ["sparse_moe_config"], moe_path, {"_SPARSE_MOE_TILE": ast.literal_eval(tile)}
)["sparse_moe_config"]
config = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}
shape = (6, 384, 5120, 576)
for rows, expected_m in [
    (512, 16),
    (1023, 16),
    (1024, 32),
    (1306, 32),
    (1536, 32),
    (2047, 32),
    (2048, 256),
]:
    actual = config_fn(config, "gfx1151", rows, *shape)
    assert actual == {**config, "BLOCK_SIZE_M": expected_m}
for arch in ("gfx950", "gfx1100"):
    assert config_fn(config, arch, 1024, *shape) == config
for dimensions in ((6, 384, 5120, 768), (8, 384, 5120, 576), (6, 192, 5120, 576)):
    assert config_fn(config, "gfx1151", 1024, *dimensions) == config
for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
    changed = {**config, key: 128}
    assert config_fn(changed, "gfx1151", 1024, *shape) == changed
result["prefill_m32_dispatch_controls"] = 15
print(json.dumps(result), flush=True)
