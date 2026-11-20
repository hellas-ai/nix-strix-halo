"""CPU-only routed-scale ownership regression against an installed SGLang runtime.

Execute actual method bodies with GEMM, route, alignment, stream and collective
stubs; test the once-only scale contract without initializing a GPU.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PATHS = {
    "caller": "sglang/srt/models/deepseek_v2.py",
    "fp8": "sglang/srt/layers/quantization/fp8.py",
    "layer": "sglang/srt/layers/moe/fused_moe_triton/layer.py",
    "helper": "sglang/srt/layers/moe/moe_runner/triton_utils/mxfp4_moe_amd.py",
    "config": "sglang/srt/layers/moe/moe_runner/base.py",
    "finalize": "sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py",
}


def node(path, name, cls=None):
    body = ast.parse(path.read_text()).body
    if cls:
        body = next(
            x for x in body if isinstance(x, ast.ClassDef) and x.name == cls
        ).body
    return next(x for x in body if getattr(x, "name", None) == name)


def execute(nodes, namespace, filename):
    # Preserve actual method bodies, decorators and future annotation semantics.
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    # Exercise installed method bodies rather than duplicate their implementation.
    exec(compile(ast.fix_missing_locations(code), str(filename), "exec"), namespace)  # noqa: S102


def module(name, **entries):
    value = types.ModuleType(name)
    value.__dict__.update(entries)
    sys.modules[name] = value
    return value


class Stream:
    def wait_stream(self, other):
        pass


class TopK(tuple):
    format = "standard"


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class TorchCPUCaller:
    # Stream scheduling only is inert; real CPU Tensor arithmetic is unchanged.
    cuda = types.SimpleNamespace(
        current_stream=lambda: Stream(), stream=lambda _: contextlib.nullcontext()
    )

    def __getattr__(self, name):
        return getattr(torch, name)


class Backend:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        if name.startswith("is_"):
            return lambda: self.name == name[3:]
        raise AttributeError(name)


def run(source):
    paths = {k: source / v for k, v in PATHS.items()}
    ns = {
        "__name__": "integration_config",
        "torch": torch,
        "dataclass": dataclasses.dataclass,
    }
    module("integration_config")
    sys.modules["integration_config"].__dict__.update(ns)
    ns = sys.modules["integration_config"].__dict__
    execute([node(paths["config"], "MoeRunnerConfig")], ns, paths["config"])
    Config = ns["MoeRunnerConfig"]
    backend = Backend("triton")
    trace = []

    def gemm(a, weights, output, *args, **kwargs):
        trace.append(("gemm", tuple(output.shape)))
        output.zero_()
        if kwargs["mul_routed_weight"]:
            output[:, 0, :] = 2  # Known unscaled routed sum2; no GEMM numerical oracle.

    module("aiter.ops.triton.utils._triton.arch_info", get_arch=lambda: "gfx1151")
    module(
        "aiter.ops.triton.utils.moe_config_utils",
        get_optimal_moe_config=lambda *a, **k: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
        },
    )
    module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size",
        moe_align_block_size=lambda *a: (None, None, None),
    )
    helper_ns = {
        "torch": torch,
        "F": F,
        "_run_mxfp4_gemm": gemm,
        "_SPARSE_MOE_TILE": {
            "gfx1151": {
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "num_warps": 4,
                "waves_per_eu": 1,
            }
        },
    }
    execute(
        [
            node(paths["helper"], x)
            for x in ("sparse_moe_config", "gate_up_moe_config", "fused_experts_mxfp4")
        ],
        helper_ns,
        paths["helper"],
    )
    helper = helper_ns["fused_experts_mxfp4"]

    def recorded_helper(*args, **kwargs):
        trace.append(("helper_factor", kwargs["routed_scaling_factor"]))
        out = helper(*args, **kwargs)
        trace.append(("helper_output", float(out.flatten()[0])))
        return out

    module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd",
        fused_experts_mxfp4=recorded_helper,
    )
    module(
        "sglang.srt.layers.moe.token_dispatcher",
        StandardCombineInput=lambda **kw: types.SimpleNamespace(**kw),
    )
    quant_ns = {
        "torch": torch,
        "_is_hip": True,
        "_is_npu": False,
        "_use_aiter": False,
        "_use_hip_int4": False,
        "_mxfp8_to_block_fp8_required": False,
        "get_moe_runner_backend": lambda: backend,
        "MoeRunner": lambda b, c: types.SimpleNamespace(runner_backend=b, config=c),
    }
    methods = [
        node(paths["fp8"], n, "Fp8MoEMethod")
        for n in ("__init__", "create_moe_runner", "apply")
    ]
    fp8_cls = next(
        x
        for x in ast.parse(paths["fp8"].read_text()).body
        if isinstance(x, ast.ClassDef) and x.name == "Fp8MoEMethod"
    )
    # Include new capability verbatim when provided by the reviewed patch.
    methods += [
        x
        for x in fp8_cls.body
        if getattr(x, "name", None) == "applies_routed_scaling_factor_on_output"
    ]
    execute(
        [
            ast.ClassDef(
                name="Fp8MoEMethod",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        quant_ns,
        paths["fp8"],
    )
    Fp8 = quant_ns["Fp8MoEMethod"]
    other_type = type("OtherQuantMethod", (), {})
    prop_ns = {
        "Fp8MoEMethod": Fp8,
        "ModelOptNvFp4FusedMoEMethod": type("ModelOptNvFp4FusedMoEMethod", (), {}),
        "UnquantizedFusedMoEMethod": type("UnquantizedFusedMoEMethod", (), {}),
        "get_moe_runner_backend": lambda: backend,
    }
    execute(
        [node(paths["layer"], "_fuses_routed_scaling_factor_in_topk")],
        prop_ns,
        paths["layer"],
    )
    for modname, cls in [
        ("expert_pack", "ExpertPackMoEMethod"),
        ("mxfp4_flashinfer_cutlass_moe", "Mxfp4FlashinferCutlassMoEMethod"),
        ("mxfp4_marlin_moe", "Mxfp4MarlinMoEMethod"),
    ]:
        module("sglang.srt.layers.quantization." + modname, **{cls: type(cls, (), {})})
    finalize_ns = {
        "torch": torch,
        "Mxfp4FlashinferTrtllmMoEMethod": type(
            "Mxfp4FlashinferTrtllmMoEMethod", (), {}
        ),
    }
    execute(
        [node(paths["finalize"], "maybe_fuse_routed_scale_and_shared_add")],
        finalize_ns,
        paths["finalize"],
    )
    caller_ns = {
        "torch": TorchCPUCaller(),
        "_is_cuda": False,
        "_is_musa": False,
        "_is_xpu": False,
        "_use_aiter": False,
        "use_intel_amx_backend": lambda *a: False,
        "get_exec": lambda: types.SimpleNamespace(
            moe=types.SimpleNamespace(enable_eplb=False)
        ),
        "KTEPWrapperMethod": type("KTEPWrapperMethod", (), {}),
        "maybe_fuse_routed_scale_and_shared_add": finalize_ns[
            "maybe_fuse_routed_scale_and_shared_add"
        ],
        "should_skip_post_experts_all_reduce": lambda **k: False,
        "post_experts_all_reduce": lambda x: x,
        "should_add_replicated_moe_output": lambda: True,
        "get_forward": lambda: types.SimpleNamespace(flashinfer_trtllm_bypass=False),
        "TopKOutputFormat": types.SimpleNamespace(BYPASSED="bypassed"),
        "get_moe_a2a_backend": lambda: Backend("none"),
        "ExpertLocationDispatchInfo": types.SimpleNamespace(init_new=lambda **kw: None),
        "envs": types.SimpleNamespace(
            SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO=types.SimpleNamespace(
                get=lambda: False
            )
        ),
    }
    branches = (
        "forward_normal",
        "forward_normal_dual_stream",
        "forward_deepep",
        "op_output",
    )
    execute(
        [node(paths["caller"], n, "DeepseekV2MoE") for n in branches],
        caller_ns,
        paths["caller"],
    )
    caller_class = next(
        x
        for x in ast.parse(paths["caller"].read_text()).body
        if isinstance(x, ast.ClassDef) and x.name == "DeepseekV2MoE"
    )
    optional = [
        x
        for x in caller_class.body
        if getattr(x, "name", None) == "_experts_apply_routed_scaling_factor"
    ]
    if optional:
        execute(optional, caller_ns, paths["caller"])
    quant_config = types.SimpleNamespace(
        use_mxfp8=False,
        weight_block_size=[128, 128],
        is_fp4_experts=True,
        dequant_fp4_to_fp8=False,
    )
    # Execute the actual V4 TopK configuration override, not a copied boolean.
    constructor = node(paths["caller"], "__init__", "DeepseekV2MoE")
    override = [
        n
        for n in ast.walk(constructor)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "is_deepseek_v4"
        and len(n.body) == 1
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Call)
        and isinstance(n.body[0].value.func, ast.Attribute)
        and isinstance(n.body[0].value.func.value, ast.Name)
        and n.body[0].value.func.value.id == "topk_kwargs"
    ]
    assert len(override) == 1
    config_ns = {
        "is_deepseek_v4": True,
        "topk_kwargs": {},
        "_use_aiter": False,
        "config": types.SimpleNamespace(
            model_type="deepseek_v41", scoring_func="sqrtsoftplus"
        ),
        "quant_config": types.SimpleNamespace(is_fp4_experts=True),
        "self": types.SimpleNamespace(
            experts=types.SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=False
            )
        ),
    }
    execute(override, config_ns, paths["caller"])
    assert config_ns["topk_kwargs"]["apply_routed_scaling_factor_on_output"] is False
    capability_controls = []
    if optional:
        for hip, fp4, runner_name in (
            (True, True, "triton"),
            (False, True, "triton"),
            (True, False, "triton"),
            (True, True, "aiter"),
        ):
            quant_ns["_is_hip"] = hip
            probe = Fp8(quant_config)
            probe.is_fp4_expert = fp4
            probe.runner = types.SimpleNamespace(runner_backend=Backend(runner_name))
            value = probe.applies_routed_scaling_factor_on_output
            assert value == (hip and fp4 and runner_name == "triton")
            capability_controls.append(
                {"hip": hip, "fp4": fp4, "runner": runner_name, "capability": value}
            )
        quant_ns["_is_hip"] = True
    records = []
    for count in (1, 2, 7):
        for shared in (False, True):
            for portable in (True, False):
                trace.clear()
                method = Fp8(quant_config)
                cfg = Config(
                    num_experts=384,
                    num_local_experts=384,
                    hidden_size=5120,
                    intermediate_size_per_partition=576,
                    top_k=6,
                    routed_scaling_factor=1.5,
                    inplace=False,
                    swiglu_limit=10,
                )

                class Experts:
                    def __call__(
                        self,
                        hidden_states,
                        topk_output,
                        _portable=portable,
                        _method=method,
                        **kw,
                    ):
                        if not _portable:
                            return torch.full_like(hidden_states, 2)
                        return _method.apply(
                            self,
                            types.SimpleNamespace(
                                hidden_states=hidden_states, topk_output=topk_output
                            ),
                        ).hidden_states

                experts = Experts()
                experts.moe_runner_config = cfg
                experts.quant_method = method if portable else other_type()
                experts.should_fuse_routed_scaling_factor_in_topk = prop_ns[
                    "_fuses_routed_scaling_factor_in_topk"
                ](experts.quant_method)
                assert not experts.should_fuse_routed_scaling_factor_in_topk
                # Native shape metadata, no checkpoint weight allocation or reads.
                experts.w13_weight = torch.empty(
                    (384, 1152, 2560), device="meta", dtype=torch.uint8
                )
                experts.w2_weight = torch.empty(
                    (384, 5120, 288), device="meta", dtype=torch.uint8
                )
                experts.w13_weight_scale_inv = torch.empty(
                    (384, 1152, 160), device="meta", dtype=torch.uint8
                )
                experts.w2_weight_scale_inv = torch.empty(
                    (384, 5120, 18), device="meta", dtype=torch.uint8
                )
                method.create_moe_runner(experts, cfg)
                assert (
                    method.moe_runner_config is cfg
                    and method.runner.runner_backend.is_triton()
                )
                hidden = torch.zeros((count, 5120), dtype=torch.bfloat16)
                fake = types.SimpleNamespace(
                    experts=experts,
                    layer_id=24,
                    is_nextn=False,
                    is_hash=False,
                    _fuse_shared_experts_inside_sbo=False,
                    _shared_expert_tp1=False,
                    is_deepseek_v4=True,
                    tp_size=4,
                    routed_scaling_factor=1.5,
                    _maybe_quant_moe_input_once=lambda x: None,
                    _forward_shared_experts=lambda *a, _hidden=hidden, _shared=shared, **kw: (
                        torch.full_like(_hidden, 5) if _shared else None
                    ),
                    _forward_gate=lambda *a, **kw: (None, None),
                    topk=lambda *a, _count=count, **kw: TopK(
                        (
                            torch.full((_count, 6), 1 / 6, dtype=torch.float32),
                            torch.arange(6, dtype=torch.int32).expand(_count, 6),
                            None,
                        )
                    ),
                    _should_quant_routed_input_mxfp8=lambda _: False,
                    _fuse_finalize_all_reduce=False,
                    num_fused_shared_experts=0,
                    gate=lambda *a, **kw: None,
                    _all_reduce_output=lambda x: x,
                )
                if optional:
                    fake._experts_apply_routed_scaling_factor = types.MethodType(
                        caller_ns["_experts_apply_routed_scaling_factor"], fake
                    )
                for branch in branches:
                    trace.clear()
                    fake.alt_stream = (
                        Stream() if branch == "forward_normal_dual_stream" else None
                    )
                    if branch == "forward_deepep":
                        result = caller_ns[branch](
                            fake,
                            hidden,
                            types.SimpleNamespace(
                                moe_num_token_non_padded=lambda: None
                            ),
                        )
                    elif branch == "op_output":
                        state = State(
                            hidden_states_after_combine=experts(hidden, fake.topk()),
                            shared_output=torch.full_like(hidden, 5)
                            if shared
                            else None,
                        )
                        caller_ns[branch](fake, state)
                        result = state.hidden_states_mlp_output
                    else:
                        result = caller_ns[branch](fake, hidden)
                    correct = 3 + (5 if shared else 0)
                    assert torch.equal(result, torch.full_like(result, correct)), (
                        branch,
                        count,
                        shared,
                        portable,
                        float(result[0, 0]),
                        correct,
                    )
                    if portable:
                        assert ("helper_factor", 1.5) in trace and (
                            "helper_output",
                            3.0,
                        ) in trace
                    records.append(
                        {
                            "branch": branch,
                            "rows": count,
                            "shared": shared,
                            "portable": portable,
                            "observed": float(result[0, 0]),
                            "independent_expected": correct,
                            "scale_exactly_once": float(result[0, 0]) == correct,
                            "trace": list(trace),
                        }
                    )
    assert not torch.cuda.is_initialized()
    return {
        "event": "complete",
        "passed": True,
        "cases": len(records),
        "legacy_cases": sum(not r["portable"] for r in records),
        "capability_controls": len(capability_controls),
        "cuda_initialized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    candidates = list(args.runtime.glob("lib/python*/site-packages"))
    assert len(candidates) == 1, "Expected an installed SGLang runtime"
    print(json.dumps(run(candidates[0])))


if __name__ == "__main__":
    main()
