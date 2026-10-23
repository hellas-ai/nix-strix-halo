"""DSpark draft stage: which block-FP8 projections are marked for the native row-generic GEMV (CPU, import stubs, no GPU).

Runs the installed `DSparkV4Stage._enable_native_projections` against stand-in modules and linears and checks that exactly
wq_a, wkv, (stage 0) main_proj get `native_gemv_rows=True` and the draft shared expert gets `native_v41_shared`, under the
same guards the target model uses for its own flags, and that every other case is left on the previous path."""

import ast
import functools
import sys
import types
from pathlib import Path

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
SRC = SITE / "sglang/srt/models/deepseek_v4_dspark.py"
tree = ast.parse(SRC.read_text(), filename=str(SRC))
stage_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkV4Stage")
method = next(n for n in stage_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_enable_native_projections")
env_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_native_draft_projections_enabled")
off_values = next(
    n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_NATIVE_DRAFT_OFF"
)


class Fp8Config:
    def __init__(self, **kw):
        self.is_checkpoint_fp8_serialized = True
        self.activation_scheme = "dynamic"
        self.weight_block_size = [32, 32]
        self.scale_fmt = "ue8m0"
        self.__dict__.update(kw)


class Fp8LinearMethod:
    def __init__(self, keywords=None, use_mxfp8=False):
        self.use_mxfp8 = use_mxfp8
        self.block_fp8_as_mxfp8 = False
        self.w8a8_block_fp8_linear = functools.partial(
            triton_w8a8_block_fp8_linear, **({"act_scale_ue8m0": True} if keywords is None else keywords)
        )


def triton_w8a8_block_fp8_linear(*a, **k):
    raise AssertionError("never executed")


class ReplicatedLinear:
    def __init__(self, in_size, out_size, method=None, bias=None):
        self.input_size, self.output_size, self.bias = in_size, out_size, bias
        self.quant_method = method if method is not None else Fp8LinearMethod()


class MergedColumnParallelLinear(ReplicatedLinear):
    def __init__(self, method=None):
        super().__init__(5120, 2304 * 2, method)
        self.output_sizes, self.output_size_per_partition, self.tp_size = [2304, 2304], 1152, 4


class RowParallelLinear(ReplicatedLinear):
    def __init__(self, method=None):
        super().__init__(2304, 5120, method)
        self.input_size_per_partition, self.tp_size = 576, 4


state = dict(hip=True, tp=4, attn_tp=4)
for name, attrs in (
    ("sglang.srt.layers.quantization.fp8", dict(Fp8Config=Fp8Config, Fp8LinearMethod=Fp8LinearMethod)),
    ("sglang.srt.layers.quantization.fp8_utils", dict(triton_w8a8_block_fp8_linear=triton_w8a8_block_fp8_linear)),
    ("sglang.srt.layers.linear", dict(MergedColumnParallelLinear=MergedColumnParallelLinear, RowParallelLinear=RowParallelLinear)),
    ("sglang.srt.utils", dict(is_hip=lambda: state["hip"])),
):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod

namespace = dict(
    functools=functools,
    os=__import__("os"),
    ReplicatedLinear=ReplicatedLinear,
    Optional=__import__("typing").Optional,
    get_parallel=lambda: types.SimpleNamespace(tp_size=state["tp"], attn_tp_size=state["attn_tp"]),
    DeepSeekV4Config=object,
    QuantizationConfig=object,
)
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
code = ast.fix_missing_locations(ast.Module(body=[future, off_values, method, env_fn], type_ignores=[]))
exec(compile(code, str(SRC), "exec"), namespace)  # noqa: S102 -- installed method with stub modules
enable = namespace["_enable_native_projections"]
env_enabled = namespace["_native_draft_projections_enabled"]


def config(**kw):
    c = types.SimpleNamespace(
        model_type="deepseek_v41", hidden_size=5120, q_lora_rank=1280, head_dim=512,
        moe_intermediate_size=2304, n_shared_experts=1, swiglu_limit=10.0,
    )
    c.__dict__.update(kw)
    return c


def stage(stage_id=0, **moe_kw):
    shared = types.SimpleNamespace(gate_up_proj=MergedColumnParallelLinear(), down_proj=RowParallelLinear())
    moe = types.SimpleNamespace(tp_size=4, moe_ep_size=1, _shared_expert_tp1=False, shared_experts=shared)
    moe.__dict__.update(moe_kw)
    return types.SimpleNamespace(
        stage_id=stage_id,
        self_attn=types.SimpleNamespace(wq_a=ReplicatedLinear(5120, 1280), wkv=ReplicatedLinear(5120, 512)),
        main_proj=ReplicatedLinear(15360, 5120) if stage_id == 0 else None,
        mlp=moe,
    )


def keywords(linear):
    return linear.quant_method.w8a8_block_fp8_linear.keywords


def marked(st):
    return {
        "wq_a": keywords(st.self_attn.wq_a), "wkv": keywords(st.self_attn.wkv),
        "main_proj": None if st.main_proj is None else keywords(st.main_proj),
        "gate_up": keywords(st.mlp.shared_experts.gate_up_proj), "down": keywords(st.mlp.shared_experts.down_proj),
    }


plain = {"act_scale_ue8m0": True}
# stage 0: everything marked
st = stage(0)
enable(st, config(), Fp8Config())
assert marked(st) == {
    "wq_a": {**plain, "native_gemv_rows": True}, "wkv": {**plain, "native_gemv_rows": True},
    "main_proj": {**plain, "native_gemv_rows": True},
    "gate_up": {**plain, "native_v41_shared": "gate_up"}, "down": {**plain, "native_v41_shared": "down"},
}, marked(st)
# later stages own no main_proj
st = stage(1)
enable(st, config(), Fp8Config())
assert marked(st)["main_proj"] is None and marked(st)["wq_a"] == {**plain, "native_gemv_rows": True}


def untouched(st):
    return all(v in (None, plain) for v in marked(st).values())


# guards: every one of them leaves the stage on the generic path
cases = [
    ("not hip", dict(hip=False), config(), Fp8Config()),
    ("tp 2", dict(tp=2), config(), Fp8Config()),
    ("attn tp 2", dict(attn_tp=2), config(), Fp8Config()),
    ("other model type", {}, config(model_type="deepseek_v4"), Fp8Config()),
    ("no fp8 config", {}, config(), None),
    ("static activation", {}, config(), Fp8Config(activation_scheme="static")),
    ("block 128", {}, config(), Fp8Config(weight_block_size=[128, 128])),
    ("not ue8m0", {}, config(), Fp8Config(scale_fmt=None)),
    ("not serialized fp8", {}, config(), Fp8Config(is_checkpoint_fp8_serialized=False)),
    ("other hidden", {}, config(hidden_size=4096), Fp8Config()),
]
for label, state_kw, cfg, qc in cases:
    state.update(hip=True, tp=4, attn_tp=4)
    state.update(state_kw)
    st = stage()
    enable(st, cfg, qc)
    assert untouched(st), (label, marked(st))
state.update(hip=True, tp=4, attn_tp=4)

# per-linear guards
st = stage()
st.self_attn.wq_a = ReplicatedLinear(5120, 1536)       # not the draft's q_lora_rank
enable(st, config(), Fp8Config())
assert keywords(st.self_attn.wq_a) == plain and keywords(st.self_attn.wkv) == {**plain, "native_gemv_rows": True}
st = stage()
st.self_attn.wkv = ReplicatedLinear(5120, 512, method=Fp8LinearMethod({"act_scale_ue8m0": True, "native_wqb_c1": True}))  # already flagged
enable(st, config(), Fp8Config())
assert keywords(st.self_attn.wkv) == {"act_scale_ue8m0": True, "native_wqb_c1": True}
st = stage()
st.self_attn.wq_a = ReplicatedLinear(5120, 1280, bias=object())
enable(st, config(), Fp8Config())
assert keywords(st.self_attn.wq_a) == plain
st = stage()
st.main_proj = ReplicatedLinear(15360, 5120, method=Fp8LinearMethod(use_mxfp8=True))
enable(st, config(), Fp8Config())
assert keywords(st.main_proj) == plain
st = stage()
st.self_attn.wq_a = types.SimpleNamespace(input_size=5120, output_size=1280, bias=None, quant_method=Fp8LinearMethod())  # not ReplicatedLinear
enable(st, config(), Fp8Config())
assert keywords(st.self_attn.wq_a) == plain
# shared expert guards
for moe_kw in (dict(tp_size=2), dict(moe_ep_size=2), dict(_shared_expert_tp1=True)):
    st = stage(**moe_kw)
    enable(st, config(), Fp8Config())
    assert keywords(st.mlp.shared_experts.gate_up_proj) == plain and keywords(st.mlp.shared_experts.down_proj) == plain, moe_kw
    assert keywords(st.self_attn.wq_a) == {**plain, "native_gemv_rows": True}
for cfg in (config(moe_intermediate_size=1536), config(n_shared_experts=2), config(swiglu_limit=7.0)):
    st = stage()
    enable(st, cfg, Fp8Config())
    assert keywords(st.mlp.shared_experts.gate_up_proj) == plain
st = stage()
st.mlp.shared_experts = None
enable(st, config(), Fp8Config())          # no shared expert module: nothing to mark, no failure
assert keywords(st.self_attn.wq_a) == {**plain, "native_gemv_rows": True}

# switch
import os

for value in ("0", "false", "OFF", "no", " 0 "):
    os.environ["SGLANG_DSV41_NATIVE_DRAFT_PROJECTIONS"] = value
    assert not env_enabled(), value
for value in ("1", "true", "", "yes"):
    os.environ["SGLANG_DSV41_NATIVE_DRAFT_PROJECTIONS"] = value
    assert env_enabled(), value
os.environ.pop("SGLANG_DSV41_NATIVE_DRAFT_PROJECTIONS")
assert env_enabled()

# the call site: the stage constructor calls it last, behind the switch
init = next(n for n in stage_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
calls = [n for n in ast.walk(init) if isinstance(n, ast.If) and "_native_draft_projections_enabled()" in ast.unparse(n.test)]
assert len(calls) == 1 and "self._enable_native_projections(config, quant_config)" in ast.unparse(calls[0])
assert init.body[-1] is calls[0]
print("draft projection flags ok")
