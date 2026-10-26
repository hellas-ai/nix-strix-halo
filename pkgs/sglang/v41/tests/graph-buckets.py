"""Installed graph geometry regression with explicit CPU metadata stubs.

No device/model is initialized. This checks capture selection and early replay
eligibility, not GPU graph execution or collective correctness.
"""

import ast
import enum
import importlib.util
import sys
import types
from pathlib import Path

RUNTIME = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages/sglang/srt"


def stub(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    sys.modules[name] = result
    return result


def source_function(path, name, globals_):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_)  # noqa: S102 -- exercise installed method with metadata stubs
    return globals_[name]


class ForwardMode:
    DECODE = object()
    TARGET_VERIFY = object()


class RaggedVerifyMode(enum.Enum):
    STATIC = "static"
    COMPACT = "compact"


ragged_mode = RaggedVerifyMode.STATIC
stub(
    "sglang.srt.speculative.ragged_verify",
    RaggedVerifyMode=RaggedVerifyMode,
    read_ragged_verify_mode=lambda: ragged_mode,
)


class DeepseekV4HipRadixBackend:
    low_ratios = (1, 2)
    index_topk = 512
    low_ratio_candidate_span = 16384
    low_ratio_identity_skip = True


state = {"bs": [1, 2], "alignment": 1, "arch": "gfx1151"}
exec_context = types.SimpleNamespace(
    graph=types.SimpleNamespace(
        cuda_graph_config=types.SimpleNamespace(
            decode=types.SimpleNamespace(bs=state["bs"])
        ),
        torch_compile_max_bs=0,
    ),
    overlap=types.SimpleNamespace(enable_two_batch_overlap=False),
)
helper_globals = {
    "get_exec": lambda: exec_context,
    "get_flags": lambda: types.SimpleNamespace(
        capture=types.SimpleNamespace(enable_torch_compile=False)
    ),
    "get_cuda_graph_max_batch_size": lambda n: n,
    "get_cuda_graph_batch_size_alignment": lambda: state["alignment"],
}
helper = source_function(
    RUNTIME / "model_executor/runner/base_cuda_graph_runner.py",
    "get_batch_sizes_to_capture",
    helper_globals,
)
stub(
    "sglang.srt.model_executor.runner.base_cuda_graph_runner",
    get_batch_sizes_to_capture=helper,
)
stub(
    "torch",
    cuda=types.SimpleNamespace(
        get_device_properties=lambda _gpu: types.SimpleNamespace(
            gcnArchName=state["arch"]
        ),
        get_device_capability=lambda _gpu: (10, 0),
    ),
)
stub("sglang.srt.model_executor.forward_batch_info", ForwardMode=ForwardMode)
stub("sglang.srt.utils", is_gfx95_supported=lambda: False, is_hip=lambda: True)
stub(
    "sglang.srt.runtime_context", get_parallel=lambda: types.SimpleNamespace(tp_size=4)
)
stub(
    "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix",
    DeepseekV4HipRadixBackend=DeepseekV4HipRadixBackend,
)

# The bucket policy is stdlib-only; load the installed file under its dotted name
# so the lazy imports in graph_variants and the runner guard resolve to it.
policy_spec = importlib.util.spec_from_file_location(
    "sglang.srt.layers.graph_bucket_policy", RUNTIME / "layers/graph_bucket_policy.py"
)
assert policy_spec and policy_spec.loader
policy = importlib.util.module_from_spec(policy_spec)
sys.modules[policy_spec.name] = policy
policy_spec.loader.exec_module(policy)
assert not policy.exact_graph_buckets_enabled(), "run this test without the exact-bucket opt-in"

spec = importlib.util.spec_from_file_location(
    "installed_graph_variants", RUNTIME / "layers/attention/graph_variants.py"
)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

config = types.SimpleNamespace(
    model_type="deepseek_v41",
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    compress_ratios=(1, 2),
    index_topk=512,
)
runner = types.SimpleNamespace(
    model_config=types.SimpleNamespace(hf_text_config=config),
    spec_algorithm=types.SimpleNamespace(is_dspark=lambda: False),
    is_draft_worker=False,
    device="cuda",
    gpu_id=0,
    attn_backend=DeepseekV4HipRadixBackend(),
    server_args=types.SimpleNamespace(max_running_requests=8),
    req_to_token_pool=types.SimpleNamespace(size=8),
)
factory = mod.create_dsv41_candidate_graph_variants
assert helper(runner, 1)[0] == [1, 2]
variants = factory(runner, ForwardMode.DECODE, 1)
assert variants is not None
assert variants.capture_labels == (
    "candidate_all",
    "candidate_c2_all",
    "candidate_unfiltered",
    "candidate_filtered",
)

# Each supported capacity captures its exact prefix without padding rows.
for buckets in ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8]):
    state["bs"] = buckets
    exec_context.graph.cuda_graph_config.decode.bs = buckets
    assert helper(runner, 1)[0] == buckets
    selected = factory(runner, ForwardMode.DECODE, 1)
    assert selected is not None and selected.capture_labels == variants.capture_labels

    class Lengths:
        device = types.SimpleNamespace(type="cpu")

        def __init__(self, values):
            self.values = values

        def numel(self):
            return len(self.values)

        def max(self):
            return max(self.values)

    for bs in buckets:
        for length, label in (
            (512, "candidate_all"),
            (513, "candidate_c2_all"),
            (1024, "candidate_c2_all"),
            (1025, "candidate_unfiltered"),
            (16384, "candidate_unfiltered"),
            (16385, "candidate_filtered"),
        ):
            values = [1] * (bs - 1) + [length]
            for order in (values, list(reversed(values))):
                assert (
                    selected.select(types.SimpleNamespace(seq_lens_cpu=Lengths(order)))
                    == label
                )

# Missing an interior bucket remains unsupported.
for buckets in ([1, 4], [1, 2, 8]):
    state["bs"] = buckets
    exec_context.graph.cuda_graph_config.decode.bs = buckets
    assert helper(runner, 1)[0] == buckets
    assert factory(runner, ForwardMode.DECODE, 1) is None

# Pool filtering to a supported prefix remains eligible; alignment that removes
# C1 does not. Selection follows effective capture geometry, not pool capacity.
state["bs"] = [1, 2]
exec_context.graph.cuda_graph_config.decode.bs = state["bs"]
runner.req_to_token_pool.size = 1
assert helper(runner, 1)[0] == [1]
assert factory(runner, ForwardMode.DECODE, 1) is not None
runner.req_to_token_pool.size = 8
state["alignment"] = 2
assert helper(runner, 1)[0] == [2]
assert factory(runner, ForwardMode.DECODE, 1) is None
state["alignment"] = 1
assert factory(runner, ForwardMode.DECODE, 2) is None
assert factory(runner, ForwardMode.TARGET_VERIFY, 1) is None
state["arch"] = "gfx1152"
assert factory(runner, ForwardMode.DECODE, 1) is None

# Verify graph labels retain per-query causality, including the width added at
# the candidate-span boundary. Score-free decode labels must never be captured.
state["arch"] = "gfx1151"
runner.spec_algorithm.is_dspark = lambda: True
for width, capacity in ((2, 4), (4, 2)):
    for buckets in ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8]):
        exec_context.graph.cuda_graph_config.decode.bs = buckets
        verify = factory(runner, ForwardMode.TARGET_VERIFY, width)
        if buckets[-1] > capacity:
            assert verify is None
            continue
        assert verify is not None
        assert verify.capture_labels == ("candidate_unfiltered", "candidate_filtered")
        for length, expected in (
            (16384 - width, "candidate_unfiltered"),
            (16385 - width, "candidate_filtered"),
        ):
            assert (
                verify.select(types.SimpleNamespace(seq_lens_cpu=Lengths([length])))
                == expected
            )
    exec_context.graph.cuda_graph_config.decode.bs = [1, 2]
    ragged_mode = RaggedVerifyMode.COMPACT
    assert factory(runner, ForwardMode.TARGET_VERIFY, width) is None
    ragged_mode = RaggedVerifyMode.STATIC
    runner.is_draft_worker = True
    assert factory(runner, ForwardMode.TARGET_VERIFY, width) is None
    runner.is_draft_worker = False
for width in (1, 3, 5, 6):
    assert factory(runner, ForwardMode.TARGET_VERIFY, width) is None
runner.spec_algorithm.is_dspark = lambda: False
assert factory(runner, ForwardMode.TARGET_VERIFY, 4) is None

# Actual early guard admits every live size up to the largest bucket (the base
# runner pads between buckets; the prestage supplies zero rows) and nothing else.
tree = ast.parse(
    (RUNTIME / "model_executor/runner/decode_cuda_graph_runner.py").read_text()
)
runner_class = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "DecodeCudaGraphRunner"
)
method = next(
    node
    for node in runner_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "can_run_graph"
)
first_guard = method.body[0]
guard = compile(ast.Expression(first_guard.test), "decode_can_run_graph_guard", "eval")
base_tree = ast.parse(
    (RUNTIME / "model_executor/runner/base_cuda_graph_runner.py").read_text()
)
pad_node = next(
    node
    for cls in base_tree.body
    if isinstance(cls, ast.ClassDef) and cls.name == "BaseCudaGraphRunner"
    for node in cls.body
    if isinstance(node, ast.FunctionDef) and node.name == "_pad_to_bucket"
)
pad_node.decorator_list = []
pad_globals = {"bisect": __import__("bisect")}
exec(  # noqa: S102 -- exercise the installed rounding function itself
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    pad_node,
                ],
                type_ignores=[],
            )
        ),
        "_pad_to_bucket",
        "exec",
    ),
    pad_globals,
)
pad_to_bucket = pad_globals["_pad_to_bucket"]
# Use the installed prestage's own eligibility method with shape-only metadata.
# CUDA tensor copies are separately covered by engram-graph-prestage-cpu.py.
prestage_tree = ast.parse((RUNTIME / "layers/engram_graph_prestage.py").read_text())
prestage_class = next(
    node for node in prestage_tree.body if isinstance(node, ast.ClassDef)
)
can_stage_node = next(
    node
    for node in prestage_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "can_stage"
)
can_stage_globals = {}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[can_stage_node], type_ignores=[])),
        "installed_can_stage",
        "exec",
    ),
    can_stage_globals,
)
can_stage = can_stage_globals["can_stage"]


def metadata(bs, width):
    return types.SimpleNamespace(
        batch_size=bs,
        forward_mode=types.SimpleNamespace(
            is_decode=lambda: width == 1, is_target_verify=lambda: width > 1
        ),
        spec_info=types.SimpleNamespace(
            draft_token_num=width, num_tokens_per_req=width, ragged_verify_layout=None
        ),
        input_ids=types.SimpleNamespace(shape=(bs * width,)),
        positions=types.SimpleNamespace(shape=(bs * width,)),
        out_cache_loc=types.SimpleNamespace(shape=(bs * width,)),
        req_pool_indices=types.SimpleNamespace(shape=(bs,)),
        engram_packed_rows=None,
    )


for width, capacity in ((1, 8), (2, 4), (4, 2)):
    for buckets in ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8]):
        if buckets[-1] > capacity:
            continue
        provider = types.SimpleNamespace(req_width=width, capture_bs=buckets)
        provider.can_stage = lambda batch: can_stage(provider, batch)
        self_ = types.SimpleNamespace(
            engram_graph_prestage=provider, capture_bs=buckets
        )
        for bs in range(10):
            batch = metadata(bs, width)
            rejected = eval(guard, {"self": self_, "forward_batch": batch})
            assert rejected == (not 1 <= bs <= buckets[-1])
            if not rejected:
                assert pad_to_bucket(bs, buckets) == min(b for b in buckets if b >= bs)
        batch = metadata(1, width)
        batch.forward_mode.is_decode = batch.forward_mode.is_target_verify = lambda: (
            False
        )
        assert eval(guard, {"self": self_, "forward_batch": batch})

# Execute the installed construction guard with metadata only, including its
# allocation call. No rejected combination may allocate graph staging buffers.
init = next(
    node
    for node in runner_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
)
file_guard = next(
    node
    for node in ast.walk(init)
    if isinstance(node, ast.If)
    and isinstance(node.test, ast.Name)
    and node.test.id == "file_engram"
)
construction = compile(
    ast.fix_missing_locations(ast.Module(body=file_guard.body, type_ignores=[])),
    "installed_engram_construction",
    "exec",
)
for width, capacity in ((1, 8), (2, 4), (4, 2)):
    for invalid in (
        None,
        "wide",
        "buckets",
        "tp",
        "pp",
        "dp",
        "cp",
        "pdmux",
        "overlap",
        "algorithm",
        "mode",
        "draft",
        "compact",
    ):
        if width == 1 and invalid in ("draft", "compact"):
            continue
        buckets = [b for b in (1, 2, 4, 8) if b <= capacity]
        subject = types.SimpleNamespace(
            capture_forward_mode=ForwardMode.DECODE
            if width == 1
            else ForwardMode.TARGET_VERIFY,
            captured_req_width=width,
            capture_bs=buckets,
            max_bs=capacity,
            enable_pdmux=False,
            enable_two_batch_overlap=False,
            device="cuda",
        )
        parallel = types.SimpleNamespace(
            tp_size=4, pp_size=1, attn_dp_size=1, attn_cp_size=1
        )
        model_runner = types.SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=types.SimpleNamespace(
                is_speculative=lambda: width > 1, is_dspark=lambda: width > 1
            ),
        )
        ragged_mode = RaggedVerifyMode.STATIC
        if invalid == "wide":
            subject.max_bs = capacity * 2
        if invalid == "buckets":
            subject.capture_bs = [1, 4]
        if invalid == "tp":
            parallel.tp_size = 2
        if invalid == "pp":
            parallel.pp_size = 2
        if invalid == "dp":
            parallel.attn_dp_size = 2
        if invalid == "cp":
            parallel.attn_cp_size = 2
        if invalid == "pdmux":
            subject.enable_pdmux = True
        if invalid == "overlap":
            subject.enable_two_batch_overlap = True
        if invalid == "algorithm":
            model_runner.spec_algorithm.is_speculative = lambda: True
            model_runner.spec_algorithm.is_dspark = lambda: False
        if invalid == "mode":
            subject.capture_forward_mode = object()
        if invalid == "draft":
            model_runner.is_draft_worker = True
        if invalid == "compact":
            ragged_mode = RaggedVerifyMode.COMPACT
        allocations = []

        def allocate(_model, max_bs, _device, *, req_width, capture_bs):
            assert tuple(capture_bs) == tuple(buckets)
            allocations.append((max_bs, req_width))

        scope = dict(
            self=subject,
            model_runner=model_runner,
            inner_model=object(),
            ForwardMode=ForwardMode,
            get_parallel=lambda: parallel,
            NativeEngramGraphPrestage=allocate,
        )
        try:
            exec(construction, scope)
        except ValueError:
            assert invalid is not None and allocations == [], invalid
        else:
            assert invalid is None and allocations == [(capacity, width)], invalid
print(
    "PASS: decode and static DSpark graph policy, causal candidate boundaries, raw-batch eligibility/padding, and fail-closed construction guards"
)
