"""Installed graph geometry regression with explicit CPU metadata stubs.

No device/model is initialized. This checks capture selection and early replay
eligibility, not GPU graph execution or collective correctness.
"""

import ast
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

# Actual candidate early guard excludes non-exact buckets before prestage.
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
for buckets in ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8]):
    self_ = types.SimpleNamespace(engram_graph_prestage=object(), capture_bs=buckets)
    for bs in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        batch = types.SimpleNamespace(
            batch_size=bs, forward_mode=types.SimpleNamespace(is_decode=lambda: True)
        )
        # Evaluate only the installed guard with the literal metadata above.
        assert eval(guard, {"self": self_, "forward_batch": batch}) == (
            bs not in buckets
        )
print(
    "PASS: exact capacity prefixes1/2/4/8; limits512/1024/16384 for each bucket and row reversal; unsupported geometry declined; actual early eligibility excludes padding"
)
