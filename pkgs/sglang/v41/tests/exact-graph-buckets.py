"""Exact decode graph buckets (SGLANG_DSV41_EXACT_GRAPH_BUCKETS) on CPU, with metadata stubs.

usage: exact-graph-buckets.py RUNTIME   (any environment; the test sets/clears the opt-in itself)

Checks, against the installed sources of the runtime:
  1. policy: which capture lists the Engram graphs accept with the opt-in off (the
     qualified power-of-two prefixes only, unchanged) and on (plus exact [1..n], n <= 8);
  2. selection: the installed get_batch_sizes_to_capture / _pad_to_bucket pick the
     exact bucket for every raw batch size of an exact list (zero padding rows), and
     the next power of two for the padded list;
  3. attention graph variants (graph_variants.py) for decode and static DSpark verify
     widths 2/4, with the eight-token-row cap;
  4. the runner's construction guard (extracted from DecodeCudaGraphRunner.__init__) and
     the stager it allocates;
  5. the real NativeEngramGraphPrestage on CPU tensors: hashes, looks up and installs
     exactly bs * width rows for every batch size 0..9, never installs padding for an
     exact list, still zero-fills padding for the padded list, and refuses a bucket that
     is not the smallest one holding the batch.
No device, model or collective is initialised; this is not GPU graph-capture evidence.
"""

import ast
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

RUNTIME = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages/sglang/srt"
ENV = "SGLANG_DSV41_EXACT_GRAPH_BUCKETS"


def set_opt_in(value):
    if value is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = value


def stub(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    sys.modules[name] = module
    return module


def load(name, relative, register=True):
    spec = importlib.util.spec_from_file_location(name, RUNTIME / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fails(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def source_function(path, name, globals_):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_)  # noqa: S102 -- installed function with metadata stubs
    return globals_[name]


# ---------------------------------------------------------------------------------------
# 1. policy
# ---------------------------------------------------------------------------------------
set_opt_in(None)
policy = load("sglang.srt.layers.graph_bucket_policy", "layers/graph_bucket_policy.py")
# The stager imports the real engram_store, so it is loaded before any sglang/torch stub below.
prestage_module = load("installed_engram_graph_prestage", "layers/engram_graph_prestage.py", register=False)
LEGACY = ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8])
LADDERS = [list(range(1, n + 1)) for n in range(1, 9)]
NOT_LADDERS = ([], [2], [1, 3], [1, 2, 4, 5], [1, 4], [1, 2, 8], [1, 2, 2, 3], [0, 1], [2, 3, 4], list(range(1, 10)), [1, 3, 2])

for value in (None, "", "0", "false", "off", "no", " 0 ", "FALSE"):
    set_opt_in(value)
    assert not policy.exact_graph_buckets_enabled(), value
for value in ("1", "true", "on", "yes", "exact"):
    set_opt_in(value)
    assert policy.exact_graph_buckets_enabled(), value

for ladder in LADDERS:
    assert policy.is_exact_ladder(ladder) and policy.is_exact_ladder(tuple(ladder))
for other in NOT_LADDERS:
    assert not policy.is_exact_ladder(other), other

set_opt_in(None)
for buckets in LEGACY:
    assert policy.engram_capture_bs_allowed(buckets)
for buckets in [b for b in LADDERS if b not in [list(x) for x in LEGACY]] + list(NOT_LADDERS):
    assert not policy.engram_capture_bs_allowed(buckets), ("default must stay closed", buckets)
set_opt_in("1")
for buckets in LEGACY + tuple(LADDERS):
    assert policy.engram_capture_bs_allowed(buckets), buckets
for buckets in NOT_LADDERS:
    # [1,2,4,5] / [1,4] ... stay refused: the opt-in admits exact ladders, not arbitrary lists.
    assert not policy.engram_capture_bs_allowed(buckets), ("opt-in must stay closed", buckets)

# prestage_capture_bs: None is the installed derivation; an explicit list must end at max_bs.
set_opt_in(None)
for max_bs, expect in ((1, (1,)), (2, (1, 2)), (4, (1, 2, 4)), (8, (1, 2, 4, 8))):
    assert policy.prestage_capture_bs(max_bs) == expect
    assert policy.prestage_capture_bs(max_bs, list(expect)) == expect
for max_bs in (0, 3, 5, 6, 7, 9, 16):
    fails(ValueError, lambda m=max_bs: policy.prestage_capture_bs(m))
fails(ValueError, lambda: policy.prestage_capture_bs(4, [1, 2, 3, 4]))      # closed by default
fails(ValueError, lambda: policy.prestage_capture_bs(4, [1, 2]))            # must end at max_bs
set_opt_in("1")
assert policy.prestage_capture_bs(4, [1, 2, 3, 4]) == (1, 2, 3, 4)
assert policy.prestage_capture_bs(8, list(range(1, 9))) == tuple(range(1, 9))
assert policy.prestage_capture_bs(4, [1, 2, 4]) == (1, 2, 4)                # legacy lists unchanged
fails(ValueError, lambda: policy.prestage_capture_bs(3, [1, 2, 4]))
fails(ValueError, lambda: policy.prestage_capture_bs(9, list(range(1, 10))))
fails(ValueError, lambda: policy.prestage_capture_bs(4, [1, 2, 4, 5]))
set_opt_in(None)

# ---------------------------------------------------------------------------------------
# 2. selection with the installed helpers
# ---------------------------------------------------------------------------------------
state = {"alignment": 1, "arch": "gfx1151"}
exec_context = types.SimpleNamespace(
    graph=types.SimpleNamespace(
        cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(bs=[1, 2])),
        torch_compile_max_bs=0,
    ),
    overlap=types.SimpleNamespace(enable_two_batch_overlap=False),
)
helper = source_function(
    RUNTIME / "model_executor/runner/base_cuda_graph_runner.py",
    "get_batch_sizes_to_capture",
    {
        "get_exec": lambda: exec_context,
        "get_flags": lambda: types.SimpleNamespace(capture=types.SimpleNamespace(enable_torch_compile=False)),
        "get_cuda_graph_max_batch_size": lambda n: n,
        "get_cuda_graph_batch_size_alignment": lambda: state["alignment"],
    },
)
stub("sglang.srt.model_executor.runner.base_cuda_graph_runner", get_batch_sizes_to_capture=helper)

base_tree = ast.parse((RUNTIME / "model_executor/runner/base_cuda_graph_runner.py").read_text())
pad_node = next(
    node
    for cls in base_tree.body
    if isinstance(cls, ast.ClassDef) and cls.name == "BaseCudaGraphRunner"
    for node in cls.body
    if isinstance(node, ast.FunctionDef) and node.name == "_pad_to_bucket"
)
pad_node.decorator_list = []
pad_globals = {"bisect": __import__("bisect")}
exec(  # noqa: S102 -- the installed rounding function itself
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), pad_node],
                type_ignores=[],
            )
        ),
        "_pad_to_bucket",
        "exec",
    ),
    pad_globals,
)
pad_to_bucket = pad_globals["_pad_to_bucket"]


def runner_for(pool):
    return types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(size=pool))


def capture_list(configured, pool, width=1):
    exec_context.graph.cuda_graph_config.decode.bs = list(configured)
    return helper(runner_for(pool), width)[0]


# Configured exact lists survive the installed filters untouched.
for n in range(1, 9):
    assert capture_list(range(1, n + 1), n) == list(range(1, n + 1))
# A longer configured list is clipped to the request pool (max-running-requests).
assert capture_list(range(1, 9), 4) == [1, 2, 3, 4]
assert capture_list([1, 2, 4], 3) == [1, 2, 3]        # pool clips a padded list into a non-prefix list
# Exact lists: every raw size has its own bucket.
for n in range(1, 9):
    buckets = list(range(1, n + 1))
    for bs in buckets:
        assert pad_to_bucket(bs, buckets) == bs
    fails(AssertionError, lambda b=buckets, n=n: pad_to_bucket(n + 1, b))
# Padded list: the 0039 rounding.
assert [pad_to_bucket(bs, [1, 2, 4]) for bs in (1, 2, 3, 4)] == [1, 2, 4, 4]
assert [pad_to_bucket(bs, [1, 2, 4, 8]) for bs in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]


def padding_rows(buckets, width):
    return {bs: (pad_to_bucket(bs, buckets) - bs) * width for bs in range(1, buckets[-1] + 1)}


assert padding_rows([1, 2, 4], 1) == {1: 0, 2: 0, 3: 1, 4: 0}
assert padding_rows([1, 2, 3, 4], 1) == {1: 0, 2: 0, 3: 0, 4: 0}
assert padding_rows([1, 2, 4], 2) == {1: 0, 2: 0, 3: 2, 4: 0}          # DSpark gamma 1 verify rows
assert padding_rows([1, 2, 3, 4], 2) == {1: 0, 2: 0, 3: 0, 4: 0}
assert padding_rows([1, 2, 4, 8], 1) == {1: 0, 2: 0, 3: 1, 4: 0, 5: 3, 6: 2, 7: 1, 8: 0}
assert all(v == 0 for v in padding_rows(list(range(1, 9)), 1).values())

# ---------------------------------------------------------------------------------------
# 3. attention graph variants
# ---------------------------------------------------------------------------------------
class ForwardMode:
    DECODE = object()
    TARGET_VERIFY = object()


class RaggedVerifyMode(enum.Enum):
    STATIC = "static"
    COMPACT = "compact"


stub(
    "sglang.srt.speculative.ragged_verify",
    RaggedVerifyMode=RaggedVerifyMode,
    read_ragged_verify_mode=lambda: RaggedVerifyMode.STATIC,
)


class DeepseekV4HipRadixBackend:
    low_ratios = (1, 2)
    index_topk = 512
    low_ratio_candidate_span = 16384
    low_ratio_identity_skip = True


stub(
    "torch",
    cuda=types.SimpleNamespace(
        get_device_properties=lambda _gpu: types.SimpleNamespace(gcnArchName=state["arch"]),
        get_device_capability=lambda _gpu: (10, 0),
    ),
)
stub("sglang.srt.model_executor.forward_batch_info", ForwardMode=ForwardMode)
stub("sglang.srt.utils", is_gfx95_supported=lambda: False, is_hip=lambda: True)
stub("sglang.srt.runtime_context", get_parallel=lambda: types.SimpleNamespace(tp_size=4))
stub(
    "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix",
    DeepseekV4HipRadixBackend=DeepseekV4HipRadixBackend,
)
variants_module = load("installed_graph_variants", "layers/attention/graph_variants.py")
text_config = types.SimpleNamespace(
    model_type="deepseek_v41",
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    compress_ratios=(1, 2),
    index_topk=512,
)
model_runner = types.SimpleNamespace(
    model_config=types.SimpleNamespace(hf_text_config=text_config),
    spec_algorithm=types.SimpleNamespace(is_dspark=lambda: False),
    is_draft_worker=False,
    device="cuda",
    gpu_id=0,
    attn_backend=DeepseekV4HipRadixBackend(),
    req_to_token_pool=types.SimpleNamespace(size=8),
)
factory = variants_module.create_dsv41_candidate_graph_variants


def variants(mode, width, buckets, dspark=False):
    exec_context.graph.cuda_graph_config.decode.bs = list(buckets)
    model_runner.spec_algorithm.is_dspark = lambda: dspark
    return factory(model_runner, mode, width)


DECODE_LABELS = ("candidate_all", "candidate_c2_all", "candidate_unfiltered", "candidate_filtered")
VERIFY_LABELS = ("candidate_unfiltered", "candidate_filtered")
# (width, largest bucket the eight-row cap allows)
for opt_in in (None, "1"):
    set_opt_in(opt_in)
    for buckets in LEGACY:
        got = variants(ForwardMode.DECODE, 1, buckets)
        assert got is not None and got.capture_labels == DECODE_LABELS, (opt_in, buckets)
    for n in range(1, 9):
        ladder = list(range(1, n + 1))
        in_legacy = ladder in [list(x) for x in LEGACY]
        got = variants(ForwardMode.DECODE, 1, ladder)
        if in_legacy or opt_in:
            assert got is not None and got.capture_labels == DECODE_LABELS, (opt_in, ladder)
        else:
            assert got is None, ("default must refuse", ladder)
    for width, cap in ((2, 4), (4, 2)):
        for n in range(1, 9):
            ladder = list(range(1, n + 1))
            in_legacy = ladder in [list(x) for x in LEGACY]
            got = variants(ForwardMode.TARGET_VERIFY, width, ladder, dspark=True)
            allowed = (in_legacy or opt_in) and n <= cap
            if allowed:
                assert got is not None and got.capture_labels == VERIFY_LABELS, (opt_in, width, ladder)
            else:
                assert got is None, (opt_in, width, ladder)
    for refused in ([1, 4], [1, 2, 8], [1, 2, 4, 5], [2, 3, 4]):
        assert variants(ForwardMode.DECODE, 1, refused) is None, (opt_in, refused)
    # The draft worker and non-DSpark verify never take the native variants.
    model_runner.is_draft_worker = True
    assert variants(ForwardMode.TARGET_VERIFY, 2, [1, 2], dspark=True) is None
    model_runner.is_draft_worker = False
    assert variants(ForwardMode.TARGET_VERIFY, 2, [1, 2], dspark=False) is None
set_opt_in(None)

# ---------------------------------------------------------------------------------------
# 4. runner construction guard + stager allocation
# ---------------------------------------------------------------------------------------
runner_tree = ast.parse((RUNTIME / "model_executor/runner/decode_cuda_graph_runner.py").read_text())
runner_class = next(n for n in runner_tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeCudaGraphRunner")
init = next(n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
file_guard = next(
    n for n in ast.walk(init) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "file_engram"
)
construction = compile(
    ast.fix_missing_locations(ast.Module(body=file_guard.body, type_ignores=[])),
    "installed_engram_construction",
    "exec",
)


def construct(width, buckets, *, max_bs=None, draft=False, ragged=RaggedVerifyMode.STATIC, parallel_overrides=None):
    subject = types.SimpleNamespace(
        capture_forward_mode=ForwardMode.DECODE if width == 1 else ForwardMode.TARGET_VERIFY,
        captured_req_width=width,
        capture_bs=list(buckets),
        max_bs=buckets[-1] if max_bs is None else max_bs,
        enable_pdmux=False,
        enable_two_batch_overlap=False,
        device="cuda",
    )
    parallel = types.SimpleNamespace(tp_size=4, pp_size=1, attn_dp_size=1, attn_cp_size=1)
    for key, value in (parallel_overrides or {}).items():
        setattr(parallel, key, value)
    runner = types.SimpleNamespace(
        is_draft_worker=draft,
        spec_algorithm=types.SimpleNamespace(is_speculative=lambda: width > 1, is_dspark=lambda: width > 1),
    )
    stub(
        "sglang.srt.speculative.ragged_verify",
        RaggedVerifyMode=RaggedVerifyMode,
        read_ragged_verify_mode=lambda: ragged,
    )
    allocations = []

    def allocate(_model, max_bs, _device, *, req_width, capture_bs):
        allocations.append((max_bs, req_width, tuple(capture_bs)))

    scope = dict(
        self=subject,
        model_runner=runner,
        inner_model=object(),
        ForwardMode=ForwardMode,
        get_parallel=lambda: parallel,
        NativeEngramGraphPrestage=allocate,
    )
    exec(construction, scope)  # noqa: S102 -- the installed construction guard
    return allocations


for opt_in in (None, "1"):
    set_opt_in(opt_in)
    for width, cap in ((1, 8), (2, 4), (4, 2)):
        for n in range(1, 9):
            ladder = list(range(1, n + 1))
            in_legacy = ladder in [list(x) for x in LEGACY]
            admitted = (in_legacy or opt_in) and n * width <= 8
            if admitted:
                assert construct(width, ladder) == [(n, width, tuple(ladder))], (opt_in, width, ladder)
            else:
                fails(ValueError, lambda w=width, l=ladder: construct(w, l))
        for refused in ([1, 4], [1, 2, 8], [1, 2, 4, 5], [2, 3, 4]):
            if refused[-1] * width > 8:
                continue
            fails(ValueError, lambda w=width, l=refused: construct(w, l))
    # Every other qualification of the guard still applies with the opt-in on.
    for overrides in ({"tp_size": 2}, {"pp_size": 2}, {"attn_dp_size": 2}, {"attn_cp_size": 2}):
        fails(ValueError, lambda o=overrides: construct(1, [1, 2], parallel_overrides=o))
    fails(ValueError, lambda: construct(2, [1, 2], draft=True))
    fails(ValueError, lambda: construct(2, [1, 2], ragged=RaggedVerifyMode.COMPACT))
    fails(ValueError, lambda: construct(1, [1, 2, 4], max_bs=16))   # more than eight token rows
set_opt_in(None)

# ---------------------------------------------------------------------------------------
# 5. the real stager on CPU tensors
# ---------------------------------------------------------------------------------------
events = []


class Event:
    def record(self, current):
        events.append(("record", self))

    def synchronize(self):
        events.append(("synchronize", self))


def allocate_tensor(shape, *, dtype, device=None, pin_memory=False):
    return torch.zeros(shape, dtype=dtype)


prestage_module.torch = types.SimpleNamespace(
    zeros=allocate_tensor,
    empty=allocate_tensor,
    uint8=torch.uint8,
    cuda=types.SimpleNamespace(
        Event=Event, current_stream=lambda: types.SimpleNamespace(wait_event=lambda event: events.append(("wait", event)))
    ),
)


class Hasher:
    def __init__(self):
        self.rows_seen = []

    def __call__(self, input_ids, batch):
        self.rows_seen.append(input_ids.numel())
        return batch.hash_ids


class Store:
    def __init__(self, layer):
        self.layer = layer

    def lookup(self, ids):
        values = (ids % 100 + 1 + self.layer * 100).to(torch.uint8)  # never zero: live rows are distinguishable from padding
        return types.SimpleNamespace(packed=values[:, None].expand(-1, prestage_module.ROW_BYTES).numpy().copy())


def model():
    return types.SimpleNamespace(
        engram_hasher=Hasher(),
        layers=[
            types.SimpleNamespace(
                engram=types.SimpleNamespace(layer_hash_index=i, embed=types.SimpleNamespace(file_store=Store(i)))
            )
            for i in range(2)
        ],
    )


def batch(bs, width=1, shift=0):
    rows = bs * width
    return types.SimpleNamespace(
        batch_size=bs,
        forward_mode=types.SimpleNamespace(is_decode=lambda: width == 1, is_target_verify=lambda: width > 1),
        spec_info=types.SimpleNamespace(draft_token_num=width, num_tokens_per_req=width, ragged_verify_layout=None)
        if width > 1
        else None,
        input_ids=torch.arange(rows),
        positions=torch.arange(rows),
        req_pool_indices=torch.arange(bs),
        out_cache_loc=torch.arange(1, rows + 1),
        engram_packed_rows=None,
        hash_ids=torch.arange(rows * 48).reshape(rows, 2, 24) + shift,
    )


def cycle(stager, current, graph_bs):
    generation = stager.prepare(current, graph_bs)
    stager.install(current, generation)
    rows = stager.gpu_rows[: graph_bs * stager.req_width].clone()
    live = current.engram_packed_rows
    assert live.shape[0] == current.batch_size * stager.req_width, "the batch view exposes the live rows only"
    stager.retire_after_replay(generation)
    return rows


def nonzero_rows(rows):
    return int((rows.reshape(rows.shape[0], -1) != 0).any(dim=1).sum())


def expected_rows(current, graph_rows):
    """What the stand-in store produces for the live rows, zeros for the padding rows."""
    live = current.hash_ids.shape[0]
    out = torch.zeros((graph_rows, 2, 24, prestage_module.ROW_BYTES), dtype=torch.uint8)
    for layer in range(2):
        values = (current.hash_ids[:, layer] % 100 + 1 + layer * 100).to(torch.uint8)
        out[:live, layer] = values[:, :, None].expand(-1, -1, prestage_module.ROW_BYTES)
    return out


PrestageClass = prestage_module.NativeEngramGraphPrestage

# Default (no opt-in): constructor semantics are unchanged.
set_opt_in(None)
default_stager = PrestageClass(model(), 4, None)
assert default_stager.capture_bs == (1, 2, 4)
assert PrestageClass(model(), 4, None, capture_bs=[1, 2, 4]).capture_bs == (1, 2, 4)
assert PrestageClass(model(), 8, None, capture_bs=(1, 2, 4, 8)).capture_bs == (1, 2, 4, 8)
fails(ValueError, lambda: PrestageClass(model(), 3, None))
fails(ValueError, lambda: PrestageClass(model(), 4, None, capture_bs=[1, 2, 3, 4]))
fails(ValueError, lambda: PrestageClass(model(), 4, None, capture_bs=[1, 2]))
fails(ValueError, lambda: PrestageClass(model(), 4, None, req_width=4))                  # 16 rows
fails(ValueError, lambda: PrestageClass(types.SimpleNamespace(engram_hasher=None, layers=[]), 4, None))
fails(ValueError, lambda: PrestageClass(types.SimpleNamespace(engram_hasher=None, layers=[]), 4, None, capture_bs=[1, 2, 4]))

# Padded [1,2,4]: bs 3 replays bucket 4, padding row is zero, one live row fewer hashed.
padded = PrestageClass(model(), 4, None, capture_bs=[1, 2, 4])
current = batch(3)
rows = cycle(padded, current, 4)
assert padded.hasher.rows_seen[-1] == 3 and nonzero_rows(rows) == 3 and rows.shape[0] == 4
assert not rows[3].any(), "padding row must be zeros"
assert torch.equal(rows, expected_rows(current, 4))
fails(ValueError, lambda: padded.prepare(batch(3), 3))                                    # not a captured bucket
set_opt_in(None)

set_opt_in("1")
for width, cap in ((1, 8), (2, 4), (4, 2)):
    for n in range(1, cap + 1):
        ladder = list(range(1, n + 1))
        stager = PrestageClass(model(), n, None, req_width=width, capture_bs=ladder)
        assert stager.capture_bs == tuple(ladder) and stager.req_width == width
        assert tuple(stager.gpu_rows.shape) == (n * width, 2, 24, prestage_module.ROW_BYTES)
        assert len(stager.host_frames) == 2 and tuple(stager.host_frames[0].shape) == tuple(stager.gpu_rows.shape)
        for bs in range(0, 10):
            current = batch(bs, width, shift=bs)
            if not 1 <= bs <= n:
                assert not stager.can_stage(current), (width, ladder, bs)
                fails(ValueError, lambda s=stager, c=current: s.prepare(c, max(ladder)))
                continue
            assert stager.can_stage(current)
            before = len(stager.hasher.rows_seen)
            rows = cycle(stager, current, bs)                       # the exact bucket
            assert len(stager.hasher.rows_seen) == before + 1 and stager.hasher.rows_seen[-1] == bs * width
            assert rows.shape[0] == bs * width, "exact bucket installs no padding rows"
            assert nonzero_rows(rows) == bs * width, "every installed row is live"
            assert torch.equal(rows, expected_rows(current, bs * width))
            # A larger bucket than the smallest one holding the batch is refused (no silent padding).
            for larger in ladder:
                if larger != bs:
                    fails(ValueError, lambda s=stager, c=batch(bs, width), g=larger: s.prepare(c, g))
        # Replaying a bigger batch after a smaller one never leaks the older rows.
        small, big = batch(1, width, shift=7), batch(n, width, shift=3)
        cycle(stager, small, 1)
        rows = cycle(stager, big, n)
        assert torch.equal(rows, expected_rows(big, n * width))
        small = batch(1, width, shift=11)                           # a batch carries its rows once: build fresh ones
        rows = cycle(stager, small, 1)
        assert torch.equal(rows, expected_rows(small, width))
    # Opt-in keeps the other limits: widths x buckets beyond eight rows, non-ladder lists, wrong end.
    fails(ValueError, lambda w=width, c=cap: PrestageClass(model(), c + 1, None, req_width=w, capture_bs=list(range(1, c + 2))))
    fails(ValueError, lambda w=width: PrestageClass(model(), 4, None, req_width=w, capture_bs=[1, 2, 4, 5]))
    fails(ValueError, lambda w=width: PrestageClass(model(), 4, None, req_width=w, capture_bs=[1, 2]))
    fails(ValueError, lambda w=width: PrestageClass(model(), 4, None, req_width=w, capture_bs=[1, 3, 4]))
# Legacy lists with the opt-in on behave exactly as without it (padding rows still zero).
stager = PrestageClass(model(), 4, None, capture_bs=[1, 2, 4])
rows = cycle(stager, batch(3), 4)
assert nonzero_rows(rows) == 3 and not rows[3].any() and stager.capture_bs == (1, 2, 4)
assert torch.equal(rows, expected_rows(batch(3), 4))
set_opt_in(None)

print(
    "PASS: exact [1..n] graph buckets are opt-in only; selection, attention variants, construction guard "
    "and Engram stager for decode and DSpark widths 2/4 install no padding rows; default lists unchanged"
)
