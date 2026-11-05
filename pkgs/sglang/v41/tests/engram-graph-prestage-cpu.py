"""Exercise the installed Engram graph prestage on CPU, including replay padding.

Uses real CPU tensors for row order, byte copies, retained views and frame reuse.
CUDA allocation/events are explicit stubs, and the hasher/store are deterministic
stand-ins. This does not qualify native hash arithmetic, DMA ordering, HIP graph
capture or TP collectives.

Contract under test: a batch of bs live rows replays the smallest captured bucket
(C3 -> C4 with [1,2,4]); only the live rows are hashed and looked up, and the
padding rows of the replay are all-zero native rows, never a previous step's rows.
"""

import ast
import copy
import importlib.util
import sys
import types
from pathlib import Path

import torch


RUNTIME = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages/sglang/srt"
spec = importlib.util.spec_from_file_location(
    "installed_engram_graph_prestage", RUNTIME / "layers/engram_graph_prestage.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
events = []


class Event:
    def record(self, current):
        events.append(("record", self))

    def synchronize(self):
        events.append(("synchronize", self))


def allocate(shape, *, dtype, device=None, pin_memory=False):
    assert device in (None, "cpu")
    return torch.zeros(shape, dtype=dtype)


stream = types.SimpleNamespace(wait_event=lambda event: events.append(("wait", event)))
module.torch = types.SimpleNamespace(
    zeros=allocate,
    empty=allocate,
    uint8=torch.uint8,
    cuda=types.SimpleNamespace(Event=Event, current_stream=lambda: stream),
)


class Hasher:
    calls = 0
    rows_seen = []

    def __call__(self, input_ids, batch):
        self.calls += 1
        self.rows_seen.append(input_ids.numel())
        return batch.hash_ids


class Store:
    def __init__(self, layer):
        self.layer = layer
        self.calls = []

    def lookup(self, ids):
        self.calls.append(ids.clone())
        # 1..200 and never zero, so a live row is distinguishable from padding.
        values = (ids % 100 + 1 + self.layer * 100).to(torch.uint8)
        packed = values[:, None].expand(-1, module.ROW_BYTES).numpy().copy()
        return types.SimpleNamespace(packed=packed)


def model():
    stores = [Store(i) for i in range(2)]
    return types.SimpleNamespace(
        engram_hasher=Hasher(),
        layers=[
            types.SimpleNamespace(
                engram=types.SimpleNamespace(
                    layer_hash_index=i,
                    embed=types.SimpleNamespace(file_store=store),
                )
            )
            for i, store in enumerate(stores)
        ],
    )


def batch(bs, shift=0, width=1):
    rows = bs * width
    return types.SimpleNamespace(
        batch_size=bs,
        forward_mode=types.SimpleNamespace(
            is_decode=lambda: width == 1, is_target_verify=lambda: width > 1
        ),
        spec_info=types.SimpleNamespace(
            draft_token_num=width, num_tokens_per_req=width, ragged_verify_layout=None
        )
        if width > 1
        else None,
        input_ids=torch.arange(rows),
        positions=torch.arange(rows),
        req_pool_indices=torch.arange(bs),
        out_cache_loc=torch.arange(1, rows + 1),
        engram_packed_rows=None,
        hash_ids=(torch.arange(rows * 48).reshape(rows, 2, 24) + shift),
    )


def fails(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def bucket_for(bs, buckets):
    return min(b for b in buckets if b >= bs)


Prestage = module.NativeEngramGraphPrestage


def replay(provider, current, graph_bs):
    """One full prepare/install/retire cycle; returns the installed padded rows."""
    before = provider.hasher.calls
    generation = provider.prepare(current, graph_bs)
    assert provider.hasher.calls == before + 1  # one canonical hasher call, live rows
    assert provider.hasher.rows_seen[-1] == current.input_ids.numel()
    provider.install(current, generation)
    padded = provider.gpu_rows[: graph_bs * provider.req_width].clone()
    # The batch view exposes the live rows only; storage is the captured buffer.
    assert current.engram_packed_rows.shape == (
        current.batch_size * provider.req_width,
        2,
        24,
        module.ROW_BYTES,
    )
    assert current.engram_packed_rows.data_ptr() == provider.gpu_rows.data_ptr()
    provider.retire_after_replay(generation)
    assert provider.pending is provider.active_generation is None
    return generation, padded


# Every supported capacity: exact buckets unchanged, in-between sizes pad up.
for max_bs in (1, 2, 4, 8):
    buckets = tuple(b for b in (1, 2, 4, 8) if b <= max_bs)
    provider = Prestage(model(), max_bs, "cpu")
    assert provider.capture_bs == buckets
    for step, bs in enumerate(range(1, max_bs + 1)):
        graph_bs = bucket_for(bs, buckets)
        current = batch(bs, 17 * step)
        # Anything but the runner's rounding is rejected before the hasher runs.
        calls, generation_before = provider.hasher.calls, provider.generation
        for wrong in range(0, 11):
            if wrong != graph_bs:
                fails(ValueError, lambda: provider.prepare(current, wrong))
        current.forward_mode.is_decode = lambda: False
        fails(ValueError, lambda: provider.prepare(current, graph_bs))
        current.forward_mode.is_decode = lambda: True
        assert provider.hasher.calls == calls
        assert provider.generation == generation_before
        _, padded = replay(provider, current, graph_bs)
        assert padded.shape == (graph_bs, 2, 24, module.ROW_BYTES)
        for layer, store in enumerate(provider.stores):
            ids = current.hash_ids[:, layer]
            # Only live ids reach the store: no padding lookups.
            assert torch.equal(store.calls[-1], ids.reshape(-1))
            expected = (ids % 100 + 1 + layer * 100).to(torch.uint8)
            assert torch.equal(
                padded[:bs, layer], expected[..., None].expand(-1, -1, module.ROW_BYTES)
            )
            assert not padded[bs:, layer].any()  # padding rows are zero rows
    # Sizes outside 1..max never reach the hasher.
    calls = provider.hasher.calls
    for bs in (0, max_bs + 1, 9, 16):
        for graph_bs in range(0, 17):
            fails(ValueError, lambda: provider.prepare(batch(bs), graph_bs))
    assert provider.hasher.calls == calls

# Padding rows must be zero even right after a full batch filled every row, and
# across both host frames and the reused device buffer.
provider = Prestage(model(), 4, "cpu")
assert provider.capture_bs == (1, 2, 4)
for index, bs in enumerate((4, 3, 2, 1, 3, 4, 3, 3, 4, 1, 3)):
    graph_bs = bucket_for(bs, provider.capture_bs)
    current = batch(bs, 13 * index)
    generation, padded = replay(provider, current, graph_bs)
    assert generation == index + 1
    assert padded[:bs].any(dim=-1).all()  # every live row carries data
    if graph_bs > bs:
        assert not padded[bs:].any()  # includes bs=3 right after a nonzero bs=4
    # Unused host-frame rows are cleared, as for exact buckets.
    frame = provider.host_frames[(generation - 1) % 2]
    assert torch.count_nonzero(frame[bs:]) == 0

# Ownership/ordering guards are unchanged for padded batches.
provider = Prestage(model(), 4, "cpu")
current = batch(3)
generation = provider.prepare(current, 4)
fails(RuntimeError, lambda: provider.prepare(current, 4))  # not retired
fails(RuntimeError, lambda: provider.install(current, generation - 1))  # stale
other = batch(2)
fails(RuntimeError, lambda: provider.install(other, generation))  # batch changed
provider.install(current, generation)
fails(RuntimeError, lambda: provider.install(current, generation))  # once only
fails(RuntimeError, lambda: provider.retire_after_replay(generation - 1))
fails(RuntimeError, lambda: provider.prepare(batch(3), 4))  # still active
provider.retire_after_replay(generation)
assert provider.pending is provider.active_generation is None
fails(RuntimeError, lambda: provider.install(batch(3), generation))
# Both events and waits are exercised; device-ordering semantics are not.
assert any(kind == "wait" for kind, _ in events)
assert any(kind == "synchronize" for kind, _ in events)
assert not torch.cuda.is_initialized()
print(
    "PASS: exact buckets unchanged; in-between sizes replay the next bucket; "
    "live-row-only hash/lookup; zero padding rows after full batches; "
    "stale/ownership guards; CUDA uninitialized"
)


# Fixed-width verification: raw request counts and flattened token rows differ.
for width, max_bs in ((2, 4), (4, 2)):
    provider = Prestage(model(), max_bs, "cpu", req_width=width)
    for step, bs in enumerate((max_bs, max_bs - 1, 1, max_bs, 1)):
        current = batch(bs, step * 11, width)
        graph_bs = bucket_for(bs, provider.capture_bs)
        rows = bs * width
        assert provider.can_stage(current)
        _, padded = replay(provider, current, graph_bs)
        for layer, store in enumerate(provider.stores):
            ids = current.hash_ids[:, layer]
            assert torch.equal(store.calls[-1], ids.reshape(-1))
            expected = (ids % 100 + 1 + layer * 100).to(torch.uint8)
            assert torch.equal(
                padded[:rows, layer],
                expected[..., None].expand(-1, -1, module.ROW_BYTES),
            )
        assert not padded[rows:].any()
        assert not provider.host_frames[(provider.generation - 1) % 2][rows:].any()
    # Reject mismatched widths, ragged layouts, mode and row geometry before hash.
    for field, value in (
        ("draft_token_num", width + 1),
        ("num_tokens_per_req", width + 1),
        ("ragged_verify_layout", object()),
    ):
        current = batch(1, width=width)
        setattr(current.spec_info, field, value)
        calls, generation_before = provider.hasher.calls, provider.generation
        assert not provider.can_stage(current)
        fails(ValueError, lambda: provider.prepare(current, 1))
        assert (provider.hasher.calls, provider.generation) == (
            calls,
            generation_before,
        )
    for field in ("input_ids", "positions", "req_pool_indices", "out_cache_loc"):
        current = batch(1, width=width)
        setattr(current, field, torch.arange(7))
        assert not provider.can_stage(current)
        fails(ValueError, lambda: provider.prepare(current, 1))
    assert not provider.can_stage(batch(1))  # ordinary decode on a verify capture
    current = batch(1, width=width)
    generation = provider.prepare(current, 1)
    # Even an identically shaped replacement batch cannot steal a generation.
    fails(RuntimeError, lambda: provider.install(batch(1, width=width), generation))
    provider.install(current, generation)
    provider.retire_after_replay(generation)
for width, max_bs in ((0, 1), (3, 1), (5, 1), (2, 8), (4, 4)):
    fails(ValueError, lambda: Prestage(model(), max_bs, "cpu", req_width=width))

# Exercise the installed canonical CPU hasher, not the deterministic stub, while
# avoiding imports of GPU launchers and distributed runtime state. The extracted
# class and hash function are unchanged; only construction and CUDA dispatch are
# replaced by a small explicit fixture. This tests integer hashing/history, not
# GPU logits or acceptance quality.
source = RUNTIME / "layers/engram.py"
tree = ast.parse(source.read_text(), filename=str(source))
selected = [
    node
    for node in tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    and node.name in ("EngramHasher", "compute_engram_hash_ids")
]
namespace = {
    "torch": torch,
    "nn": torch.nn,
    "_cuda_kernels": lambda _: False,
    "MODE_DECODE": 0,
    "MODE_VERIFY": 1,
    "MODE_EXTEND": 2,
    "MM_PAD_SHIFT_VALUE": 1 << 20,
}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    *selected,
                ],
                type_ignores=[],
            )
        ),
        str(source),
        "exec",
    ),
    namespace,
)
CanonicalHasher = namespace["EngramHasher"]


def canonical_hasher():
    value = CanonicalHasher.__new__(CanonicalHasher)
    torch.nn.Module.__init__(value)
    value.max_ngram_size = 4
    value.pad_id = 0
    value.image_token_id = 255
    # Non-identity compression and different heads/layers expose indexing errors.
    value.token_map = (torch.arange(256) * 17) % 127
    value.multipliers = torch.tensor([[101, 103, 107, 109], [113, 127, 131, 137]])
    value.primes = torch.arange(3, 51).reshape(2, 3, 8)
    value.offsets = torch.arange(48).reshape(2, 24) * 53
    value.init_history(9, "cpu")
    value.history[:9] = torch.arange(27).reshape(9, 3) + 31
    return value


for width, max_bs in ((2, 4), (4, 2)):
    for bs in range(1, max_bs + 1):
        for prefix_length in (0, 1, 2, 61_440):
            for use_image in (False, True):
                current = batch(bs, width=width)
                current.req_pool_indices = torch.tensor([6, 1, 4, 2][:bs])
                current.input_ids = (torch.arange(bs * width) + 91) % 251
                current.positions = (
                    (torch.arange(width)[None, :] + prefix_length)
                    .expand(bs, -1)
                    .reshape(-1)
                )
                if use_image:
                    current.input_ids[0] = (1 << 20) + 7
                hasher = canonical_hasher()
                before = hasher.history.clone()
                ids = hasher(current.input_ids, current)
                assert torch.equal(hasher.history, before)  # verify is read-only
                # Compare each verify row with a sequential one-token decode of
                # the same candidate chain. This catches request/position mixes.
                sequential = copy.deepcopy(hasher)
                rows = []
                for i, slot in enumerate(current.req_pool_indices):
                    for j in range(width):
                        one = batch(1)
                        one.req_pool_indices = slot.reshape(1)
                        one.input_ids = current.input_ids[
                            i * width + j : i * width + j + 1
                        ]
                        one.positions = current.positions[
                            i * width + j : i * width + j + 1
                        ]
                        rows.append(sequential(one.input_ids, one))
                assert torch.equal(ids, torch.cat(rows))
                # Run canonical hashing through prepare/install/retire and the
                # real CPU byte copies; none may commit unaccepted candidates.
                fixture = model()
                fixture.engram_hasher = hasher
                provider = Prestage(fixture, max_bs, "cpu", req_width=width)
                graph_bs = bucket_for(bs, provider.capture_bs)
                generation = provider.prepare(current, graph_bs)
                provider.install(current, generation)
                provider.retire_after_replay(generation)
                assert torch.equal(hasher.history, before)
                for layer, store in enumerate(provider.stores):
                    assert torch.equal(store.calls[-1], ids[:, layer].reshape(-1))
                assert not provider.gpu_rows[bs * width : graph_bs * width].any()
                # Every accepted length includes the anchor. The bonus remains
                # outside this block. Slots outside the batch must not change.
                verify = current.input_ids.reshape(bs, width)
                for accepted in range(1, width + 1):
                    hasher.history.copy_(before)
                    lens = (torch.arange(bs) + accepted - 1) % width + 1
                    hasher.commit_after_verify(verify, current.req_pool_indices, lens)
                    expected = before.clone()
                    for i, slot in enumerate(current.req_pool_indices):
                        # Raw multimodal IDs remain in stored history until hash normalization.
                        expected[slot] = torch.cat(
                            (before[slot], verify[i, : lens[i]])
                        )[-3:]
                    assert torch.equal(hasher.history, expected)
assert not torch.cuda.is_initialized()
print(
    "PASS: fixed verify widths 2/4, request/token geometry, zero padding, generation ownership; canonical verify hashes equal sequential decode; history unchanged until accepted-length commit; image boundaries and 61K positions; CUDA uninitialized"
)
