"""Synthetic native-format fixture with an independent bit-level decode oracle."""

import json
import math
import os
import struct
import sys
import tempfile
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

runtime = Path(sys.argv[1]).resolve()
from sglang.srt.layers import engram_store as impl

assert str(Path(impl.__file__).resolve()).startswith(str(runtime) + "/"), impl.__file__


def native_file(path, weights, scales, *, scale_first=False):
    # Hand-written safetensors layout, including deliberately unaligned offsets.
    entries = [("weight", weights, "F8_E4M3"), ("scale", scales, "F8_E8M0")]
    if scale_first:
        entries.reverse()
    header, payload = {}, bytearray()
    for name, data, dtype in entries:
        start = len(payload)
        payload.extend(data.tobytes())
        header[name] = {
            "shape": list(data.shape),
            "dtype": dtype,
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return header, len(raw) + 8


def literal_values(packed):
    # Decode the published E4M3FN/E8M0 bit encodings directly. Do not call a
    # Torch FP8 conversion or any implementation helper for this reference.
    table = []
    for byte in range(256):
        sign = -1.0 if byte & 128 else 1.0
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        if exponent == 15 and mantissa == 7:
            value = math.nan
        elif exponent == 0:
            value = sign * math.ldexp(mantissa, -9)
        else:
            value = sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
        table.append(value)
    weights = np.array(table, dtype=np.float32)[packed[..., :256]]
    scale_table = np.array(
        [math.ldexp(1.0, b - 127) for b in range(255)] + [math.nan], dtype=np.float32
    )
    scales = np.repeat(scale_table[packed[..., 256:]], 32, axis=-1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fp32 = weights * scales
    bits = fp32.view(np.uint32)
    nan = np.isnan(fp32)
    # BF16 round-to-nearest, ties-to-even in integer arithmetic.
    rounded = ((bits.astype(np.uint64) + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(
        np.uint16
    )
    return rounded, nan


def check_values(batch, device="cpu", value=None):
    expected, nan = literal_values(batch.packed)
    if value is None:
        value = batch.dequantize(device)
    got = value.cpu().view(torch.uint16).numpy()
    got_nan = ((got & 0x7F80) == 0x7F80) & ((got & 0x7F) != 0)
    assert np.array_equal(got_nan, nan), (device, "NaN mismatch")
    if not np.array_equal(got[~nan], expected[~nan]):
        wrong = (~nan) & (got != expected)
        at = tuple(np.argwhere(wrong)[0])
        raise AssertionError(
            (device, "BF16 bits", int(wrong.sum()), at, int(got[at]), int(expected[at]))
        )
    return value


def fails(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def check_graph_staging(embedding, literal):
    from sglang.srt.layers.engram_graph_prestage import NativeEngramGraphPrestage
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # Synthetic IDs isolate staging ownership and freshness. Hash arithmetic
    # is unchanged by this path; this fixture does not emulate a tokenizer.
    class Hasher:
        calls = 0

        def __call__(self, input_ids, batch):
            self.calls += 1
            return batch.test_hash_ids

    hasher = Hasher()
    layers = [
        SimpleNamespace(engram=SimpleNamespace(layer_hash_index=i, embed=embedding))
        for i in range(2)
    ]
    provider = NativeEngramGraphPrestage(
        SimpleNamespace(engram_hasher=hasher, layers=layers), 8, "cuda"
    )
    store = embedding.file_store
    previous_limit = store.max_batch_rows
    store.max_batch_rows = 8 * 24

    def batch(ids):
        bs = ids.shape[0]
        return SimpleNamespace(
            batch_size=bs,
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.ones(bs, device="cuda", dtype=torch.int64),
            positions=torch.ones(bs, device="cuda", dtype=torch.int64),
            req_pool_indices=torch.arange(bs, device="cuda"),
            out_cache_loc=torch.arange(1, bs + 1, device="cuda"),
            engram_packed_rows=None,
            test_hash_ids=ids,
        )

    try:
        graphs = {}
        for bs in (1, 2, 4, 8):

            def forward(bs=bs):
                return tuple(
                    embedding(None, prepared_packed_rows=provider.gpu_rows[:bs, i])
                    for i in range(2)
                )

            forward()
            graph = torch.cuda.CUDAGraph()
            with (
                patch.object(
                    store,
                    "lookup",
                    side_effect=AssertionError("file read during capture"),
                ),
                torch.cuda.graph(graph),
            ):
                outputs = forward()
            graphs[bs] = (graph, outputs)
            for shift in (0, 97, 0):
                ids = (np.arange(bs * 48).reshape(bs, 2, 24) * 13 + shift) % len(
                    literal
                )
                current = batch(torch.from_numpy(ids).cuda())
                calls = hasher.calls
                generation = provider.prepare(current, bs)
                assert hasher.calls == calls + 1
                fails(RuntimeError, partial(provider.prepare, current, bs))
                assert hasher.calls == calls + 1  # rejection precedes history commit
                fails(RuntimeError, partial(provider.install, current, generation - 1))
                provider.install(current, generation)
                fails(RuntimeError, partial(provider.install, current, generation))
                fails(
                    RuntimeError, partial(provider.retire_after_replay, generation - 1)
                )
                graph.replay()
                provider.retire_after_replay(generation)
                for i, value in enumerate(outputs):
                    expected = impl.RowBatch(
                        literal[ids[:, i]], np.ones((bs, 24), bool), 0, 0, 0
                    )
                    check_values(expected, "cuda", value=value)
        # Alternate streams and bucket widths without a global synchronization.
        # Clone each result before retiring its generation, so later graph work
        # cannot overwrite the values that the independent row oracle checks.
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        torch.cuda.synchronize()
        retained = []
        for step, bs in enumerate((8, 1, 4, 2, 8)):
            ids = (np.arange(bs * 48).reshape(bs, 2, 24) * 17 + step * 97) % len(
                literal
            )
            with torch.cuda.stream(streams[step % 2]):
                current = batch(torch.from_numpy(ids).cuda())
                generation = provider.prepare(current, bs)
                provider.install(current, generation)
                graph, outputs = graphs[bs]
                graph.replay()
                owned = tuple(value.clone() for value in outputs)
                provider.retire_after_replay(generation)
            retained.append((ids, owned))
        torch.cuda.synchronize()
        for ids, values in retained:
            for i, value in enumerate(values):
                expected = impl.RowBatch(
                    literal[ids[:, i]], np.ones(ids[:, i].shape, bool), 0, 0, 0
                )
                check_values(expected, "cuda", value=value)
        assert len(provider.host_frames) == 2
        calls = hasher.calls
        fails(
            ValueError,
            lambda: provider.prepare(
                batch(torch.zeros((3, 2, 24), dtype=torch.int64, device="cuda")), 3
            ),
        )
        assert hasher.calls == calls
        current = batch(torch.zeros((1, 2, 24), dtype=torch.int64, device="cuda"))
        generation = provider.generation
        with patch.object(
            store, "lookup", side_effect=OSError("synthetic row read failure")
        ):
            fails(OSError, lambda: provider.prepare(current, 1))
        assert provider.generation == generation + 1 and hasher.calls == calls + 1
        assert provider.pending is None and provider.active_generation is None
        # The serving runner terminates on this post-commit error. Do not retry.
        torch.cuda.synchronize()
    finally:
        store.max_batch_rows = previous_limit
    print(
        json.dumps(
            {
                "event": "prepared_graph_rows",
                "c1_c2_c4_c8_ABA": True,
                "cross_stream_bucket_transitions": [8, 1, 4, 2, 8],
                "stale_generation_rejected": True,
                "read_failure_propagated": True,
                "synthetic_hasher": True,
                "actual_collective_test": False,
            }
        ),
        flush=True,
    )


cases = 0
with tempfile.TemporaryDirectory(prefix="engram-exact-") as directory:
    root = Path(directory)
    n = 513
    weights = ((np.arange(n)[:, None] * 37 + np.arange(256)) % 256).astype(np.uint8)
    scales = ((np.arange(n)[:, None] * 8 + np.arange(8)) % 256).astype(np.uint8)
    path = root / "native.safetensors"
    header, base = native_file(path, weights, scales, scale_first=True)
    literal = np.concatenate((weights, scales), axis=1)
    with impl.EngramRowStore.from_safetensors(
        path,
        "weight",
        path,
        "scale",
        row_start=3,
        row_end=n - 2,
        max_batch_rows=1024,
        max_read_bytes=7 * 264,
    ) as store:
        for shape in [(0, 24), (1, 24), (3, 24), (7, 24)]:
            ids = (np.arange(math.prod(shape)).reshape(shape) * 31) % n
            batch = store.lookup(ids)
            owned = (ids >= 3) & (ids < n - 2)
            expected = literal[ids].copy()
            expected[~owned] = 0
            assert np.array_equal(batch.packed, expected)
            assert batch.unique_rows == len(set(ids[owned].reshape(-1).tolist()))
            assert batch.bytes_read == batch.unique_rows * 264
            check_values(batch)
            cases += 1
        ids = np.array([[3, n - 3, 17, 3, 0, n - 1, -1]], dtype=np.int64)
        mask = ids != -1
        batch = store.lookup(ids, mask=mask)
        expected = np.zeros((*ids.shape, 264), dtype=np.uint8)
        expected[0, :4] = literal[ids[0, :4]]
        assert np.array_equal(batch.packed, expected)
        assert batch.unique_rows == 3 and batch.bytes_read == 3 * 264
        snapshot = batch.packed.copy()
        store.lookup(np.array([20, 21]))
        assert np.array_equal(snapshot, batch.packed)
        check_values(batch)
        cases += 1
        # Exhaust all 256 weight bytes and all 256 scale bytes in one lookup.
        check_values(store.lookup(np.arange(3, 3 + 256)))
        cases += 1
        # Deterministically block a read to inspect readiness, bounded queueing,
        # cancellation and snapshot ownership without a timing assumption.
        entered, release = threading.Event(), threading.Event()
        original_read = store._read_run

        def blocked_read(run):
            entered.set()
            assert release.wait(5)
            return original_read(run)

        store._read_run = blocked_read
        with impl.EngramPrefetcher(store) as prefetch:
            mutable_ids = np.array([3, 7, 3])
            pending = prefetch.submit(mutable_ids)
            assert entered.wait(5) and not pending.done()
            mutable_ids[:] = 9
            cancelled = prefetch.submit(np.array([8]))
            fails(RuntimeError, lambda: prefetch.submit(np.array([9])))
            assert cancelled.cancel()
            release.set()
            ready = pending.result(timeout=5)
            assert np.array_equal(ready.packed, literal[[3, 7, 3]])
            failed = prefetch.submit(np.array([n]))
            fails(IndexError, lambda: failed.result(timeout=5))
            later = prefetch.submit(np.array([11, 3])).result(timeout=5)
            assert np.array_equal(later.packed, literal[[11, 3]])
            assert np.array_equal(ready.packed, literal[[3, 7, 3]])
        store._read_run = original_read
        cases += 5
        for bad in [
            np.array([-1]),
            np.array([n]),
            np.array([2**64 - 1], dtype=np.uint64),
        ]:
            fails(IndexError, lambda bad=bad: store.lookup(bad))
            cases += 1
        fails(TypeError, lambda: store.lookup(np.array([1.0])))
        fails(ValueError, lambda: store.lookup(np.zeros(1025, dtype=np.int64)))
        fails(ValueError, lambda: store.lookup(np.array([3]), mask=np.array([1])))
        fails(
            ValueError,
            lambda: store.lookup(np.array([3]), mask=np.array([True, False])),
        )
        cases += 4
        packed = root / "packed.bin"
        store.pack_range(packed, 3, n - 5)
        payload = packed.read_bytes()[4096:]
        assert payload == literal[3 : n - 2].tobytes()
        with impl.EngramRowStore.from_packed(
            packed, max_read_bytes=264 * 7
        ) as repacked:
            got = repacked.lookup(ids, mask=mask)
            assert np.array_equal(got.packed, expected)
            check_values(got)
            ascending = repacked.lookup(np.arange(3, 24))
            assert ascending.read_operations == 3
            assert ascending.bytes_read == 21 * 264
        fails(FileExistsError, lambda: store.pack_range(packed, 3, 1))
        fails(
            ValueError,
            lambda: store.pack_range(
                root / "too-big.bin", 3, 20, max_output_bytes=4096
            ),
        )
        assert not (root / "too-big.bin").exists()
        cases += 4
        with impl.EngramRowStore.from_packed(packed) as truncated:
            with packed.open("r+b") as f:
                f.truncate(4096 + 264)
            fails(OSError, lambda: truncated.lookup(np.array([n - 3])))
        fails(ValueError, lambda: impl.EngramRowStore.from_packed(packed))
        cases += 2
    fails(RuntimeError, lambda: store.lookup(np.array([3])))
    cases += 1
    # Separate files, the opposite tensor order, and an empty ownership range.
    other = root / "other.safetensors"
    native_file(other, weights, scales)
    with impl.EngramRowStore.from_safetensors(path, "weight", other, "scale") as split:
        assert np.array_equal(
            split.lookup(np.array([512, 0, 19, 512])).packed, literal[[512, 0, 19, 512]]
        )
    with impl.EngramRowStore.from_safetensors(
        path, "weight", path, "scale", row_start=3, row_end=3
    ) as empty:
        batch = empty.lookup(np.array([0, 3, n - 1]))
        assert batch.unique_rows == batch.read_operations == batch.bytes_read == 0
        assert not batch.packed.any()
    cases += 2
    # Reject metadata before requesting payload rows.
    for field, value in [
        ("dtype", "BF16"),
        ("shape", [n, 128]),
        ("data_offsets", [0, n * 256 + 1]),
    ]:
        invalid = json.loads(json.dumps(header))
        invalid["weight"][field] = value
        raw = json.dumps(invalid).encode()
        invalid_path = root / "invalid.safetensors"
        invalid_path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (n * 264))
        fails(
            ValueError,
            partial(
                impl.EngramRowStore.from_safetensors,
                invalid_path,
                "weight",
                invalid_path,
                "scale",
            ),
        )
        cases += 1
    print(
        json.dumps(
            {
                "event": "cpu_exact",
                "cases": cases,
                "packed_byte_mismatches": 0,
                "bf16_bit_mismatches": 0,
                "native_checkpoint": False,
            }
        ),
        flush=True,
    )
    gpu_peak = 0
    if os.environ.get("ENGRAM_CPU_ONLY") != "1":
        torch.cuda.set_per_process_memory_fraction(0.03)
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1151")
        with impl.EngramRowStore.from_safetensors(
            path, "weight", path, "scale", row_start=3, row_end=n - 2
        ) as store:
            for ids in [np.arange(3, 259), np.array([[3, 33, n - 3, 3, 0, n - 1]])]:
                check_values(store.lookup(ids), "cuda")
                cases += 1
            # Capture decode only. CPU readiness and fixed-buffer refresh happen
            # before every replay; all copies and replays use the same HIP stream.
            shape = (3, 24)
            host = torch.empty((*shape, 264), dtype=torch.uint8, pin_memory=True)
            staged = torch.empty_like(host, device="cuda")
            host.zero_()
            staged.copy_(host, non_blocking=True)
            for _ in range(3):
                impl.dequantize_packed_rows(staged)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                values = impl.dequantize_packed_rows(staged)
            replay_ids = []
            with impl.EngramPrefetcher(store) as prefetch:
                for step in range(5):
                    ids = (
                        (np.arange(72).reshape(shape) * (17 + step) + 3 + step * 19) % n
                    ).astype(np.int64)
                    ids[0, :3] = [3 + step, 3 + step, -1]
                    active = ids != -1
                    batch = prefetch.submit(ids, mask=active).result(timeout=5)
                    host.copy_(torch.from_numpy(batch.packed))
                    staged.copy_(host, non_blocking=True)
                    graph.replay()
                    torch.cuda.synchronize()
                    check_values(batch, "cuda", value=values)
                    replay_ids.append(int(ids[0, 0]))
                    cases += 1
            assert len(set(replay_ids)) == 5
            print(
                json.dumps(
                    {
                        "event": "graph_exact",
                        "changed_id_replays": 5,
                        "bf16_bit_mismatches": 0,
                    }
                ),
                flush=True,
            )
        torch.cuda.synchronize()
        gpu_peak = torch.cuda.max_memory_allocated()
        assert gpu_peak < 128 * 1024**2
    from unittest.mock import patch

    import safetensors

    device = "cpu" if os.environ.get("ENGRAM_CPU_ONLY") == "1" else "cuda"
    if device == "cpu":
        print(
            json.dumps(
                {
                    "event": "complete",
                    "cases": cases,
                    "maximum_error": 0,
                    "native_checkpoint": False,
                    "embedding_test": False,
                }
            ),
            flush=True,
        )
        raise SystemExit(0)

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.environ import envs
    from sglang.srt.layers.engram import EngramEmbedding
    from sglang.srt.model_loader.loader import DefaultModelLoader
    from sglang.srt.runtime_context import (
        get_context,
        get_model,
        get_parallel,
        reset_context,
    )
    from sglang.srt.server_args import ServerArgs

    assert str(
        Path(sys.modules[EngramEmbedding.__module__].__file__).resolve()
    ).startswith(str(runtime) + "/")
    assert str(
        Path(sys.modules[DefaultModelLoader.__module__].__file__).resolve()
    ).startswith(str(runtime) + "/")
    model_root = root / "model"
    model_root.mkdir()
    weight_key, scale_key = (
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
    )
    model_header = {
        weight_key: {
            "shape": [n, 256],
            "dtype": "F8_E4M3",
            "data_offsets": [0, n * 256],
        },
        scale_key: {
            "shape": [n, 8],
            "dtype": "F8_E8M0",
            "data_offsets": [n * 256, n * 264],
        },
        "sentinel": {
            "shape": [2],
            "dtype": "F32",
            "data_offsets": [n * 264, n * 264 + 8],
        },
    }
    raw = json.dumps(model_header).encode()
    raw = raw.ljust((len(raw) + 7) // 8 * 8, b" ")
    model_path = model_root / "mixed.safetensors"
    model_path.write_bytes(
        struct.pack("<Q", len(raw))
        + raw
        + weights.tobytes()
        + scales.tobytes()
        + struct.pack("<ff", 3.25, -7.5)
    )
    (model_root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: model_path.name for key in model_header}})
    )
    # Install genuine resolved config bags, keeping all backend/architecture facts
    # supplied by the actual CPU/HIP runtime.
    get_context().set_server_args(ServerArgs(model_path=str(model_root)))
    ids_np = ((np.arange(72).reshape(3, 24) * 31) % n).astype(np.int64)
    ids = torch.from_numpy(ids_np).to(device)
    all_rows = impl.RowBatch(
        literal[ids_np], np.ones(ids_np.shape, dtype=bool), 0, 0, 0
    )
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("file"),
    ):
        with get_parallel().override(
            tp_size=1, tp_rank=0, attn_tp_size=1, moe_tp_size=1, attn_tp_rank=0
        ):
            embedding = EngramEmbedding(n, 256, 1)
            try:
                assert embedding.weight.numel() == embedding.scale.numel() == 0
                assert embedding.host_table is None
                embedding.file_store.max_batch_rows = 17
                check_values(all_rows, device, value=embedding(ids))
                changed = torch.flip(ids, [0, 1])
                changed_ref = impl.RowBatch(
                    literal[ids_np[::-1, ::-1]].copy(),
                    np.ones(ids_np.shape, bool),
                    0,
                    0,
                    0,
                )
                check_values(changed_ref, device, value=embedding(changed))
                fails(
                    RuntimeError,
                    lambda: embedding._load_rows(embedding.weight, torch.empty(0)),
                )
                assert embedding(
                    torch.empty((0, 24), dtype=torch.int64, device=device)
                ).shape == (0, 24, 256)
                embedding.finish_load("component fixture")
                check_graph_staging(embedding, literal)
                cases += 4
            finally:
                embedding.close_file_store()
        parts = []
        for rank in range(4):
            with get_parallel().override(
                tp_size=4,
                tp_rank=rank,
                attn_tp_size=4,
                moe_tp_size=4,
                attn_tp_rank=rank,
            ):
                embedding = EngramEmbedding(n, 256, 1)
                try:
                    assert embedding.weight.numel() == embedding.scale.numel() == 0
                    parts.append(embedding._owned_rows(ids))
                finally:
                    embedding.close_file_store()
        # Exercise the same exact integer ownership reconstruction contract as
        # upstream's ROCm all-reduce, without claiming this is an RCCL test.
        combined = (
            torch.stack([value.view(torch.int32) for value in parts])
            .to(torch.int64)
            .sum(0)
            .to(torch.int32)
            .view(torch.bfloat16)
        )
        check_values(all_rows, device, value=combined)
        cases += 1
        loader = DefaultModelLoader(LoadConfig())
        source = DefaultModelLoader.Source(str(model_root), None)
        resolved = DefaultModelLoader.ResolvedSource(
            source, str(model_root), (str(model_path),), True
        )
        touched = []
        real_open = safetensors.safe_open

        class AuditedOpen:
            def __init__(self, *args, **kwargs):
                self.inner = real_open(*args, **kwargs)

            def __enter__(self):
                self.opened = self.inner.__enter__()
                return self

            def __exit__(self, *args):
                return self.inner.__exit__(*args)

            def get_tensor(self, key):
                touched.append(key)
                assert key not in (weight_key, scale_key)
                return self.opened.get_tensor(key)

        with get_model().override(
            weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False
        ):
            with patch.object(safetensors, "safe_open", AuditedOpen):
                remaining = dict(
                    loader._get_weights_iterator(source, resolved_source=resolved)
                )
            assert touched == ["sentinel"]
            assert torch.equal(remaining["sentinel"], torch.tensor([3.25, -7.5]))
            fails(
                ValueError,
                lambda: loader.start_checkpoint_prefetch((resolved,), num_threads=1),
            )
        with get_model().override(weight_loader_disable_mmap=True):
            fails(
                ValueError,
                lambda: loader._get_weights_iterator(source, resolved_source=resolved),
            )
        cases += 3
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_peak = torch.cuda.max_memory_allocated()
        assert gpu_peak < 128 * 1024**2
    reset_context()
    print(
        json.dumps(
            {
                "event": "embedding_exact",
                "tp1_forward": True,
                "tp4_owned_rows": True,
                "actual_collective_test": False,
                "zero_resident_parameters": True,
                "loader_tensors_materialized": touched,
                "maximum_error": 0,
            }
        ),
        flush=True,
    )

    print(
        json.dumps(
            {
                "event": "complete",
                "cases": cases,
                "maximum_error": 0,
                "fixture_bytes": sum(
                    p.stat().st_size for p in root.rglob("*") if p.is_file()
                ),
                "gpu_peak_bytes": gpu_peak,
                "gpu_budget_fraction": 0.03,
                "native_checkpoint": False,
            }
        ),
        flush=True,
    )
