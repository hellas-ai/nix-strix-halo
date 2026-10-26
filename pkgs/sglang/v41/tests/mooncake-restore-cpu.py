"""Fault-injection tests for unpublished Mooncake restores (CPU tensors only)."""

import ast
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
from typing import NamedTuple
import unittest

import torch


ROOT = Path(sys.argv.pop(1))
SRT = ROOT / "sglang/srt"
if not SRT.exists():
    SRT = next(ROOT.glob("lib/python*/site-packages/sglang/srt"))


def definitions(path, names, namespace):
    source = ast.parse(path.read_text())
    selected = [node for node in source.body if getattr(node, "name", None) in names]
    assert len(selected) == len(names)
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)


Phase = Enum("Phase", "ABORT PREPARE COMMIT")
LinkPhase = Enum("LinkPhase", "LOOKUP LOAD OFFLOAD")
ComponentType = Enum("ComponentType", "FULL SWA")
PoolName = NS(KV="kv", SWA="swa", MAMBA="mamba")
log = logging.getLogger("mooncake-restore-test")
log.addHandler(logging.NullHandler())
log.propagate = False
namespace = dict(
    torch=torch,
    logger=log,
    NamedTuple=NamedTuple,
    PoolName=PoolName,
    ExternalLinkerLoadPhase=Phase,
    LinkerTransferPhase=LinkPhase,
    ComponentType=ComponentType,
    InsertParams=lambda **kw: NS(**kw),
)
definitions(
    SRT / "mem_cache/unified_cache/unified_cache_linker.py",
    {"ExternalCacheHitMarker", "UnifiedCacheLinkerWrapper"},
    namespace,
)
Wrapper = namespace["UnifiedCacheLinkerWrapper"]
Marker = namespace["ExternalCacheHitMarker"]


# Use the real component rollback and mapping hooks, without importing GPU kernels.
def component_hook(filename, class_name):
    tree = ast.parse(
        (SRT / "mem_cache/unified_cache/components" / filename).read_text()
    )
    cls = next(n for n in tree.body if getattr(n, "name", None) == class_name)
    method = next(
        n for n in cls.body if getattr(n, "name", None) == "update_external_linker_load"
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ns = dict(namespace)
    exec(compile(ast.fix_missing_locations(unit), filename, "exec"), ns)
    return ns["update_external_linker_load"]


class Allocator:
    def __init__(self, name, events, fail=False):
        self.name, self.events, self.fail = name, events, fail
        self.live = set()

    def allocate(self, count):
        if self.fail:
            return None
        start = 100 if self.name == "kv" else 200
        slots = torch.arange(start, start + count)
        self.live.update(slots.tolist())
        self.events.append(("allocate", self.name))
        return slots

    def free(self, slots):
        self.events.append(("free", self.name))
        for slot in slots.tolist():
            assert slot in self.live, "double free"
            self.live.remove(slot)


class Full:
    component_type = ComponentType.FULL
    update_external_linker_load = component_hook("full.py", "FullComponent")

    def __init__(self, allocator):
        self.allocator = allocator

    def _full_allocator(self):
        return self.allocator

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase is LinkPhase.LOAD
        slots = self.allocator.allocate(len(keys) * 2)
        return None if slots is None else NS(name="kv", keys=keys, device_indices=slots)


class SWA:
    component_type = ComponentType.SWA
    sliding_window_size = 4
    update_external_linker_load = component_hook("swa.py", "SWAComponent")

    def __init__(self, cache):
        self.cache = cache

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase is LinkPhase.LOAD
        slots = self.cache.token_to_kv_pool_allocator.swa_attn_allocator.allocate(
            len(keys) * 2
        )
        return (
            None if slots is None else NS(name="swa", keys=keys, device_indices=slots)
        )


class RequestKV:
    def __init__(self):
        self.component_evicted_seqlens = {ComponentType.SWA: 1}

    def get_evicted_seqlen(self, component):
        return self.component_evicted_seqlens.get(component, 0)

    def set_evicted_seqlen(self, component, value):
        self.component_evicted_seqlens[component] = value


def fixture(*, restored=True, reduce=None, fail_alloc=None):
    events, mappings = [], []
    full, swa = (Allocator(name, events, fail_alloc == name) for name in ("kv", "swa"))
    cache = NS(
        page_size=2,
        token_to_kv_pool_allocator=NS(swa_attn_allocator=swa),
        _all_reduce_attn_groups=reduce or (lambda state, op: None),
    )
    cache.token_to_kv_pool_allocator.set_full_to_swa_mapping = lambda a, b: (
        mappings.append((a.clone(), b.clone()))
    )
    tree = NS(empty_match_result=NS(device_indices=torch.empty(0, dtype=torch.int64)))
    cache.tree_core = tree

    def insert(params):
        events.append(("publish", "tree"))
        tree.tail = params.value[params.prev_prefix_len :].clone()
        return NS(
            last_device_node=9, adopted_ranges={ct: [(4, 8)] for ct in ComponentType}
        )

    cache.insert = insert
    tree.collect_full_device_indices = lambda last, previous: tree.tail
    tree.mark_external_cache_stored_path = lambda last, previous: events.append(
        ("mark_stored", "tree")
    )

    def restore(rid, transfers):
        events.append(("restore_finished", restored))
        return restored

    backend = NS(
        restores_before_publish=True,
        restore=restore,
        cancel_queued_load=lambda rid: False,
        reset=lambda: None,
    )
    wrapper = Wrapper.__new__(Wrapper)
    wrapper.cache, wrapper.cache_linker = cache, backend
    wrapper._components, wrapper._skip_swa = (Full(full), SWA(cache)), False
    wrapper.hit_markers = {"r": Marker(list(range(8)), ["h1", "h2"], 4)}
    wrapper.failed_loads, wrapper.pending_loads, wrapper.pending_offloads = (
        set(),
        {},
        [],
    )
    request = NS(
        rid="r",
        last_node=3,
        prefix_indices=torch.arange(4),
        priority=0,
        kv=RequestKV(),
        host_hit_length=4,
        swa_host_hit_length=4,
        mamba_host_hit_length=0,
    )
    return NS(
        wrapper=wrapper,
        req=request,
        events=events,
        mappings=mappings,
        full=full,
        swa=swa,
    )


class Votes:
    """Real CPU rank rendezvous: no rank can observe a vote before all arrive."""

    def __init__(self, count):
        self.barrier = threading.Barrier(count, timeout=10)
        self.values = [None] * count

    def rank(self, rank):
        def reduce(state, op):
            self.values[rank] = state.item()
            self.barrier.wait()
            state.fill_(min(self.values))
            self.barrier.wait()

        return reduce


class PublicationTests(unittest.TestCase):
    def assert_miss(self, f, result):
        self.assertEqual(result[0].numel(), 0)
        self.assertEqual(result[1], 3)
        self.assertEqual(f.req.prefix_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(f.req.kv.component_evicted_seqlens, {ComponentType.SWA: 1})
        self.assertFalse(f.full.live or f.swa.live or f.mappings)
        self.assertNotIn(("publish", "tree"), f.events)
        self.assertNotIn(("mark_stored", "tree"), f.events)
        self.assertEqual(f.req.host_hit_length, 0)
        self.assertFalse(f.wrapper.pending_loads)
        self.assertIn("r", f.wrapper.failed_loads)

    def test_failure_never_publishes_or_changes_request_boundaries(self):
        f = fixture(restored=False)
        self.assert_miss(f, f.wrapper.load_back(f.req))
        self.assertLess(
            f.events.index(("restore_finished", False)), f.events.index(("free", "swa"))
        )
        self.assertLess(f.events.index(("free", "swa")), f.events.index(("free", "kv")))

    def test_failed_request_skips_remote_lookup_then_releases_marker(self):
        f = fixture(restored=False)
        f.wrapper.load_back(f.req)
        result = object()
        self.assertIs(f.wrapper.match(None, f.req, result), result)
        f.wrapper.release_request("r")
        self.assertFalse(f.wrapper.failed_loads)

    def test_success_publishes_only_after_restore(self):
        f = fixture()
        indices, node = f.wrapper.load_back(f.req)
        self.assertEqual(indices.tolist(), [100, 101, 102, 103])
        self.assertEqual(node, 9)
        self.assertLess(
            f.events.index(("restore_finished", True)),
            f.events.index(("publish", "tree")),
        )
        self.assertEqual(len(f.mappings), 2)  # PREPARE then COMMIT
        self.assertFalse(f.wrapper.pending_loads or f.wrapper.failed_loads)

    def test_any_rank_storage_failure_rolls_back_every_rank(self):
        votes = Votes(4)
        ranks = [
            fixture(restored=(rank != 2), reduce=votes.rank(rank)) for rank in range(4)
        ]
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda f: f.wrapper.load_back(f.req), ranks))
        for f, result in zip(ranks, results):
            self.assert_miss(f, result)

    def test_any_rank_allocation_failure_skips_io_on_every_rank(self):
        for pool in ("kv", "swa"):
            with self.subTest(pool=pool):
                votes = Votes(4)
                ranks = [
                    fixture(
                        reduce=votes.rank(rank), fail_alloc=pool if rank == 1 else None
                    )
                    for rank in range(4)
                ]
                with ThreadPoolExecutor(max_workers=4) as workers:
                    results = list(
                        workers.map(lambda f: f.wrapper.load_back(f.req), ranks)
                    )
                for f, result in zip(ranks, results):
                    self.assert_miss(f, result)
                    self.assertFalse(
                        any(event[0] == "restore_finished" for event in f.events)
                    )

    def test_reset_clears_failed_requests(self):
        f = fixture(restored=False)
        f.wrapper.load_back(f.req)
        f.wrapper.reset()
        self.assertFalse(f.wrapper.failed_loads or f.wrapper.hit_markers)

    def test_failed_slots_can_be_reused_by_a_later_request(self):
        f = fixture(restored=False)
        f.wrapper.load_back(f.req)
        f.wrapper.release_request("r")
        f.wrapper.cache_linker.restore = lambda rid, transfers: True
        f.wrapper.hit_markers["r"] = Marker(list(range(8)), ["h1", "h2"], 4)
        indices, node = f.wrapper.load_back(f.req)
        self.assertEqual(indices.tolist(), [100, 101, 102, 103])
        self.assertEqual(node, 9)
        self.assertFalse(f.wrapper.failed_loads)

    def test_other_backends_retain_their_async_load_contract(self):
        f = fixture()
        f.wrapper.cache_linker.restores_before_publish = False
        f.wrapper.cache.inc_lock_ref = lambda node: NS(to_dec_params=lambda: "lock")
        f.wrapper.cache.dec_lock_ref = lambda node, params: None

        def queue(rid, transfers):
            f.events.append(("queue", rid))
            return True

        f.wrapper.cache_linker.load = queue
        _, node = f.wrapper.load_back(f.req)
        self.assertEqual(node, 9)
        self.assertLess(
            f.events.index(("publish", "tree")), f.events.index(("queue", "r"))
        )
        self.assertEqual(f.wrapper.pending_loads, {"r": (9, "lock")})
        self.assertFalse(any(event[0] == "restore_finished" for event in f.events))


# Exercise the actual backend restore method with completion-returning CPU I/O.
device = NS(current_stream=lambda: NS(synchronize=lambda: None))
backend_ns = dict(
    torch=torch, logger=log, UnifiedCacheLinker=object, device_module=device
)
definitions(
    SRT / "mem_cache/storage/mooncake_store/mooncake_direct_linker.py",
    {"MooncakeDirectLinker"},
    backend_ns,
)
Mooncake = backend_ns["MooncakeDirectLinker"]


def backend(results):
    instance = Mooncake.__new__(Mooncake)
    instance.stats = {"load": 0}
    transfers = [
        NS(name=name, keys=[name], host_indices=torch.arange(2))
        for name in ("kv", "swa")
    ]
    instance.pool_group = NS(resolve_transfers=lambda _: transfers)
    instance.pools = {
        name: NS(get_page_buffer_meta=lambda _: ([100, 200], [8, 8]))
        for name in ("kv", "swa")
    }
    calls = []

    def get(keys, pointers, sizes):
        calls.append(keys[0])
        result = results[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    instance.storage = NS(
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
        _pack_multi_buffer_meta=lambda keys, pointers, sizes: ([pointers], [sizes]),
        _get_batch_zero_copy_impl=get,
    )
    return instance, calls


class BackendTests(unittest.TestCase):
    def test_exact_byte_counts_all_pools(self):
        b, calls = backend([[16], [16]])
        self.assertTrue(b.restore("r", []))
        self.assertEqual(calls, ["kv", "swa"])
        self.assertEqual(b.stats["load"], 1)

    def test_partial_restore_and_all_error_forms_are_misses(self):
        for result in (
            [-600],
            [8],
            [32],
            [],
            None,
            -600,
            RuntimeError("transport failed"),
        ):
            with self.subTest(result=result):
                b, calls = backend([[16], result])
                self.assertFalse(b.restore("r", []))
                self.assertEqual(calls, ["kv", "swa"])
                self.assertEqual(b.stats["load"], 0)

    def test_device_error_is_not_swallowed_as_storage_miss(self):
        original = device.current_stream

        def sync_error():
            raise RuntimeError("device failure")

        device.current_stream = lambda: NS(synchronize=sync_error)
        try:
            b, calls = backend([[16], [16]])
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                b.restore("r", [])
            self.assertEqual(calls, [])
        finally:
            device.current_stream = original


if __name__ == "__main__":
    unittest.main()
