"""CPU reproduction of the packed SWA collision and cache-identity regressions."""

import ast
from collections.abc import Callable, Sequence
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys
import unittest
from unittest.mock import patch

import torch


ROOT = Path(sys.argv.pop(1))
SRT = ROOT / "sglang/srt"
if not SRT.exists():
    SRT = next(ROOT.glob("lib/python*/site-packages/sglang/srt"))


def definitions(path, names, namespace):
    source = ast.parse(path.read_text())
    selected = [item for item in source.body if getattr(item, "name", None) in names]
    assert len(selected) == len(names), names
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


ns = dict(torch=torch, Callable=Callable, Sequence=Sequence, Any=Any, replace=replace)
definitions(
    SRT / "mem_cache/hybrid_cache/linker_pool_assembler.py",
    {"DevicePoolEntry", "DevicePoolGroup", "_with_packed_draft_mapping"},
    ns,
)
spec = importlib.util.spec_from_file_location(
    "cache_namespace", SRT / "mem_cache/storage/mooncake_store/cache_namespace.py"
)
namespace_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(namespace_module)
cache_namespace = namespace_module.direct_cache_namespace


def group(draft_layers=0, capacity=2, page_bytes=135168):
    buffers = [
        torch.empty((capacity, page_bytes), dtype=torch.uint8)
        for _ in range(40 + draft_layers)
    ]
    mapping = ns["_with_packed_draft_mapping"](
        {i: i for i in range(40)},
        target_device_layer_num=40,
        draft_layer_num=draft_layers,
    )
    entry = ns["DevicePoolEntry"](
        name="swa",
        indices_from_pool="swa",
        device_pool=None,
        components=[buffers],
        layer_mapping=mapping,
        page_size=256,
        rows_are_pages=True,
    )
    return ns["DevicePoolGroup"]([entry], 40, 256, rank_replicated=True)


def identity(pool_group, numerics="runtime-A", **options):
    return cache_namespace(
        pool_group, numerics_id=numerics, server_args=SimpleNamespace(**options)
    )


class CacheNamespaceTests(unittest.TestCase):
    def test_exact_failed_range_and_registered_destinations(self):
        production, candidate = group(), group(3)
        prod_pool, candidate_pool = production.entries[0], candidate.entries[0]
        _, prod_sizes = prod_pool.get_page_buffer_meta(torch.arange(256))
        production_object_bytes = sum(prod_sizes)
        ptrs, sizes, offsets = candidate_pool.get_prepared_layer_range_meta([0], 0)
        self.assertEqual(production_object_bytes, 5406720)
        self.assertEqual(sizes, [[135168, 135168]])
        self.assertEqual(sum(sizes[0]), 270336)
        self.assertEqual(offsets, [[0, 5406720]])
        self.assertGreater(offsets[0][1] + sizes[0][1], production_object_bytes)
        # The requested device destinations are inside the same whole-storage
        # allocations that register_buffers() registers. The invalid range is
        # the remote serialized object, not missing draft destination memory.
        allocations = [
            (buf.untyped_storage().data_ptr(), buf.untyped_storage().nbytes())
            for buf in candidate_pool.get_hybrid_pool_buffer()
        ]
        for pointer, size in zip(ptrs[0], sizes[0]):
            self.assertTrue(
                any(
                    base <= pointer and pointer + size <= base + length
                    for base, length in allocations
                )
            )
        _, candidate_sizes = candidate_pool.get_page_buffer_meta(torch.arange(256))
        self.assertEqual(sum(candidate_sizes), 5812224)

    def test_packed_draft_layout_changes_namespace(self):
        self.assertNotEqual(identity(group()), identity(group(3)))

    def test_addresses_and_capacity_do_not_change_namespace(self):
        self.assertEqual(identity(group(capacity=2)), identity(group(capacity=5)))

    def test_numeric_build_changes_namespace_without_layout_change(self):
        pools = group()
        self.assertNotEqual(identity(pools), identity(pools, numerics="runtime-B"))

    def test_page_bytes_and_layer_mapping_are_part_of_identity(self):
        a, b = group(), group(page_bytes=149760)
        self.assertNotEqual(identity(a), identity(b))
        b = group()
        b.entries[0].layer_mapping[0] = 1
        self.assertNotEqual(identity(a), identity(b))

    def test_gamma_and_kernel_policy_are_part_of_identity(self):
        pools = group(3)
        self.assertNotEqual(
            identity(pools, speculative_num_draft_tokens=2),
            identity(pools, speculative_num_draft_tokens=4),
        )
        with patch.dict("os.environ", {"SGLANG_DSV4_KV_LAYOUT": "v41"}):
            a = identity(pools)
        with patch.dict("os.environ", {"SGLANG_DSV4_KV_LAYOUT": "v4"}):
            self.assertNotEqual(a, identity(pools))

    def test_integrated_kernel_switches_change_namespace(self):
        pools = group()
        for key, a, b in (
            ("SGLANG_DSV41_DSPARK_ADAPTIVE_VERIFY", "0", "1"),
            ("SGLANG_DSV41_NATIVE_RMSNORM_TREE", "match", "invariant"),
            ("SGLANG_DSV41_MXFP4_GATE_UP_ROWS", "live", "wmma"),
            ("SGLANG_DSV41_MXFP4_DECODE_GEMM", "owner", "aiter"),
            ("SGLANG_DSV41_MXFP4_DOWN_CFG", "64,64,4,1", "32,64,4,1"),
        ):
            with self.subTest(key=key):
                with patch.dict("os.environ", {key: a}):
                    first = identity(pools)
                with patch.dict("os.environ", {key: b}):
                    self.assertNotEqual(first, identity(pools))

    def test_swa_bounded_replay_flags_change_namespace(self):
        pools = group()
        for flag in (
            "enable_decoder_swa_bounded_replay",
            "enable_encoder_swa_bounded_replay",
        ):
            with self.subTest(flag=flag):
                self.assertNotEqual(
                    identity(pools, **{flag: False}), identity(pools, **{flag: True})
                )
                self.assertEqual(
                    identity(pools, **{flag: True}), identity(pools, **{flag: True})
                )

    def test_rank_local_network_names_are_excluded(self):
        pools = group()
        with patch.dict(
            "os.environ", {"MOONCAKE_DEVICE": "mlx5_0", "DS41_NODE_RANK": "0"}
        ):
            a = identity(pools)
        with patch.dict(
            "os.environ", {"MOONCAKE_DEVICE": "mlx5_1", "DS41_NODE_RANK": "3"}
        ):
            self.assertEqual(a, identity(pools))

    def test_missing_numeric_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "cache_numerics_id"):
            identity(group(), numerics="")


if __name__ == "__main__":
    unittest.main()
