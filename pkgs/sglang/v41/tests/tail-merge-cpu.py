"""CPU check for the variable last prefill chunk (patch 0083, SGLANG_PREFILL_TAIL_MERGE_ROWS).

The real `_select_prefill_admission`, `add_chunked_req` and `_merge_tail` are cut out of schedule_policy.py and run
against stand-in objects, one scheduling pass at a time, so the chunk sequence a request gets is the one the scheduler
would build: the first chunk through admission, every later chunk through the chunked-request path.
"""

import ast
import os
import sys
import textwrap
import types
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(sys.argv.pop(1))
SRT = ROOT / "sglang/srt"
if not SRT.exists():
    SRT = next(ROOT.glob("lib/python*/site-packages/sglang/srt"))
SOURCE = SRT / "managers/schedule_policy.py"

tree = ast.parse(SOURCE.read_text())
top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
adder_class = top["PrefillAdder"]
methods = {n.name: n for n in adder_class.body if isinstance(n, ast.FunctionDef)}

namespace = {
    "Optional": Optional,
    "os": os,
    "dataclass": dataclass,
    "Enum": Enum,
    "auto": auto,
    "CLIP_MAX_NEW_TOKENS": 4096,
    "Req": object,
}


def load(node):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


for name in ("AddReqResult", "_PrefillAdmission", "_read_tail_merge_rows", "tail_merge_fits"):
    load(top[name])
for name in ("ceil_paged_tokens", "_merge_tail", "_select_prefill_admission", "add_chunked_req"):
    node = methods[name]
    load(node)


class Budget:
    """The plain (non-SWA) budget, plus a switch to make the merged forward not fit."""

    def __init__(self, fits=True):
        self.fits = fits

    def available_chunk_tokens(self, chunk_limit):
        return chunk_limit

    def fit_chunk(self, *, extend_input_len, max_new_tokens, chunk_limit):
        if self.fits or chunk_limit <= 1536:
            return chunk_limit
        return 1536


def make_adder(merge_rows, chunk=1536, page=256, fits=True):
    namespace["TAIL_MERGE_ROWS"] = merge_rows
    namespace["get_schedule"] = lambda: SimpleNamespace(schedule_policy="shortest-prefill-first")
    adder = SimpleNamespace(
        page_size=page, rem_chunk_tokens=chunk, rem_input_tokens=16384, can_run_list=[], exact_chunk_fill=False,
        dllm_config=None, kv_shard_granule=0, prefill_delayer_single_pass=None, chunked_req_limit=None,
        memory_budget=Budget(fits), new_chunked_req=None,
    )
    adder._check_prefill_budget = lambda req, **kw: (True, adder.rem_chunk_tokens)
    adder._check_prefill_tile_budget = lambda tokens: None
    adder._swa_new_tokens = lambda req: 0
    adder._kv_shard_reserve_scratch = lambda **kw: True
    adder._mamba_gap_budget_for_req = lambda req: 0
    adder._update_prefill_budget = lambda *a, **kw: None
    for name in ("ceil_paged_tokens", "_merge_tail", "_select_prefill_admission", "add_chunked_req"):
        setattr(adder, name, types.MethodType(namespace[name], adder))
    return adder


class Req:
    def __init__(self, prefix, extend):
        self.prefix_indices = list(range(prefix))
        self.full_untruncated_fill_ids = list(range(prefix + extend))
        self.extend_range = None
        self.retracted_stain = False
        self.sampling_params = SimpleNamespace(max_new_tokens=200)

    def set_extend_range(self, start, end):
        self.extend_range = SimpleNamespace(start=start, end=end, length=end - start)


def chunks(extend, prefix=8192, merge_rows=0, chunk=1536, fits=True):
    """Row counts of the forwards a request with `extend` uncached rows gets."""
    req = Req(prefix, extend)
    rows = []
    chunked = None
    for _ in range(64):
        adder = make_adder(merge_rows, chunk, fits=fits)
        if chunked is not None:
            leftover = adder.add_chunked_req(chunked)
        else:
            admission = adder._select_prefill_admission(
                req, total_tokens=extend + 200, host_hit_length=0, swa_host_hit_length=0,
                truncation_align_size=None, has_chunked_req=False,
            )
            assert not isinstance(admission, namespace["AddReqResult"]), admission
            req.set_extend_range(admission.prefix_len, admission.prefix_len + admission.extend_len)
            leftover = req if admission.is_chunked else None
        rows.append(req.extend_range.length)
        if leftover is None:
            assert req.extend_range.end == len(req.full_untruncated_fill_ids)
            return rows
        req.prefix_indices = list(range(req.extend_range.end))
        chunked = req
    raise AssertionError("request never finished")


class TailMergeTest(unittest.TestCase):
    def test_off_matches_the_unmodified_chunking(self):
        for extend, expect in ((1000, [1000]), (1536, [1536]), (1735, [1536, 199]), (3272, [1536, 1536, 200]),
                               (4608, [1536, 1536, 1536]), (4600, [1536, 1536, 1528])):
            with self.subTest(extend=extend):
                self.assertEqual(chunks(extend), expect)

    def test_short_tail_rides_with_the_last_full_chunk(self):
        for extend, expect in ((1735, [1735]), (3272, [1536, 1736]), (3100, [1536, 1564]), (1536 + 511, [2047]),
                               (1536 + 512, [1536, 512]), (4608 + 100, [1536, 1536, 1636])):
            with self.subTest(extend=extend):
                self.assertEqual(chunks(extend, merge_rows=511), expect)
        self.assertEqual(chunks(1536 + 600, merge_rows=600), [2136])

    def test_rows_are_conserved_and_nothing_exceeds_chunk_plus_merge(self):
        for merge in (0, 128, 511, 600):
            for extend in list(range(1, 2000, 37)) + list(range(2000, 9000, 101)):
                with self.subTest(merge=merge, extend=extend):
                    rows = chunks(extend, merge_rows=merge)
                    self.assertEqual(sum(rows), extend)
                    self.assertTrue(all(r > 0 for r in rows))
                    self.assertLessEqual(max(rows), 1536 + merge)
                    # Every chunk but the last is a full chunk; the last never exceeds chunk + merge.
                    self.assertTrue(all(r == 1536 for r in rows[:-1]))
                    # Merging never adds a forward.
                    self.assertLessEqual(len(rows), len(chunks(extend)))

    def test_memory_that_cannot_hold_the_merged_forward_chunks_as_before(self):
        self.assertEqual(chunks(3272, merge_rows=511, fits=False), chunks(3272))
        self.assertEqual(chunks(1735, merge_rows=511, fits=False), chunks(1735))

    def test_only_the_first_request_of_a_batch_merges(self):
        namespace["TAIL_MERGE_ROWS"] = 511
        adder = make_adder(511)
        adder.can_run_list = [object()]
        req = Req(8192, 1735)
        admission = adder._select_prefill_admission(
            req, total_tokens=1935, host_hit_length=0, swa_host_hit_length=0, truncation_align_size=None,
            has_chunked_req=False,
        )
        # With another request already in the batch the long request is a chunked
        # candidate exactly as before (and the one-chunked-request rule applies).
        self.assertTrue(admission.is_chunked or admission is namespace["AddReqResult"].OTHER)

    def test_a_capped_chunk_keeps_its_cap(self):
        fits = namespace["tail_merge_fits"]
        self.assertTrue(fits(1735, 1536, 511, 1536, True))
        self.assertFalse(fits(1735, 1024, 511, 1536, True))   # memory-capped chunk
        self.assertFalse(fits(1735, 1536, 511, 1536, False))  # second request in the batch
        self.assertFalse(fits(1735, 1536, 0, 1536, True))     # off
        self.assertFalse(fits(1536, 1536, 511, 1536, True))   # nothing past the chunk
        self.assertFalse(fits(1536 + 512, 1536, 511, 1536, True))
        self.assertFalse(fits(1735, 1536, 511, None, True))   # chunked prefill disabled

    def test_environment_parsing(self):
        read = namespace["_read_tail_merge_rows"]
        old = os.environ.pop("SGLANG_PREFILL_TAIL_MERGE_ROWS", None)
        try:
            self.assertEqual(read(), 0)
            for value, expect in (("0", 0), ("511", 511), (" 600 ", 600)):
                os.environ["SGLANG_PREFILL_TAIL_MERGE_ROWS"] = value
                self.assertEqual(read(), expect)
            for bad in ("-1", "1.5", "many"):
                os.environ["SGLANG_PREFILL_TAIL_MERGE_ROWS"] = bad
                with self.assertRaises(ValueError):
                    read()
        finally:
            os.environ.pop("SGLANG_PREFILL_TAIL_MERGE_ROWS", None)
            if old is not None:
                os.environ["SGLANG_PREFILL_TAIL_MERGE_ROWS"] = old

    def test_hooks_are_in_the_source(self):
        text = SOURCE.read_text()
        self.assertIn("TAIL_MERGE_ROWS = _read_tail_merge_rows()", text)
        self.assertEqual(text.count("self._merge_tail("), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
