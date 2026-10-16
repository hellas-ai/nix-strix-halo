"""CPU checks for the default-off per-chunk scheduler timing record (patch 0080).

The recorder is plain Python, so it runs here against fake batches and a fake
clock. The scheduler hooks are checked by executing the real
``_should_defer_prefill`` and by looking for each hook site in the source.
"""

import ast
import importlib.util
import json
import sys
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(sys.argv.pop(1))
SRT = ROOT / "sglang/srt"
if not SRT.exists():
    SRT = next(ROOT.glob("lib/python*/site-packages/sglang/srt"))
RECORDER = SRT / "managers/scheduler_components/timing_record.py"
SCHEDULER = SRT / "managers/scheduler.py"

spec = importlib.util.spec_from_file_location("timing_record", RECORDER)
timing_record = importlib.util.module_from_spec(spec)
sys.modules["timing_record"] = timing_record
spec.loader.exec_module(timing_record)
Recorder = timing_record.SchedulingTimingRecorder


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"

    def is_mixed(self):
        return self.name == "MIXED"

    def is_extend_without_speculative(self):
        return self.name in ("EXTEND", "MIXED")


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_req(rid, input_len, cached=0, queue_entry=0.0, first_forward=0.0, recv=0.0):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[0] * input_len,
        cached_tokens=cached,
        time_stats=SimpleNamespace(
            scheduler_recv_time=recv,
            wait_queue_entry_time=queue_entry,
            forward_entry_time=first_forward,
        ),
    )


def make_batch(mode, reqs, prefix, extend, launch_mono, decoding=None, iteration=1):
    return SimpleNamespace(
        forward_mode=Mode(mode),
        reqs=list(reqs),
        prefix_lens=list(prefix),
        extend_lens=list(extend),
        extend_num_tokens=sum(extend),
        launch_ts=launch_mono,
        forward_iter=iteration,
        decoding_reqs=decoding,
    )


class Fixture:
    def __init__(self):
        self.lines = []
        self.clock = FakeClock()
        # monotonic() is the same clock plus a fixed offset: launch_ts must be
        # re-based into the recorder's clock.
        self.mono_offset = 5.0
        self.recorder = Recorder(
            emit=self.lines.append,
            clock=self.clock,
            monotonic=lambda: self.clock() + self.mono_offset,
            wall=lambda: 1.7e9 + self.clock(),
        )

    def mono(self):
        return self.clock() + self.mono_offset

    def records(self):
        return [
            json.loads(line[len(timing_record.LINE_PREFIX):])
            for line in self.lines
            if line.startswith(timing_record.LINE_PREFIX)
        ]


class RecorderTest(unittest.TestCase):
    def test_default_off_and_rank_gate(self):
        self.assertIsNone(Recorder.from_env(0, environ={}))
        self.assertIsNone(Recorder.from_env(0, environ={timing_record.ENV_FLAG: "0"}))
        self.assertIsNone(Recorder.from_env(1, environ={timing_record.ENV_FLAG: "1"}))
        on = Recorder.from_env(0, 8, 1536, environ={timing_record.ENV_FLAG: "1"})
        self.assertIsNotNone(on)

    def test_init_lines_are_tagged(self):
        fx = Fixture()
        self.assertEqual(len(fx.lines), 1)
        self.assertTrue(fx.lines[0].startswith(timing_record.INIT_PREFIX))
        init = json.loads(fx.lines[0][len(timing_record.INIT_PREFIX):])
        self.assertEqual(init["clock"], "perf_counter")
        self.assertAlmostEqual(init["mono_minus_clock"], fx.mono_offset, places=6)

    def test_two_chunk_request_split(self):
        fx = Fixture()
        r = fx.recorder
        clock = fx.clock
        req = make_req("a", 3000, cached=1000, recv=999.5)
        req.time_stats.wait_queue_entry_time = clock()
        r.on_enqueue(req)

        # Another request's decode runs while ours waits: 2 deferred passes,
        # two decode passes of 50 ms each, one empty planning pass of 20 ms.
        for _ in range(2):
            self.assertIsNone(r.on_defer(True, True))
            launch = fx.mono()
            clock.advance(0.05)
            r.on_result(make_batch("DECODE", [], [], [], launch), clock(), clock())
        t0 = clock()
        clock.advance(0.02)
        r.on_plan(None, t0, clock())
        r.on_defer(False, True)

        # Plan the first chunk: 30 ms of planning, then a 1.5 s forward.
        t0 = clock()
        clock.advance(0.03)
        batch = make_batch("EXTEND", [req], [1000], [1536], 0.0, iteration=7)
        r.on_plan(batch, t0, clock())
        req.time_stats.forward_entry_time = clock()
        clock.advance(0.01)  # run_batch entry before launch
        batch.launch_ts = fx.mono()
        clock.advance(1.5)
        start = clock()
        r.on_result(batch, start, clock(), waiting=0, running=1)

        first = fx.records()[0]
        self.assertEqual(first["kind"], "EXTEND")
        self.assertEqual(first["M"], 1536)
        self.assertEqual(first["iter"], 7)
        self.assertAlmostEqual(first["wall_s"], 1.5, places=5)
        self.assertAlmostEqual(first["plan_s"], 0.03, places=5)
        self.assertAlmostEqual(first["launch"], batch.launch_ts - fx.mono_offset, places=5)
        entry = first["reqs"][0]
        self.assertEqual(entry["rid"], "a")
        self.assertEqual((entry["chunk"], entry["last"]), (0, False))
        self.assertEqual((entry["prefix"], entry["ext"], entry["pending"]), (1000, 1536, 464))
        self.assertEqual(entry["cached"], 1000)
        self.assertEqual(entry["ref"], "queue")
        self.assertEqual(entry["defer_since_ref"], 2)
        self.assertAlmostEqual(entry["decode_s_since_ref"], 0.10, places=5)
        self.assertAlmostEqual(entry["plan_s_since_ref"], 0.05, places=5)
        self.assertAlmostEqual(entry["extend_s_since_ref"], 0.0, places=6)
        self.assertEqual(first["cum"]["decode_passes"], 2)
        self.assertEqual(first["cum"]["plan_empty"], 1)
        self.assertEqual(first["cum"]["defer_waiting"], 2)
        self.assertEqual(entry["recv"], 999.5)
        self.assertIsNotNone(entry["first_forward"])

        # Second chunk: 3 deferred passes with one 40 ms decode pass between.
        for _ in range(3):
            r.on_defer(True, True)
        launch = fx.mono()
        clock.advance(0.04)
        r.on_result(make_batch("DECODE", [], [], [], launch), clock(), clock())
        t0 = clock()
        clock.advance(0.01)
        batch2 = make_batch("EXTEND", [req], [2536], [464], 0.0, iteration=9)
        r.on_plan(batch2, t0, clock())
        batch2.launch_ts = fx.mono()
        clock.advance(0.7)
        r.on_result(batch2, clock(), clock())

        second = fx.records()[1]
        entry = second["reqs"][0]
        self.assertEqual((entry["chunk"], entry["last"], entry["pending"]), (1, True, 0))
        self.assertEqual(entry["ref"], "prev_chunk")
        self.assertEqual(entry["defer_since_ref"], 3)
        self.assertAlmostEqual(entry["decode_s_since_ref"], 0.04, places=5)
        self.assertAlmostEqual(entry["extend_s_since_ref"], 0.0, places=6)
        self.assertGreater(second["n"], first["n"])
        self.assertAlmostEqual(second["cum"]["decode_s"], 0.14, places=5)
        self.assertAlmostEqual(second["cum"]["extend_s"], 2.2, places=5)
        # The request is forgotten after its last chunk.
        self.assertNotIn("a", r._ref)

    def test_other_request_chunk_counts_as_extend_wall(self):
        fx = Fixture()
        r = fx.recorder
        a = make_req("a", 3000)
        b = make_req("b", 1700)
        r.on_enqueue(a)
        r.on_enqueue(b)
        batch_a = make_batch("EXTEND", [a], [0], [1536], 0.0)
        t0 = fx.clock()
        fx.clock.advance(0.01)
        r.on_plan(batch_a, t0, fx.clock())
        batch_a.launch_ts = fx.mono()
        fx.clock.advance(2.0)
        r.on_result(batch_a, fx.clock(), fx.clock())
        batch_b = make_batch("EXTEND", [b], [0], [1536], 0.0)
        t0 = fx.clock()
        fx.clock.advance(0.01)
        r.on_plan(batch_b, t0, fx.clock())
        batch_b.launch_ts = fx.mono()
        fx.clock.advance(2.0)
        r.on_result(batch_b, fx.clock(), fx.clock())
        rec_b = fx.records()[1]["reqs"][0]
        # b queued behind a's 2 s chunk: it shows up as other-request extend wall.
        self.assertAlmostEqual(rec_b["extend_s_since_ref"], 2.0, places=5)
        self.assertEqual((rec_b["chunk"], rec_b["last"], rec_b["pending"]), (0, False, 164))

    def test_decode_pass_logs_nothing_and_tracks_max(self):
        fx = Fixture()
        r = fx.recorder
        before = len(fx.lines)
        for wall in (0.05, 0.494, 0.06):
            launch = fx.mono()
            fx.clock.advance(wall)
            r.on_result(make_batch("DECODE", [], [], [], launch), fx.clock(), fx.clock())
        self.assertEqual(len(fx.lines), before)
        self.assertAlmostEqual(r.decode_pass_max_s, 0.494, places=5)
        # A verify pass is decode work too.
        launch = fx.mono()
        fx.clock.advance(0.1)
        r.on_result(make_batch("TARGET_VERIFY", [], [], [], launch), fx.clock(), fx.clock())
        self.assertEqual(r.decode_passes, 4)
        # The maximum is reported in the next prefill record, then reset.
        req = make_req("a", 100)
        r.on_enqueue(req)
        batch = make_batch("EXTEND", [req], [0], [100], 0.0)
        r.on_plan(batch, fx.clock(), fx.clock())
        batch.launch_ts = fx.mono()
        fx.clock.advance(0.2)
        r.on_result(batch, fx.clock(), fx.clock())
        self.assertAlmostEqual(fx.records()[0]["cum"]["decode_pass_max_s"], 0.494, places=5)
        self.assertEqual(r.decode_pass_max_s, 0.0)

    def test_mixed_is_one_joint_forward(self):
        fx = Fixture()
        r = fx.recorder
        new = make_req("n", 1536)
        running = make_req("d", 500)
        r.on_enqueue(new)
        batch = make_batch(
            "MIXED", [new, running], [0, 499], [1536, 1], 0.0, decoding=[running]
        )
        r.on_plan(batch, fx.clock(), fx.clock())
        batch.launch_ts = fx.mono()
        fx.clock.advance(2.5)
        r.on_result(batch, fx.clock(), fx.clock())
        record = fx.records()[0]
        self.assertEqual(record["kind"], "MIXED")
        self.assertEqual(record["decode_rows"], 1)
        self.assertEqual([q["rid"] for q in record["reqs"]], ["n"])
        self.assertEqual(record["cum"]["decode_passes"], 0)
        self.assertAlmostEqual(record["cum"]["extend_s"], 2.5, places=5)

    def test_idle_and_other_passes_do_not_move_counters(self):
        fx = Fixture()
        r = fx.recorder
        batch = make_batch("IDLE", [], [], [], fx.mono())
        fx.clock.advance(1.0)
        r.on_result(batch, fx.clock(), fx.clock())
        self.assertIsNone(r.prev_end)
        self.assertEqual((r.decode_passes, r.extend_passes), (0, 0))
        self.assertEqual(fx.records(), [])

    def test_health_check_is_flagged(self):
        fx = Fixture()
        req = make_req("HEALTH_CHECK_1", 1)
        fx.recorder.on_enqueue(req)
        batch = make_batch("EXTEND", [req], [0], [1], 0.0)
        fx.recorder.on_plan(batch, fx.clock(), fx.clock())
        batch.launch_ts = fx.mono()
        fx.clock.advance(0.1)
        fx.recorder.on_result(batch, fx.clock(), fx.clock())
        self.assertTrue(fx.records()[0]["reqs"][0]["health"])

    def test_tracked_requests_are_bounded(self):
        fx = Fixture()
        for i in range(timing_record._MAX_TRACKED_REQUESTS + 50):
            fx.recorder.on_enqueue(make_req(f"r{i}", 10))
        self.assertLessEqual(len(fx.recorder._ref), timing_record._MAX_TRACKED_REQUESTS)

    def test_a_bug_in_the_record_does_not_reach_the_scheduler(self):
        fx = Fixture()
        r = fx.recorder
        batch = make_batch("EXTEND", [make_req("a", 100)], [0], [100], fx.mono())
        batch.prefix_lens = None
        batch.reqs[0].origin_input_ids = None  # len(None) raises inside on_result
        fx.clock.advance(0.1)
        self.assertIsNone(r.on_result(batch, fx.clock(), fx.clock()))
        self.assertTrue(r.failed)
        lines = len(fx.lines)
        # Disabled: every later hook is a no-op and nothing more is logged.
        r.on_enqueue(make_req("b", 10))
        r.on_defer(True, True)
        r.on_plan(None, 0.0, 1.0)
        good = make_batch("EXTEND", [make_req("c", 10)], [0], [10], fx.mono())
        r.on_result(good, fx.clock(), fx.clock())
        self.assertEqual(len(fx.lines), lines)
        self.assertEqual((r.defer_calls, r.plan_calls), (0, 0))

    def test_real_clocks_agree_on_this_platform(self):
        # Documented assumption of the clock re-basing: launch_ts is monotonic.
        offset = time.monotonic() - time.perf_counter()
        recorder = Recorder(emit=lambda line: None)
        self.assertAlmostEqual(recorder._mono_offset, offset, places=2)


class SchedulerWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCHEDULER.read_text()
        tree = ast.parse(cls.source)
        cls.methods = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        cls.methods[item.name] = item

    def load(self, name):
        module = ast.Module(body=[self.methods[name]], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(SCHEDULER), "exec"), namespace)
        return namespace[name]

    def scheduler(self, remaining, record=None, waiting=True, chunked=False):
        return SimpleNamespace(
            _prefill_decode_interval_remaining=remaining,
            timing_record=record,
            waiting_queue=[object()] if waiting else [],
            chunked_req=object() if chunked else None,
        )

    def test_defer_semantics_unchanged_when_off(self):
        defer = self.load("_should_defer_prefill")
        s = self.scheduler(3)
        self.assertEqual([defer(s) for _ in range(5)], [True, True, True, False, False])
        self.assertEqual(s._prefill_decode_interval_remaining, 0)

    def test_defer_counts_only_when_work_waits(self):
        defer = self.load("_should_defer_prefill")
        fx = Fixture()
        s = self.scheduler(2, record=fx.recorder, waiting=False)
        self.assertEqual([defer(s) for _ in range(3)], [True, True, False])
        self.assertEqual(
            (fx.recorder.defer_calls, fx.recorder.defer_passes, fx.recorder.defer_waiting_passes),
            (3, 2, 0),
        )
        s = self.scheduler(2, record=fx.recorder, waiting=False, chunked=True)
        self.assertEqual([defer(s) for _ in range(3)], [True, True, False])
        self.assertEqual(fx.recorder.defer_waiting_passes, 2)

    def test_hook_sites_exist(self):
        for needle in (
            "SchedulingTimingRecorder.from_env(",
            "self.timing_record.on_enqueue(req)",
            "timing_record.on_plan(",
            "timing_record.on_result(",
            "self.timing_record.on_defer(",
        ):
            self.assertIn(needle, self.source, needle)
        # Disabled path must be a single attribute test: no hook runs unguarded.
        self.assertEqual(self.source.count("timing_record.on_result("), 1)
        process = ast.unparse(self.methods["process_batch_result"])
        self.assertIn("if timing_record is not None", process)
        enqueue = ast.unparse(self.methods["_add_request_to_queue"])
        self.assertIn("if self.timing_record is not None", enqueue)

    def test_flag_is_named_in_runtime(self):
        self.assertIn(timing_record.ENV_FLAG, RECORDER.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
