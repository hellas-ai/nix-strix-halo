"""CPU check: bounded replay refuses prompt logprobs at scheduler admission (patch 0082).

The model forward raises on a request that asks for logprobs of prompt rows the
late layers never compute, and an exception there stops the whole TP group. The
scheduler must turn that request into an abort instead. The guard sits inline in
``Scheduler.handle_generate_request``; this test pulls that statement out of the
real source and runs it against stand-in objects.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(sys.argv.pop(1))
SRT = ROOT / "sglang/srt"
if not SRT.exists():
    SRT = next(ROOT.glob("lib/python*/site-packages/sglang/srt"))
SCHEDULER = SRT / "managers/scheduler.py"


def find_guard():
    tree = ast.parse(SCHEDULER.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "handle_generate_request":
                    for stmt in ast.walk(item):
                        if (
                            isinstance(stmt, ast.If)
                            and "enable_decoder_swa_bounded_replay" in ast.unparse(stmt.test)
                        ):
                            return stmt
    raise AssertionError("bounded replay admission guard not found")


GUARD = find_guard()
module = ast.Module(
    body=[
        ast.FunctionDef(
            name="guard",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=a) for a in ("self", "req", "get_exec")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[GUARD, ast.Return(value=ast.Constant(value="fell-through"))],
            decorator_list=[],
        )
    ],
    type_ignores=[],
)
namespace = {}
exec(compile(ast.fix_missing_locations(module), str(SCHEDULER), "exec"), namespace)
guard = namespace["guard"]


def fake_exec(enabled):
    return lambda: SimpleNamespace(
        features=SimpleNamespace(enable_decoder_swa_bounded_replay=enabled)
    )


def make_req(return_logprob, logprob_start_len, prompt_len=5000):
    req = SimpleNamespace(
        return_logprob=return_logprob,
        logprob_start_len=logprob_start_len,
        origin_input_ids=[0] * prompt_len,
        aborted=None,
    )
    req.set_finish_with_abort = lambda message: setattr(req, "aborted", message)
    return req


def run(enabled, return_logprob, start, prompt_len=5000):
    queued = []
    self = SimpleNamespace(_add_request_to_queue=queued.append)
    req = make_req(return_logprob, start, prompt_len)
    result = guard(self, req, fake_exec(enabled))
    return req, queued, result


class GuardTest(unittest.TestCase):
    def test_prompt_logprobs_are_refused_not_fatal(self):
        for start in (0, 1, 4999):
            with self.subTest(start=start):
                req, queued, result = run(True, True, start)
                self.assertIsNone(result)  # the guard returned
                self.assertIn("bounded-replay", req.aborted)
                self.assertEqual(queued, [req])
                self.assertEqual(req.logprob_start_len, -1)

    def test_output_logprobs_and_plain_requests_pass(self):
        # logprob_start_len is resolved to the prompt length for chat logprobs,
        # and is -1 when the request asks for no logprobs.
        for wants_logprobs, start in ((True, 5000), (True, -1), (False, -1), (False, 0)):
            with self.subTest(return_logprob=wants_logprobs, start=start):
                req, queued, result = run(True, wants_logprobs, start)
                self.assertEqual(result, "fell-through")
                self.assertIsNone(req.aborted)
                self.assertEqual(queued, [])

    def test_off_means_untouched(self):
        req, queued, result = run(False, True, 0)
        self.assertEqual(result, "fell-through")
        self.assertIsNone(req.aborted)
        self.assertEqual(queued, [])

    def test_guard_runs_after_logprob_start_resolution(self):
        text = SCHEDULER.read_text()
        resolved = text.index("req.logprob_start_len = recv_req.logprob_start_len")
        guard_at = text.index("not available with --enable-decoder-swa-bounded-replay")
        self.assertLess(resolved, guard_at)


if __name__ == "__main__":
    unittest.main(verbosity=2)
