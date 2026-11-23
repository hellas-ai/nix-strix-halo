#!/usr/bin/env python3
"""Tests for the streaming vLLM benchmark client (stdlib only)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "vllm-stream-client.py"
spec = importlib.util.spec_from_file_location("vllm_stream_client", MODULE_PATH)
client = importlib.util.module_from_spec(spec)
sys.modules["vllm_stream_client"] = client
spec.loader.exec_module(client)


def sse(*events: str) -> io.BytesIO:
    return io.BytesIO("\n".join(events).encode("utf-8"))


def data_event(obj: dict) -> str:
    return "data: " + json.dumps(obj)


def text_event(piece: str) -> str:
    return data_event({"choices": [{"text": piece}]})


def usage_event(completion: int = 2) -> str:
    return data_event(
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": completion}}
    )


def consume(events: str) -> dict:
    return client.consume_stream(
        sse(events), start=time.perf_counter(), request_index=0
    )


def ok_row(
    *,
    prompt_tokens=5,
    completion_tokens=10,
    duration_s=2.0,
    ttft_s=1.0,
    decode_s=1.0,
) -> dict:
    return {
        "request_index": 0,
        "ok": True,
        "start": 0.0,
        "end": duration_s,
        "duration_s": duration_s,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": "",
    }


class StreamSuccessTests(unittest.TestCase):
    def test_full_stream_ok(self):
        row = consume(
            "\n".join(
                [text_event("Hel"), text_event("lo"), usage_event(), "data: [DONE]"]
            )
        )
        self.assertTrue(row["ok"])
        self.assertEqual(row["error"], "")
        self.assertEqual(row["text_bytes"], 5)
        self.assertEqual(row["completion_tokens"], 2)
        self.assertEqual(row["prompt_tokens"], 5)
        self.assertIsNotNone(row["ttft_s"])

    def test_done_without_usage_is_ok(self):
        row = consume("\n".join([text_event("hi"), "data: [DONE]"]))
        self.assertTrue(row["ok"])
        self.assertIsNone(row["completion_tokens"])

    def test_zero_token_response_with_usage_is_ok(self):
        row = consume("\n".join([usage_event(completion=0), "data: [DONE]"]))
        self.assertTrue(row["ok"])
        self.assertEqual(row["completion_tokens"], 0)
        self.assertIsNone(row["ttft_s"])

    def test_comments_and_blank_lines_ignored(self):
        row = consume(
            "\n".join([": keepalive", "", "   ", text_event("a"), "data: [DONE]"])
        )
        self.assertTrue(row["ok"])
        self.assertEqual(row["text_bytes"], 1)


class StreamFailureTests(unittest.TestCase):
    def test_error_event_with_text_is_failure(self):
        row = consume(
            "\n".join(
                [
                    text_event("par"),
                    data_event({"error": {"message": "CUDA OOM"}}),
                    "data: [DONE]",
                ]
            )
        )
        self.assertFalse(row["ok"])
        self.assertIn("CUDA OOM", row["error"])
        self.assertLessEqual(len(row["error"]), 320)

    def test_error_event_without_done_is_failure(self):
        row = consume(text_event("x") + "\n" + data_event({"error": "boom"}))
        self.assertFalse(row["ok"])
        self.assertIn("boom", row["error"])

    def test_eof_without_done_after_text_is_failure(self):
        row = consume(text_event("trun"))
        self.assertFalse(row["ok"])
        self.assertIn("without [DONE]", row["error"])

    def test_eof_without_done_after_finish_reason_is_failure(self):
        row = consume(
            "\n".join(
                [
                    text_event("done-ish"),
                    data_event({"choices": [{"text": "", "finish_reason": "length"}]}),
                ]
            )
        )
        self.assertFalse(row["ok"])
        self.assertIn("without [DONE]", row["error"])

    def test_eof_without_done_empty_stream_is_failure(self):
        row = consume("")
        self.assertFalse(row["ok"])

    def test_error_message_is_bounded(self):
        row = consume("\n".join([data_event({"error": "x" * 10000}), "data: [DONE]"]))
        self.assertFalse(row["ok"])
        self.assertLessEqual(len(row["error"]), 320)

    def test_malformed_json_is_failure(self):
        row = consume("data: {not json\n" + "data: [DONE]")
        self.assertFalse(row["ok"])
        self.assertIn("malformed", row["error"])


class AggregateTests(unittest.TestCase):
    def aggregate_with_rows(self, rows):
        import argparse

        args = argparse.Namespace(
            run_id="r",
            transport="t",
            parallelism="p",
            model="m",
            concurrency=len(rows),
            max_tokens=8,
            prompt="prompt",
            temperature=0.0,
            ignore_eos=True,
            timeout=5.0,
            endpoint="e",
            socket_ifname="",
            rdma_hca="",
            vllm_extra_args="",
            server_log="",
        )
        originals = client.stream_one, client.base_row
        client.stream_one = lambda **kw: rows[kw["request_index"]].copy()
        client.base_row = lambda a, p: {}
        try:
            return client.aggregate(args, "prompt")
        finally:
            client.stream_one, client.base_row = originals

    def test_failed_stream_excluded_from_throughput(self):
        good = {
            "request_index": 0,
            "ok": True,
            "start": 0.0,
            "end": 2.0,
            "duration_s": 2.0,
            "ttft_s": 1.0,
            "decode_s": 1.0,
            "prompt_tokens": 4,
            "completion_tokens": 10,
            "error": "",
        }
        bad = {
            "request_index": 1,
            "ok": False,
            "start": 0.0,
            "end": 0.5,
            "error": "stream error: CUDA OOM",
        }
        row = self.aggregate_with_rows([good, bad])
        self.assertEqual(row["requests_ok"], "1")
        self.assertEqual(row["requests_failed"], "1")
        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["total_completion_tokens"], "10")  # bad row excluded
        self.assertIn("CUDA OOM", row["errors"])

    def test_all_failed(self):
        bad = {
            "request_index": 0,
            "ok": False,
            "start": 0.0,
            "end": 0.1,
            "error": "stream ended without [DONE]",
        }
        row = self.aggregate_with_rows([bad])
        self.assertEqual(row["requests_failed"], "1")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["total_tps"], "0.000000")

    def test_success_without_usage_leaves_aggregates_unavailable(self):
        row = self.aggregate_with_rows(
            [ok_row(prompt_tokens=None, completion_tokens=None)]
        )
        self.assertEqual(row["requests_ok"], "1")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["total_prompt_tokens"], "")
        self.assertEqual(row["prompt_tokens_per_req"], "")
        self.assertEqual(row["total_completion_tokens"], "")
        self.assertEqual(row["total_tps"], "")
        self.assertEqual(row["output_tps_mean"], "")
        self.assertEqual(row["decode_tps_mean"], "")

    def test_one_missing_completion_disables_completion_aggregates(self):
        rows = [
            ok_row(completion_tokens=10, prompt_tokens=5),
            ok_row(completion_tokens=None, prompt_tokens=5),
        ]
        row = self.aggregate_with_rows(rows)
        self.assertEqual(row["total_completion_tokens"], "")
        self.assertEqual(row["total_tps"], "")
        self.assertEqual(row["output_tps_mean"], "")
        self.assertEqual(row["decode_tps_mean"], "")
        # Prompt counts were still fully reported by every successful request.
        self.assertEqual(row["total_prompt_tokens"], "10")

    def test_one_missing_prompt_only_disables_prompt_aggregates(self):
        rows = [
            ok_row(prompt_tokens=5, completion_tokens=10),
            ok_row(prompt_tokens=None, completion_tokens=10),
        ]
        row = self.aggregate_with_rows(rows)
        self.assertEqual(row["total_prompt_tokens"], "")
        self.assertEqual(row["prompt_tokens_per_req"], "")
        self.assertEqual(row["total_completion_tokens"], "20")
        self.assertNotEqual(row["total_tps"], "")
        self.assertEqual(row["output_tps_mean"], "5.000000")

    def test_failed_request_does_not_invalidate_measurements(self):
        good = ok_row(prompt_tokens=5, completion_tokens=10)
        bad = {
            "request_index": 1,
            "ok": False,
            "start": 0.0,
            "end": 0.5,
            "error": "stream error: CUDA OOM",
        }
        row = self.aggregate_with_rows([good, bad])
        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["total_prompt_tokens"], "5")
        self.assertEqual(row["total_completion_tokens"], "10")

    def test_measured_zero_usage_is_not_missing(self):
        row = self.aggregate_with_rows([ok_row(prompt_tokens=0, completion_tokens=0)])
        self.assertEqual(row["total_prompt_tokens"], "0")
        self.assertEqual(float(row["prompt_tokens_per_req"]), 0.0)
        self.assertEqual(row["total_completion_tokens"], "0")
        self.assertEqual(row["total_tps"], "0.000000")
        self.assertEqual(row["output_tps_mean"], "0.000000")

    def test_fully_reported_throughput_preserved(self):
        row = self.aggregate_with_rows(
            [
                ok_row(prompt_tokens=5, completion_tokens=10),
                ok_row(prompt_tokens=5, completion_tokens=10),
            ]
        )
        self.assertEqual(row["total_prompt_tokens"], "10")
        self.assertEqual(row["prompt_tokens_per_req"], "5.000000")
        self.assertEqual(row["total_completion_tokens"], "20")
        self.assertGreater(float(row["total_tps"]), 0.0)
        self.assertEqual(row["output_tps_mean"], "5.000000")


class MeasuredTokensTests(unittest.TestCase):
    def test_accepts_nonnegative_ints(self):
        self.assertEqual(client.measured_tokens(0), 0)
        self.assertEqual(client.measured_tokens(7), 7)

    def test_rejects_bool(self):
        self.assertIsNone(client.measured_tokens(True))
        self.assertIsNone(client.measured_tokens(False))

    def test_rejects_invalid_counts(self):
        self.assertIsNone(client.measured_tokens(None))
        self.assertIsNone(client.measured_tokens(-1))
        self.assertIsNone(client.measured_tokens(1.0))
        self.assertIsNone(client.measured_tokens("3"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
