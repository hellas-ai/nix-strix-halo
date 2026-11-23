#!/usr/bin/env python3
"""Tests for the GLM streamed-response client (stdlib only)."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "glm53-api-check.py"
spec = importlib.util.spec_from_file_location("glm53_api_check", MODULE_PATH)
client = importlib.util.module_from_spec(spec)
sys.modules["glm53_api_check"] = client
spec.loader.exec_module(client)


def data(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj)).encode() + b"\n"


def text(piece: str, finish=None) -> bytes:
    return data({"choices": [{"delta": {"content": piece}, "finish_reason": finish}]})


DONE = b"data: [DONE]\n"


def parse(*lines: bytes):
    return client.parse_stream(list(lines), started=time.monotonic())


class ValidStreamTests(unittest.TestCase):
    def test_comments_blanks_usage_and_done(self):
        message, finish, usage, first = parse(
            b": keepalive\n",
            b"\n",
            b"   \n",
            data({"usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
            text("Hel"),
            text("lo", finish="stop"),
            DONE,
        )
        self.assertEqual(message["content"], "Hello")
        self.assertEqual(finish, "stop")
        self.assertEqual(usage, {"prompt_tokens": 3, "completion_tokens": 2})
        self.assertIsNotNone(first)

    def test_usage_only_event_with_done(self):
        message, finish, usage, first = parse(
            data({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 0}}),
            DONE,
        )
        self.assertEqual(message["content"], "")
        self.assertIsNone(finish)
        self.assertIsNone(first)
        self.assertEqual(usage["prompt_tokens"], 1)

    def test_tool_call_fragments_assembled(self):
        message, _, _, _ = parse(
            data({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_", "function": {"name": "read_", "arguments": "{\"pa"}}]}}]}),
            data({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "1", "function": {"name": "file", "arguments": "th\": \"a\"}"}}]}}]}),
            DONE,
        )
        self.assertEqual(len(message["tool_calls"]), 1)
        call = message["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a"})

    def test_null_error_is_not_an_error(self):
        message, _, _, _ = parse(
            data({"error": None, "choices": [{"delta": {"content": "ok"}}]}),
            DONE,
        )
        self.assertEqual(message["content"], "ok")

    def test_optional_space_after_data_colon(self):
        message, finish, _, _ = parse(
            b'data:{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b"data:[DONE]\n",
        )
        self.assertEqual(message["content"], "ok")
        self.assertEqual(finish, "stop")


class ErrorEventTests(unittest.TestCase):
    def test_non_null_errors_with_optional_space(self):
        for prefix in (b"data:", b"data: "):
            for error in ({"message": "backend failed"}, {}, False, 0):
                with self.subTest(prefix=prefix, error=error):
                    with self.assertRaises(client.StreamError):
                        parse(
                            text("partial", finish="stop"),
                            prefix + json.dumps({"error": error}).encode() + b"\n",
                            DONE,
                        )

    def test_error_after_text_is_rejected(self):
        with self.assertRaises(client.StreamError):
            parse(
                text("partial"),
                data({"error": {"message": "CUDA OOM"}}),
                DONE,
            )

    def test_error_after_tool_fragment_is_rejected(self):
        with self.assertRaises(client.StreamError):
            parse(
                data({"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"name": "read_file"}}]}}]}),
                data({"error": "boom"}),
                DONE,
            )

    def test_error_message_is_bounded_and_useful(self):
        with self.assertRaises(client.StreamError) as ctx:
            parse(data({"error": "x" * 10000}), DONE)
        self.assertIn("x", str(ctx.exception))
        self.assertLessEqual(len(str(ctx.exception)), 300)

    def test_error_dict_message_preserved(self):
        with self.assertRaises(client.StreamError) as ctx:
            parse(data({"error": {"message": "CUDA OOM"}}), DONE)
        self.assertIn("CUDA OOM", str(ctx.exception))


class TruncatedStreamTests(unittest.TestCase):
    def test_eof_before_done_is_rejected(self):
        with self.assertRaises(client.StreamError):
            parse(text("trun"))

    def test_eof_after_stop_is_rejected(self):
        with self.assertRaises(client.StreamError):
            parse(text("done-ish", finish="stop"))

    def test_eof_with_no_events_is_rejected(self):
        with self.assertRaises(client.StreamError):
            parse()


if __name__ == "__main__":
    unittest.main(verbosity=2)
