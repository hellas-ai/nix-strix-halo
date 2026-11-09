"""CPU-only response and evidence gate tests; no network or fleet access."""
import importlib.util
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location("c8_client", Path(__file__).with_name("client.py"))
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def stream(events, done=True):
    lines = [b": heartbeat\n", b"\n"]
    lines += [b"data: " + json.dumps(e).encode() + b"\n" for e in events]
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


def parse(lines):
    now = time.monotonic()
    return client.parse_stream(lines, now, now + 30)


USAGE = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


class StreamTests(unittest.TestCase):
    def test_interleaved_reasoning_usage_and_fragmented_tool_arguments(self):
        result = parse(stream([
            {"choices": [{"delta": {"reasoning_content": "check"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_marker", "arguments": '{"name":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"marker.txt"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": USAGE}]))
        call = result["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"name": "marker.txt"})
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["usage"], USAGE)
        self.assertGreaterEqual(result["first_generated_s"], 0)

    def test_content_and_length_cap_are_retained(self):
        value = parse(stream([{"choices": [{"delta": {"content": "123"}, "finish_reason": "length"}], "usage": USAGE}]))
        self.assertEqual(value["message"]["content"], "123")
        self.assertEqual(value["finish_reason"], "length")

    def test_error_even_after_finish_is_failure(self):
        with self.assertRaisesRegex(client.ProbeError, "unavailable"):
            parse(stream([{"choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}]}, {"error": {"message": "unavailable"}}]))

    def test_eof_without_done_fails(self):
        with self.assertRaisesRegex(client.ProbeError, "Incomplete"):
            parse(stream([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}], "usage": USAGE}], False))

    def test_done_without_finish_fails(self):
        with self.assertRaises(client.ProbeError):
            parse(stream([{"choices": [{"delta": {"content": "ok"}}], "usage": USAGE}]))

    def test_absent_invalid_and_zero_usage_fail(self):
        for usage in (None, {}, {**USAGE, "completion_tokens": True}, {**USAGE, "completion_tokens": 0}, {**USAGE, "total_tokens": -1}):
            with self.subTest(usage=usage), self.assertRaises(client.ProbeError):
                parse(stream([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}], "usage": usage}]))

    def test_expired_deadline_and_invalid_json_fail(self):
        with self.assertRaisesRegex(client.ProbeError, "deadline"):
            client.parse_stream([b"data: [DONE]"], 0, 0)
        with self.assertRaises(json.JSONDecodeError):
            parse([b"data: broken"])

    def test_http_wrapper_saves_returned_usage_and_error(self):
        probe = client.Client("http://fixture/v1", 30, "fake-token")
        wire = b"".join(stream([{"choices": [{"delta": {"content": "3973"}, "finish_reason": "stop"}], "usage": USAGE}]))
        with patch.object(client.urllib.request, "urlopen", return_value=io.BytesIO(wire)) as opener:
            result = probe.chat(client.body("fixture", "generated fixture", 128), "request")
        self.assertTrue(result["ok"])
        self.assertEqual(result["usage"], USAGE)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://fixture/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-token")
        self.assertNotIn("fake-token", json.dumps(result))
        failure = urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, io.BytesIO(b"private error page"))
        with patch.object(client.urllib.request, "urlopen", side_effect=failure):
            result = probe.chat(client.body("fixture", "generated fixture", 128), "failure")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "HTTP 503")
        self.assertNotIn("private", json.dumps(result))


class EvidenceTests(unittest.TestCase):
    def test_rank_counts_are_not_added(self):
        self.assertEqual(client.running_count({"loads": [{"num_running_reqs": 4}] * 4}), 4)
        self.assertIsNone(client.running_count({"loads": [{"num_running_reqs": "8"}]}))
        self.assertIsNone(client.running_count({"max_running_requests": 8}))

    def test_metrics_labels_duplicates_and_bad_values(self):
        self.assertEqual(client.metrics_running('sglang:num_running_reqs{model_name="glm"} 8\nsglang:num_running_reqs{rank="1"} 8\n'), 8)
        self.assertIsNone(client.metrics_running("# HELP num_running_reqs\nother_metric 8"))
        with self.assertRaises(client.ProbeError):
            client.metrics_running("sglang:num_running_reqs -1")

    def test_client_concurrency_alone_cannot_pass(self):
        rows = [{"ok": True, "usage": USAGE, "start_monotonic": 1, "wall_s": 4}] * 8
        self.assertFalse(client.summarize(rows, [], 8)["worker_target_observed"])
        samples = [{"monotonic": 1 + i * .2, "running": 8} for i in range(8)]
        result = client.summarize(rows, samples, 8)
        self.assertTrue(result["worker_target_observed"])
        self.assertEqual(result["returned_completion_tokens"], 32)
        self.assertEqual(result["aggregate_returned_tokens_per_wall_second"], 8)
        self.assertFalse(client.summarize(rows, [{"monotonic": t, "running": 8} for t in (1, 2.5, 4)], 8)["worker_target_observed"])

    def test_errors_and_missing_samples_do_not_inflate_tokens(self):
        rows = [{"ok": True, "usage": USAGE, "start_monotonic": 1, "wall_s": 2},
                {"ok": False, "start_monotonic": 1, "wall_s": 4, "error": "HTTP500"}]
        result = client.summarize(rows, [], 8)
        self.assertEqual(result["returned_completion_tokens"], 4)
        self.assertEqual(result["successful_requests"], 1)
        self.assertEqual(result["aggregate_returned_tokens_per_wall_second"], 1)

    def test_configuration_whitelist(self):
        result = client.selected_config({"server_args": {"model_path": "/models/fixture", "api_key": "secret", "model_revision": "abc"}, "internal_states": [{"max_running_requests": 8}], "request_text": "private"})
        self.assertEqual(result, {"server_args": {"model_path": "/models/fixture", "model_revision": "abc"}, "internal_states": [{"max_running_requests": 8}]})
        self.assertEqual(client.config_values({"a": {"speculative_algorithm": None}}, {"speculative_algorithm"}), [None])
        self.assertEqual(client.config_values({}, {"speculative_algorithm"}), [])


if __name__ == "__main__":
    unittest.main()
