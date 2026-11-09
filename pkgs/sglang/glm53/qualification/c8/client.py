#!/usr/bin/env python3
"""Bounded HTTP C8 acceptance probe. Uses only the Python standard library."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request


class ProbeError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def api_error(value):
    if isinstance(value, dict) and value.get("error") is not None:
        error = value["error"]
        message = error.get("message", "API error") if isinstance(error, dict) else str(error)
        raise ProbeError(str(message)[:300])


def checked_usage(value):
    if not isinstance(value, dict):
        raise ProbeError("Missing returned token usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ProbeError("Invalid returned token count: " + key)
    if value["completion_tokens"] == 0:
        raise ProbeError("No generated tokens")
    return value


def parse_stream(lines, started, deadline):
    content, reasoning, calls = [], [], {}
    usage = finish = first = None
    done = False
    size = 0
    for line in lines:
        if time.monotonic() > deadline:
            raise ProbeError("Request deadline exceeded")
        size += len(line)
        if size > 4 * 1024 * 1024:
            raise ProbeError("Stream exceeds 4 MiB bound")
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        api_error(event)
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ProbeError("Unexpected multiple choices")
            delta = choice.get("delta", {})
            if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                if first is None:
                    first = time.monotonic() - started
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            finish = choice.get("finish_reason") or finish
            for part in delta.get("tool_calls") or []:
                call = calls.setdefault(part["index"], {"id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
                call["id"] += part.get("id") or ""
                for key in ("name", "arguments"):
                    call["function"][key] += part.get("function", {}).get(key) or ""
    if not done or finish is None or first is None:
        raise ProbeError("Incomplete stream: requires generated event, finish reason and [DONE]")
    return {"message": {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning),
                "tool_calls": [calls[k] for k in sorted(calls)]},
            "reasoning_characters": sum(map(len, reasoning)), "finish_reason": finish,
            "usage": checked_usage(usage), "first_generated_s": first}


def running_count(value):
    """Require one scheduler to reach the target; never add rank/replica counts."""
    rows = value.get("loads") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    counts = [row.get("num_running_reqs") for row in rows]
    if any(type(n) is not int or n < 0 for n in counts):
        return None
    return max(counts)


def metrics_running(text):
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r"(?:sglang:|sglang_)?num_running_reqs(?:\{[^}]*\})?\s+([0-9.eE+-]+)(?:\s+\d+)?", line)
        if match:
            n = float(match[1])
            if not math.isfinite(n) or n < 0 or not n.is_integer():
                raise ProbeError("Invalid running-request metric")
            values.append(int(n))
    return max(values) if values else None


CONFIG_KEYS = {"model_path", "model_revision", "revision", "served_model_name", "tp_size",
    "pp_size", "nnodes", "max_running_requests", "context_length", "max_total_tokens",
    "max_total_num_tokens", "cuda_graph_bs", "cuda_graph_bs_decode", "disable_cuda_graph",
    "speculative_algorithm", "speculative_num_steps", "speculative_num_draft_tokens",
    "version", "max_mamba_cache_size", "prefill_decode_interval"}


def selected_config(value):
    """Keep only configuration, without other server fields or request contents."""
    if not isinstance(value, dict):
        return {}
    out = {k: v for k, v in value.items() if k in CONFIG_KEYS}
    for key in ("server_args", "internal_states"):
        child = value.get(key)
        if isinstance(child, dict):
            out[key] = selected_config(child)
        elif isinstance(child, list):
            out[key] = [selected_config(v) for v in child if isinstance(v, dict)]
    return out


def config_values(value, keys):
    if isinstance(value, dict):
        return [v for k, v in value.items() if k in keys] + [r for v in value.values() for r in config_values(v, keys)]
    if isinstance(value, list):
        return [r for v in value for r in config_values(v, keys)]
    return []


class Client:
    def __init__(self, base, timeout, token=None):
        self.base, self.timeout, self.token = base.rstrip("/"), timeout, token

    def open(self, path, body=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        req = urllib.request.Request(self.base + path,
            None if body is None else json.dumps(body).encode(), headers)
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            # Do not persist arbitrary error pages, prompts or credentials.
            raise ProbeError("HTTP " + str(exc.code)) from exc

    def get(self, path, raw=False):
        with self.open(path, timeout=2) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ProbeError("Telemetry response exceeds size bound")
        return data.decode() if raw else json.loads(data)

    def chat(self, body, label, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=15)
        start = time.monotonic()
        row = {"label": label, "start_monotonic": start, "request_sha256": digest(body)}
        try:
            with self.open("/chat/completions", body) as response:
                result = parse_stream(response, start, start + self.timeout)
            row.update(result)
            row["ok"] = True
        except Exception as exc:
            row.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:300])
        row["wall_s"] = time.monotonic() - start
        return row


class Monitor:
    def __init__(self, client):
        self.client, self.samples, self.errors = client, [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.path = None

    def sample(self):
        paths = [self.path] if self.path else ["/v1/loads?include=core", "/get_server_info", "/server_info", "/metrics"]
        for path in paths:
            try:
                value = self.client.get(path, raw=path == "/metrics")
                count = metrics_running(value) if path == "/metrics" else running_count(value)
                if count is not None:
                    self.path = path
                    self.samples.append({"monotonic": time.monotonic(), "running": count, "source": path})
                    return
            except Exception as exc:
                if len(self.errors) < 20:
                    self.errors.append({"source": path, "error": type(exc).__name__})

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(.2)


def summarize(rows, samples, target):
    good = [r for r in rows if r["ok"]]
    tokens = sum(r["usage"]["completion_tokens"] for r in good)
    start = min(r["start_monotonic"] for r in rows)
    end = max(r["start_monotonic"] + r["wall_s"] for r in rows)
    observed = [s for s in samples if start <= s["monotonic"] <= end]
    # Require multiple samples during one uninterrupted >=target interval.
    longest = 0.0
    run_start = None
    run_count = 0
    max_count = 0
    previous = None
    for sample in observed:
        if previous is not None and sample["monotonic"] - previous > 1:
            run_start = None
        previous = sample["monotonic"]
        if sample["running"] >= target:
            if run_start is None:
                run_start, run_count = sample["monotonic"], 0
            run_count += 1
            longest = max(longest, sample["monotonic"] - run_start)
            max_count = max(max_count, run_count)
        else:
            run_start = None
    return {"requests": len(rows), "successful_requests": len(good),
        "returned_completion_tokens": tokens, "wall_s": end - start,
        "aggregate_returned_tokens_per_wall_second": tokens / (end - start),
        "worker_running_peak": max((s["running"] for s in observed), default=None),
        "worker_target_samples_in_longest_count_run": max_count,
        "worker_target_longest_observed_interval_s": longest,
        "worker_target_observed": max_count >= 3 and longest >= 1,
        "measurement_scope": "End-to-end streamed group including prefill; not isolated decode throughput"}


def body(model, prompt, max_tokens):
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "low",
        "stream": True, "stream_options": {"include_usage": True}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="OpenAI-compatible URL including /v1")
    p.add_argument("--worker-url", help="Direct worker root URL for scheduler/config evidence")
    p.add_argument("--model", default="glm-5.3-flash-uncensored")
    p.add_argument("--expected-revision", help="Expected checkpoint revision; read back when server exposes it")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--token-env", default="OPENAI_API_KEY", help="Optional bearer token environment variable")
    args = p.parse_args()
    levels = [int(v) for v in args.concurrency.split(",")]
    if not levels or any(v < 1 or v > 8 for v in levels) or not 1 <= args.repeats <= 4 or not 64 <= args.max_tokens <= 1024 or not 5 <= args.timeout <= 600:
        p.error("Require concurrency1..8, repeats1..4, max-tokens64..1024, timeout5..600")
    args.out.mkdir(parents=True, exist_ok=False)
    client = Client(args.base_url, args.timeout, os.environ.get(args.token_env))
    root = args.worker_url or args.base_url.rstrip("/").removesuffix("/v1")
    worker = Client(root, 2, os.environ.get(args.token_env))
    report = {"schema": "GLM53_C8_HTTP_ACCEPTANCE_V1", "status": "INCOMPLETE", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url, "worker_url": root, "model": args.model,
        "expected_revision": args.expected_revision, "revision_verified": False,
        "concurrency": levels, "repeats": args.repeats, "max_tokens": args.max_tokens,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": [], "protocol": [], "configuration": {}, "telemetry_errors": [],
        "scope": "Protocol, fixture correctness and observed scheduler occupancy; no numerical equivalence or general coding-quality claim"}

    def save():
        path = args.out / "report.json"
        temp = args.out / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(path)

    for endpoint in ("/get_server_info", "/server_info"):
        try:
            raw = worker.get(endpoint)
            report["configuration"][endpoint] = selected_config(raw)
        except Exception as exc:
            report["telemetry_errors"].append({"source": endpoint, "error": type(exc).__name__})
    actual_revisions = [r for r in config_values(report["configuration"], {"revision", "model_revision"}) if isinstance(r, str)]
    report["observed_revisions"] = sorted(set(actual_revisions))
    report["revision_verified"] = bool(args.expected_revision and actual_revisions and all(r == args.expected_revision for r in actual_revisions))
    algorithms = config_values(report["configuration"], {"speculative_algorithm"})
    report["ordinary_decode_configuration_verified"] = bool(algorithms) and all(a is None for a in algorithms)
    save()
    monitor = Monitor(worker)
    monitor.thread.start()
    try:
        for concurrency in levels:
            barrier = threading.Barrier(concurrency)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = []
                for i in range(concurrency):
                    a, b = 137 + i, 29
                    request = body(args.model, f"Calculate {a} * {b}. Reply with only the integer.", 128)
                    futures.append(pool.submit(client.chat, request, f"short-c{concurrency}-q{i}", barrier))
                rows = [f.result() for f in futures]
            for i, row in enumerate(rows):
                row["fixture_correct"] = row["ok"] and row["message"]["content"].strip() == str((137 + i) * 29) and row["finish_reason"] == "stop"
            report["groups"].append({"kind": "short-correctness", "concurrency": concurrency,
                "rows": rows, "summary": summarize(rows, monitor.samples, concurrency)})
            save()
        target = max(levels)
        for repeat in range(args.repeats):
            barrier = threading.Barrier(target)
            with ThreadPoolExecutor(max_workers=target) as pool:
                futures = []
                for i in range(target):
                    lines = (32 if repeat % 2 == 0 else 16 * (i + 1))
                    fixture = "\n".join(f"def fixture_{i}_{j}(x): return x + {j}" for j in range(lines))
                    prompt = fixture + "\nExplain each function in order, one numbered line per function. Then enumerate integers1 through1000 without omissions. Continue until your output limit."
                    request = body(args.model, prompt, args.max_tokens)
                    futures.append(pool.submit(client.chat, request, f"long-r{repeat}-q{i}", barrier))
                rows = [f.result() for f in futures]
            report["groups"].append({"kind": "long-uniform" if repeat % 2 == 0 else "long-mixed", "concurrency": target,
                "rows": rows, "summary": summarize(rows, monitor.samples, target)})
            save()
        tools = [{"type": "function", "function": {"name": "read_marker",
            "description": "Read a generated marker fixture", "parameters": {"type": "object",
            "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}}}]
        request = body(args.model, "Call read_marker with name marker.txt exactly once.", 256)
        request.update(tools=tools, tool_choice={"type": "function", "function": {"name": "read_marker"}})
        row = client.chat(request, "streamed-tool-call")
        row["fixture_correct"] = False
        if row["ok"]:
            calls = row["message"]["tool_calls"]
            try:
                row["fixture_correct"] = len(calls) == 1 and bool(calls[0]["id"]) and calls[0]["function"]["name"] == "read_marker" and json.loads(calls[0]["function"]["arguments"]) == {"name": "marker.txt"} and row["finish_reason"] == "tool_calls"
            except (ValueError, KeyError, TypeError):
                pass
        report["protocol"].append(row)
        if row["fixture_correct"]:
            followup = body(args.model, "", 128)
            followup["messages"] = [{"role": "user", "content": "Read marker.txt, then reply only with the marker."}, row["message"],
                {"role": "tool", "tool_call_id": row["message"]["tool_calls"][0]["id"], "content": "GLM_C8_MARKER_53"}]
            followup["tools"] = tools
            followup["tool_choice"] = "none"
            result = client.chat(followup, "tool-continuation")
            result["fixture_correct"] = result["ok"] and result["message"]["content"].strip() == "GLM_C8_MARKER_53" and result["finish_reason"] == "stop"
            report["protocol"].append(result)
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=10)
        report["worker_samples"] = monitor.samples
        report["telemetry_errors"].extend(monitor.errors)
        save()
    rows = [r for g in report["groups"] for r in g["rows"]] + report["protocol"]
    correct = all(r["ok"] and r.get("fixture_correct", True) for r in rows) and len(report["protocol"]) == 2
    occupancy = all(g["summary"]["worker_target_observed"] for g in report["groups"] if g["kind"].startswith("long"))
    revision_ok = not args.expected_revision or report["revision_verified"]
    report["status"] = "PASS" if correct and occupancy and revision_ok and report["ordinary_decode_configuration_verified"] and not monitor.thread.is_alive() else "FAIL_OR_INCONCLUSIVE"
    report["fixture_and_protocol_pass"] = correct
    report["sustained_worker_occupancy_pass"] = occupancy
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print(json.dumps({"status": report["status"], "report": str(args.out / "report.json")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
