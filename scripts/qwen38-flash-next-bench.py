#!/usr/bin/env python3
"""Async TTFT/decode/prefix-cache benchmark for Qwen Flash-Next on SGLang.

The harness deliberately uses one asyncio event loop and direct HTTP/1.1
connections.  It does not put blocking urllib calls in Python threads: that
pattern made earlier concurrent client throughput results wrong by up to 4x.

Each trial flushes the radix cache, sends a concurrent wave of unique cold
prompts, then sends a concurrent wave whose prompts reuse those cold prefixes.
Every run has a fresh nonce, and cache state is checked against
``meta_info.cached_tokens`` whenever the server supplies it.

The append-only JSONL contains run metadata, one record per request, per-trial
summaries, and a final run summary.  ``variant`` and ``block_index`` make files
suitable for externally interleaved A/B/A/B runs; repeated ``trial`` summaries
carry minima as well as medians for min-of-N analysis.

SGLang's own server-side generation throughput remains authoritative.  The
client wave throughput here is a useful async cross-check, not a replacement.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import secrets
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable


SCHEMA = "qwen38-flash-next-bench-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure concurrent cold/warm TTFT, decode, and end-to-end latency "
            "against SGLang; append durable ABAB-ready JSONL."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30801)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="append-only JSONL output path (parent directories are created)",
    )
    parser.add_argument(
        "--variant",
        default="unlabeled",
        help="server/config label, normally A or B for an interleaved campaign",
    )
    parser.add_argument(
        "--block-index",
        type=int,
        default=0,
        help="external ABAB block/order index",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="distinct cache prefixes (and cold requests) per trial",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=1,
        help="warm requests with distinct tails per seeded prefix",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=16384)
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--token-id-low", type=int, default=1000)
    parser.add_argument("--token-id-high", type=int, default=100000)
    parser.add_argument(
        "--nonce",
        help="reproduce prompt construction; omit during measurement for fresh prompts",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--flush-timeout", type=float, default=120.0)
    args = parser.parse_args()

    positive = {
        "trials": args.trials,
        "samples": args.samples,
        "warm_repeats": args.warm_repeats,
        "concurrency": args.concurrency,
        "prefix_tokens": args.prefix_tokens,
        "query_tokens": args.query_tokens,
        "max_new_tokens": args.max_new_tokens,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        parser.error("must be positive: " + ", ".join(bad))
    if args.token_id_low < 0 or args.token_id_high <= args.token_id_low:
        parser.error("token ID range must satisfy 0 <= low < high")
    if args.request_timeout <= 0 or args.flush_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


@dataclass
class HTTPResponse:
    status: int
    reason: str
    headers: dict[str, str]
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        try:
            transfer = self.headers.get("transfer-encoding", "").lower()
            length_text = self.headers.get("content-length")
            if "chunked" in transfer:
                while True:
                    size_line = await self.reader.readline()
                    if not size_line:
                        raise RuntimeError("unexpected EOF in chunked response")
                    size = int(size_line.split(b";", 1)[0].strip(), 16)
                    if size == 0:
                        while True:
                            trailer = await self.reader.readline()
                            if trailer in (b"\r\n", b"\n", b""):
                                break
                        break
                    yield await self.reader.readexactly(size)
                    ending = await self.reader.readexactly(2)
                    if ending != b"\r\n":
                        raise RuntimeError("malformed chunk terminator")
            elif length_text is not None:
                remaining = int(length_text)
                while remaining:
                    chunk = await self.reader.read(min(65536, remaining))
                    if not chunk:
                        raise RuntimeError("unexpected EOF in fixed-length response")
                    remaining -= len(chunk)
                    yield chunk
            else:
                while True:
                    chunk = await self.reader.read(65536)
                    if not chunk:
                        break
                    yield chunk
        finally:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    async def read(self) -> bytes:
        return b"".join([chunk async for chunk in self.iter_bytes()])


class AsyncSGLangClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    async def request(
        self, method: str, path: str, payload: Any | None = None
    ) -> HTTPResponse:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        headers = [
            f"{method} {path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Accept: application/json, text/event-stream",
            "Connection: close",
            f"Content-Length: {len(body)}",
        ]
        if payload is not None:
            headers.append("Content-Type: application/json")
        writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body)
        await writer.drain()

        status_line = await reader.readline()
        if not status_line:
            writer.close()
            raise RuntimeError("server closed before returning an HTTP status")
        fields = status_line.decode("iso-8859-1").rstrip().split(" ", 2)
        if len(fields) < 2 or not fields[1].isdigit():
            writer.close()
            raise RuntimeError(f"malformed HTTP status: {status_line!r}")
        status = int(fields[1])
        reason = fields[2] if len(fields) == 3 else ""
        response_headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, sep, value = line.decode("iso-8859-1").partition(":")
            if not sep:
                writer.close()
                raise RuntimeError(f"malformed HTTP header: {line!r}")
            response_headers[name.strip().lower()] = value.strip()
        return HTTPResponse(status, reason, response_headers, reader, writer)

    async def json_request(
        self, method: str, path: str, payload: Any | None, timeout: float
    ) -> Any:
        async def operation() -> Any:
            response = await self.request(method, path, payload)
            body = await response.read()
            if not 200 <= response.status < 300:
                text = body.decode("utf-8", "replace")
                raise RuntimeError(f"{path} HTTP {response.status}: {text[:1000]}")
            if not body:
                return None
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                # /flush_cache has returned both JSON and plain text across
                # SGLang releases.  The HTTP status is the contract we need.
                return body.decode("utf-8", "replace")

        return await asyncio.wait_for(operation(), timeout=timeout)


class DurableJSONL:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def write_batch(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self._file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def seeded_tokens(
    nonce: str, low: int, high: int, count: int, *identity: object
) -> list[int]:
    digest = hashlib.sha256(
        (nonce + "\0" + "\0".join(map(str, identity))).encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    return [rng.randrange(low, high) for _ in range(count)]


def scalar_meta(meta: Any) -> dict[str, Any]:
    """Keep useful server metadata without accidentally logging token arrays."""
    if not isinstance(meta, dict):
        return {}
    return {
        key: value
        for key, value in meta.items()
        if value is None or isinstance(value, (bool, int, float, str))
    }


def cache_state(kind: str, cached_tokens: Any) -> tuple[str, bool]:
    if not isinstance(cached_tokens, int):
        return "unavailable", True
    if kind == "cold":
        return ("verified_miss", True) if cached_tokens == 0 else ("contaminated", False)
    return ("verified_hit", True) if cached_tokens > 0 else ("unexpected_miss", False)


async def generate_once(
    client: AsyncSGLangClient,
    timeout: float,
    request_record: dict[str, Any],
    input_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    loop = asyncio.get_running_loop()
    started = loop.time()
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_tokens = 0
    final_obj: dict[str, Any] = {}

    async def operation() -> None:
        nonlocal first_token_at, last_token_at, completion_tokens, final_obj
        response = await client.request("POST", "/generate", payload)
        if not 200 <= response.status < 300:
            body = await response.read()
            raise RuntimeError(
                f"/generate HTTP {response.status}: "
                f"{body.decode('utf-8', 'replace')[:1000]}"
            )
        pending = ""
        async for chunk in response.iter_bytes():
            pending += chunk.decode("utf-8", "replace")
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.rstrip("\r")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                obj = json.loads(data)
                if not isinstance(obj, dict):
                    continue
                final_obj = obj
                count = obj.get("meta_info", {}).get("completion_tokens", 0)
                if isinstance(count, int) and count > completion_tokens:
                    now = loop.time()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    completion_tokens = count

    try:
        await asyncio.wait_for(operation(), timeout=timeout)
        finished = loop.time()
        meta = final_obj.get("meta_info", {})
        if isinstance(meta, dict):
            server_count = meta.get("completion_tokens")
            if isinstance(server_count, int):
                completion_tokens = max(completion_tokens, server_count)
        decode_s = None
        decode_tok_s = None
        ms_per_decode_token = None
        if (
            completion_tokens > 1
            and first_token_at is not None
            and last_token_at is not None
            and last_token_at > first_token_at
        ):
            decode_s = last_token_at - first_token_at
            decode_tok_s = (completion_tokens - 1) / decode_s
            ms_per_decode_token = 1000.0 * decode_s / (completion_tokens - 1)
        cached_tokens = meta.get("cached_tokens") if isinstance(meta, dict) else None
        state, valid = cache_state(request_record["kind"], cached_tokens)
        return {
            **request_record,
            "record_type": "request",
            "ok": True,
            "valid_measurement": valid,
            "cache_state": state,
            "prompt_tokens": meta.get("prompt_tokens") if isinstance(meta, dict) else None,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": None if first_token_at is None else first_token_at - started,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
            "ms_per_decode_token": ms_per_decode_token,
            "e2e_s": finished - started,
            "server_meta": scalar_meta(meta),
            "finished_at_utc": utc_now(),
        }
    except Exception as error:  # Preserve failures in the durable campaign log.
        return {
            **request_record,
            "record_type": "request",
            "ok": False,
            "valid_measurement": False,
            "cache_state": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "e2e_s": loop.time() - started,
            "finished_at_utc": utc_now(),
        }


async def run_wave(
    jobs: list[tuple[dict[str, Any], list[int]]],
    client: AsyncSGLangClient,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(
        record: dict[str, Any], prompt: list[int]
    ) -> dict[str, Any]:
        async with semaphore:
            return await generate_once(
                client, args.request_timeout, record, prompt, args.max_new_tokens
            )

    started = asyncio.get_running_loop().time()
    results = await asyncio.gather(*(guarded(record, prompt) for record, prompt in jobs))
    return results, asyncio.get_running_loop().time() - started


def numeric_summary(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [
        float(record[metric])
        for record in records
        if record.get("ok")
        and record.get("valid_measurement")
        and isinstance(record.get(metric), (int, float))
        and math.isfinite(float(record[metric]))
    ]
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def summarize_wave(
    kind: str, records: list[dict[str, Any]], wall_s: float
) -> dict[str, Any]:
    total_tokens = sum(
        record.get("completion_tokens", 0)
        for record in records
        if record.get("ok") and isinstance(record.get("completion_tokens"), int)
    )
    return {
        "kind": kind,
        "requests": len(records),
        "ok": sum(bool(record.get("ok")) for record in records),
        "valid": sum(bool(record.get("valid_measurement")) for record in records),
        "cache_state_counts": {
            state: sum(record.get("cache_state") == state for record in records)
            for state in sorted({str(record.get("cache_state")) for record in records})
        },
        "wave_wall_s": wall_s,
        "client_completion_tok_s": total_tokens / wall_s if wall_s > 0 else None,
        "ttft_s": numeric_summary(records, "ttft_s"),
        "decode_tok_s": numeric_summary(records, "decode_tok_s"),
        "e2e_s": numeric_summary(records, "e2e_s"),
        "cached_tokens": numeric_summary(records, "cached_tokens"),
    }


def base_record(args: argparse.Namespace, run_id: str, nonce: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "run_nonce": nonce,
        "variant": args.variant,
        "block_index": args.block_index,
        "tag": args.tag,
    }


async def async_main(args: argparse.Namespace) -> int:
    run_nonce = args.nonce or f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"
    run_id = hashlib.sha256(run_nonce.encode("utf-8")).hexdigest()[:16]
    base = base_record(args, run_id, run_nonce)
    writer = DurableJSONL(args.out)
    client = AsyncSGLangClient(args.host, args.port)
    all_requests: list[dict[str, Any]] = []
    trial_summaries: list[dict[str, Any]] = []
    exit_code = 0

    try:
        try:
            server_info = await client.json_request(
                "GET", "/get_server_info", None, min(args.request_timeout, 60.0)
            )
            server_info_record = scalar_meta(server_info)
        except Exception as error:
            server_info_record = {"_error": f"{type(error).__name__}: {error}"}
        writer.write_batch(
            [
                {
                    **base,
                    "record_type": "run_start",
                    "started_at_utc": utc_now(),
                    "argv": sys.argv,
                    "host": args.host,
                    "port": args.port,
                    "trials": args.trials,
                    "samples": args.samples,
                    "warm_repeats": args.warm_repeats,
                    "concurrency": args.concurrency,
                    "prefix_tokens_requested": args.prefix_tokens,
                    "query_tokens_requested": args.query_tokens,
                    "max_new_tokens_requested": args.max_new_tokens,
                    "server_info": server_info_record,
                }
            ]
        )

        for trial in range(args.trials):
            try:
                flush_result = await client.json_request(
                    "POST", "/flush_cache", None, args.flush_timeout
                )
                flush_error = None
            except Exception as error:
                flush_result = None
                flush_error = f"{type(error).__name__}: {error}"
            flush_record = {
                **base,
                "record_type": "cache_flush",
                "trial": trial,
                "at_utc": utc_now(),
                "ok": flush_error is None,
                "response": flush_result,
                "error": flush_error,
            }
            writer.write_batch([flush_record])
            if flush_error is not None:
                print(f"trial {trial}: cache flush failed: {flush_error}", file=sys.stderr)
                exit_code = 2
                break

            prefixes: list[list[int]] = []
            cold_jobs: list[tuple[dict[str, Any], list[int]]] = []
            for pair in range(args.samples):
                prefix = seeded_tokens(
                    run_nonce,
                    args.token_id_low,
                    args.token_id_high,
                    args.prefix_tokens,
                    "prefix",
                    trial,
                    pair,
                )
                tail = seeded_tokens(
                    run_nonce,
                    args.token_id_low,
                    args.token_id_high,
                    args.query_tokens,
                    "cold-tail",
                    trial,
                    pair,
                )
                prefixes.append(prefix)
                cold_jobs.append(
                    (
                        {
                            **base,
                            "trial": trial,
                            "kind": "cold",
                            "pair_id": pair,
                            "warm_repeat": None,
                            "request_ordinal": pair,
                            "submitted_at_utc": utc_now(),
                        },
                        prefix + tail,
                    )
                )

            cold, cold_wall = await run_wave(cold_jobs, client, args)
            writer.write_batch(cold)
            all_requests.extend(cold)

            warm_jobs: list[tuple[dict[str, Any], list[int]]] = []
            for pair, prefix in enumerate(prefixes):
                seed_ok = cold[pair].get("ok") and cold[pair].get("valid_measurement")
                if not seed_ok:
                    continue
                for repeat in range(args.warm_repeats):
                    tail = seeded_tokens(
                        run_nonce,
                        args.token_id_low,
                        args.token_id_high,
                        args.query_tokens,
                        "warm-tail",
                        trial,
                        pair,
                        repeat,
                    )
                    warm_jobs.append(
                        (
                            {
                                **base,
                                "trial": trial,
                                "kind": "warm",
                                "pair_id": pair,
                                "warm_repeat": repeat,
                                "request_ordinal": len(warm_jobs),
                                "submitted_at_utc": utc_now(),
                            },
                            prefix + tail,
                        )
                    )

            if warm_jobs:
                warm, warm_wall = await run_wave(warm_jobs, client, args)
            else:
                warm, warm_wall = [], 0.0
            writer.write_batch(warm)
            all_requests.extend(warm)

            trial_summary = {
                **base,
                "record_type": "trial_summary",
                "trial": trial,
                "finished_at_utc": utc_now(),
                "cold": summarize_wave("cold", cold, cold_wall),
                "warm": summarize_wave("warm", warm, warm_wall),
            }
            cold_ttft = trial_summary["cold"]["ttft_s"]["median"]
            warm_ttft = trial_summary["warm"]["ttft_s"]["median"]
            trial_summary["warm_ttft_speedup"] = (
                cold_ttft / warm_ttft
                if isinstance(cold_ttft, float)
                and isinstance(warm_ttft, float)
                and warm_ttft > 0
                else None
            )
            trial_summary["valid"] = (
                trial_summary["cold"]["valid"] == len(cold)
                and len(warm) == len(warm_jobs)
                and trial_summary["warm"]["valid"] == len(warm)
                and bool(warm)
            )
            if not trial_summary["valid"]:
                exit_code = 2
            writer.write_batch([trial_summary])
            trial_summaries.append(trial_summary)
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "valid": trial_summary["valid"],
                        "cold_ttft_median_s": cold_ttft,
                        "warm_ttft_median_s": warm_ttft,
                        "warm_ttft_speedup": trial_summary["warm_ttft_speedup"],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )

        valid_trials = [trial for trial in trial_summaries if trial["valid"]]
        run_summary = {
            **base,
            "record_type": "run_summary",
            "finished_at_utc": utc_now(),
            "valid": exit_code == 0 and len(valid_trials) == args.trials,
            "trials_completed": len(trial_summaries),
            "valid_trials": len(valid_trials),
            "cold": summarize_wave(
                "cold", [r for r in all_requests if r.get("kind") == "cold"], 0.0
            ),
            "warm": summarize_wave(
                "warm", [r for r in all_requests if r.get("kind") == "warm"], 0.0
            ),
            "trial_cold_ttft_median_s": numeric_summary(
                [
                    {
                        "ok": trial["valid"],
                        "valid_measurement": trial["valid"],
                        "value": trial["cold"]["ttft_s"]["median"],
                    }
                    for trial in trial_summaries
                ],
                "value",
            ),
            "trial_warm_ttft_median_s": numeric_summary(
                [
                    {
                        "ok": trial["valid"],
                        "valid_measurement": trial["valid"],
                        "value": trial["warm"]["ttft_s"]["median"],
                    }
                    for trial in trial_summaries
                ],
                "value",
            ),
        }
        writer.write_batch([run_summary])
        print(json.dumps(run_summary, sort_keys=True), file=sys.stderr)
        return exit_code
    finally:
        writer.close()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
