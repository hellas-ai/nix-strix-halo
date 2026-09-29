#!/usr/bin/env python3
"""Check cached/fresh generation at physical, packed-index and chunk boundaries.

Use with radix caching and 1024-token chunked prefill. Synthetic IDs exercise
cache ownership and numerical consistency; they do not measure language quality.
This requires an otherwise idle server because it flushes the prefix cache.
"""

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:30053")
parser.add_argument("--max-logprob-delta", type=float, default=0.0)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--prefixes",
    type=int,
    nargs="+",
    help="Common prefix lengths to check; default sweeps all 28 boundaries",
)
args = parser.parse_args()
assert math.isfinite(args.max_logprob_delta) and args.max_logprob_delta >= 0
base = args.base_url.rstrip("/")


def req(path, body=None):
    r = urllib.request.Request(
        base + path,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=600) as f:
        data = f.read()
        try:
            return json.loads(data)
        except ValueError:
            return data.decode()


for _ in range(900):
    try:
        req("/health")
        break
    except (OSError, TimeoutError):
        time.sleep(2)
else:
    raise SystemExit("server not healthy")


def generate(ids, tokens=16):
    return req(
        "/generate",
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": len(ids) - 1,
        },
    )


original = [1000 + i % 251 for i in range(3073)]
lengths = args.prefixes or [
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    2111,
    2112,
    2113,
    2175,
    2176,
    2177,
    2239,
    2240,
    2241,
    2303,
    2304,
    2305,
    3071,
]
assert all(0 < n < len(original) for n in lengths)
results = []
for common in lengths:
    req("/flush_cache")
    generate(original, 8)
    branch = original[:common] + [752, 317]
    cached = generate(branch)
    req("/flush_cache")
    fresh = generate(branch)
    a = cached["meta_info"]["output_token_logprobs"]
    b = fresh["meta_info"]["output_token_logprobs"]
    assert len(a) == len(b) == 16 and all(math.isfinite(v[0]) for v in a + b)
    assert fresh["meta_info"]["cached_tokens"] == 0
    if common >= 1024:
        assert cached["meta_info"]["cached_tokens"] >= 1024
    equal_ids = cached["output_ids"] == fresh["output_ids"]
    delta = max(abs(x[0] - y[0]) for x, y in zip(a, b, strict=True))
    result = {
        "common_prefix": common,
        "prompt_tokens": len(branch),
        "cached_tokens": cached["meta_info"]["cached_tokens"],
        "same_ids": equal_ids,
        "max_abs_logprob_delta": delta,
    }
    results.append(result)
    print(json.dumps(result), flush=True)
if args.output:
    args.output.write_text(json.dumps(results, indent=2) + "\n")
assert all(
    x["same_ids"] and x["max_abs_logprob_delta"] <= args.max_logprob_delta
    for x in results
), results
print("PASS cache boundary token/probability comparisons", flush=True)
