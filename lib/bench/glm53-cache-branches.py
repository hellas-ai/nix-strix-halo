#!/usr/bin/env python3
"""Compare cached and fresh native generation across shared-prefix branches.

Uses synthetic token IDs to exercise cache plumbing. Passing does not establish
language quality; combine with real-prompt retrieval and coding acceptance.
The server must have radix caching enabled. Graphs are controlled at launch.
"""

import argparse
import json
import math
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:30053")
args = parser.parse_args()
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


for _ in range(180):
    try:
        req("/health")
        break
    except (OSError, TimeoutError):
        time.sleep(2)
else:
    raise RuntimeError("server not healthy")


def generate(ids):
    started = time.monotonic()
    r = req(
        "/generate",
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": len(ids) - 1,
        },
    )
    meta = r["meta_info"]
    lp = meta["output_token_logprobs"]
    assert len(lp) == 8 and all(math.isfinite(x[0]) for x in lp), meta
    return {
        "wall_seconds": time.monotonic() - started,
        "ids": r["output_ids"],
        "logprobs": [x[0] for x in lp],
        "cached_tokens": meta["cached_tokens"],
    }


ids = [1000 + (i % 251) for i in range(3073)]
failures = []


def compare(label, first, second):
    delta = max(
        abs(a - b) for a, b in zip(first["logprobs"], second["logprobs"], strict=True)
    )
    same_ids = first["ids"] == second["ids"]
    if not same_ids or delta >= 0.05:
        failures.append(label)
    return {"same_ids": same_ids, "max_logprob_delta": delta}


# Establish run-to-run variation independently of cache reuse. A cache
# comparison alone cannot distinguish a reuse bug from nondeterminism.
req("/flush_cache")
control_first = generate(ids + [875, 97, 351])
req("/flush_cache")
control_second = generate(ids + [875, 97, 351])
assert control_first["cached_tokens"] == control_second["cached_tokens"] == 0
print(
    json.dumps(
        {
            "case": "fresh-repeat-control",
            "first": control_first,
            "second": control_second,
            **compare("fresh-repeat-control", control_first, control_second),
        }
    ),
    flush=True,
)
req("/flush_cache")
cold = generate(ids)
warm = generate(ids)
print(
    json.dumps(
        {
            "case": "identical",
            "cold": cold,
            "warm": warm,
            **compare("identical", cold, warm),
        }
    ),
    flush=True,
)
assert cold["cached_tokens"] == 0 and warm["cached_tokens"] > len(ids) // 2
for label, branch in [
    ("append", ids + [875, 97, 351]),
    ("truncate", ids[:2117] + [752, 317]),
    ("diverge", ids[:1536] + [1729 + (i % 31) for i in range(257)]),
]:
    cached = generate(branch)
    req("/flush_cache")
    fresh = generate(branch)
    print(
        json.dumps(
            {
                "case": label,
                "cached": cached,
                "fresh": fresh,
                **compare(label, cached, fresh),
            }
        ),
        flush=True,
    )
    assert cached["cached_tokens"] > 0 and fresh["cached_tokens"] == 0
    # Restore the original prefix for the next branch.
    generate(ids)
assert not failures, f"Failed numerical comparisons: {failures}"
print(
    "PASS cached/fresh token and probability checks: repeated prefix, append, truncate, divergence",
    flush=True,
)
