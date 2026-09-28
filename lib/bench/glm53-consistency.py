#!/usr/bin/env python3
"""Compare decode and teacher-forced prefill logprobs on the same token IDs.

This checks serving-path consistency, not agreement with the original FP8 model.
The thresholds are investigation gates, not a quantization quality claim.
"""

import argparse
import json
import math
import time
import urllib.request


def request(base, path, payload=None, *, json_response=True):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        None if payload is None else json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response) if json_response else response.read().decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30053")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-logprob-delta", type=float, default=0.05)
    parser.add_argument("--mean-logprob-delta", type=float, default=0.01)
    args = parser.parse_args()
    # The second prompt crosses the initial 1024-token prefill chunk boundary.
    prompts = [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "Review this configuration history.\n" +
        "The build uses Nix, Python, and ROCm. Tests must pass before release.\n" * 128 +
        "Write a short Python function that returns whether an integer is even:\n",
    ]
    failed = False
    for index, prompt in enumerate(prompts):
        request(args.base_url, "/flush_cache", json_response=False)
        started = time.monotonic()
        generated = request(args.base_url, "/generate", {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": args.tokens,
                                "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": 0,
        })
        generated_seconds = time.monotonic() - started
        meta = generated["meta_info"]
        decoded = meta["output_token_logprobs"]
        input_ids = [item[1] for item in meta["input_token_logprobs"]]
        output_ids = [item[1] for item in decoded]
        assert len(input_ids) == meta["prompt_tokens"]
        assert len(output_ids) == args.tokens
        if index == 1:
            assert len(input_ids) > 1024
        request(args.base_url, "/flush_cache", json_response=False)
        scored = request(args.base_url, "/generate", {
            "input_ids": input_ids + output_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True, "logprob_start_len": 0,
        })
        prefilled = scored["meta_info"]["input_token_logprobs"][len(input_ids):]
        assert [item[1] for item in prefilled] == output_ids
        assert not meta.get("cached_tokens", 0)
        assert not scored["meta_info"].get("cached_tokens", 0)
        deltas = [abs(a[0] - b[0]) for a, b in zip(decoded, prefilled, strict=True)]
        assert all(math.isfinite(d) for d in deltas)
        maximum, mean = max(deltas), sum(deltas) / len(deltas)
        passed = maximum <= args.max_logprob_delta and mean <= args.mean_logprob_delta
        failed |= not passed
        print(json.dumps({
            "prompt_index": index, "prompt_tokens": len(input_ids),
            "output_tokens": len(output_ids), "generated_seconds": generated_seconds,
            "max_logprob_delta": maximum, "mean_logprob_delta": mean,
            "thresholds": {"max": args.max_logprob_delta, "mean": args.mean_logprob_delta},
            "passed": passed, "generated": generated, "scored": scored,
        }), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
