#!/usr/bin/env python3
"""Check retrieval at three context depths and repeated-prefix correctness.

Run once with radix caching disabled, then with it enabled. This diagnostic is
not a substitute for coding acceptance or a quantization/reference comparison.
The context budget includes space reserved for reasoning and the answer.
"""

import argparse
import json
import time
import urllib.request


def request(base, path, payload=None, *, raw=False):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        None if payload is None else json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as response:
        return response.read().decode() if raw else json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30053")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--output-reserve", type=int, default=2048)
    parser.add_argument("--expect-cache", choices=["on", "off"], required=True)
    args = parser.parse_args()
    assert 0 < args.output_reserve < args.context
    target = args.context - args.output_reserve
    expected = {"early": "amber-7251", "middle": "cedar-4938", "late": "violet-8614"}

    def make_messages(count):
        # Deterministic, distinct records avoid testing only a repeated token.
        rows = [f"record_{i:06d}: build={i % 97}, tests={i % 31}, status=passed\n"
                for i in range(count)]
        for fraction, (key, value) in zip((0.1, 0.5, 0.9), expected.items(), strict=True):
            rows[int(count * fraction)] = f"ACCEPTANCE_MARKER {key} = {value}\n"
        return [{"role": "user", "content":
                 "Read these build records. Three ACCEPTANCE_MARKER records contain "
                 "values to remember.\n" + "".join(rows) +
                 "\nReturn only a JSON object mapping early, middle, and late to "
                 "the exact values in their ACCEPTANCE_MARKER records."}]

    template = {"model": "glm-5.3-flash", "reasoning_effort": "low",
                "chat_template_kwargs": {"clear_thinking": True, "reasoning_effort": "low"}}

    def tokenize(count):
        messages = make_messages(count)
        result = request(args.base_url, "/tokenize", {**template, "messages": messages})
        return messages, result["count"]

    # Find the largest complete-record prompt fitting the target token budget.
    lo, hi = 10, 256
    while tokenize(hi)[1] <= target:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        _, tokens = tokenize(middle)
        if tokens <= target:
            lo = middle
        else:
            hi = middle
    messages, prompt_tokens = tokenize(lo)
    assert target - 64 <= prompt_tokens <= target, (prompt_tokens, target)
    print(json.dumps({"context": args.context, "prompt_tokens": prompt_tokens,
                      "output_reserve": args.output_reserve,
                      "expected": expected, "records": lo}), flush=True)
    request(args.base_url, "/flush_cache", raw=True)
    previous = None
    for phase in ("cold", "warm"):
        started = time.monotonic()
        result = request(args.base_url, "/v1/chat/completions", {
            **template, "messages": messages, "temperature": 0,
            "max_tokens": args.output_reserve,
        })
        elapsed = time.monotonic() - started
        print(json.dumps({"phase": phase, "wall_seconds": elapsed,
                          "result": result}), flush=True)
        choice = result["choices"][0]
        assert choice["finish_reason"] == "stop", choice
        content = choice["message"]["content"].strip()
        assert json.loads(content) == expected, content
        usage = result["usage"]
        assert usage["prompt_tokens"] == prompt_tokens, usage
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if phase == "cold" or args.expect_cache == "off":
            assert cached == 0, usage
        else:
            assert cached > prompt_tokens // 2, usage
        comparable = (content, choice["message"].get("reasoning_content"))
        if previous is not None:
            assert comparable == previous, "Cold/warm answer or reasoning differs"
        previous = comparable
    print(json.dumps({"context_and_cache": "passed", "cache": args.expect_cache}), flush=True)


if __name__ == "__main__":
    main()
