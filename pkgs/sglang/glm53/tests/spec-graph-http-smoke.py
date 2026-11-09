#!/usr/bin/env python3
"""Short native-API graph qualification; invoke only inside the campaign lock.

Runs after model startup, before the long benchmark. This is a runtime smoke,
not a statistical proof of end-to-end model equivalence. JSON preserves all
returned tokens/logprobs and actual-width counters for inspection.
"""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import time
import urllib.request


def generate(url, label, count, temperature=1.0, delay=0.0):
    time.sleep(delay)
    body = {"text": f"A patient teacher explains {label}: counting, arithmetic, and checking every step.\n",
            "sampling_params": {"temperature": temperature, "top_p": .95,
                                "max_new_tokens": count, "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": -1,
            "top_logprobs_num": 3, "return_text_in_logprobs": False}
    request = urllib.request.Request(url.rstrip("/") + "/generate",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=90) as response:
        value = json.load(response)
    return {"label": label, "requested": count, "temperature": temperature,
            "start": start, "end": time.time(), "response": value}


def parse_widths(value):
    try:
        widths = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("verify widths must be comma-separated integers") from exc
    if not widths or any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError("verify widths must be positive integers")
    return widths


def validate(result, expect_spec, verify_widths=None):
    value, requested = result["response"], result["requested"]
    meta, ids = value["meta_info"], value["output_ids"]
    assert meta["completion_tokens"] == requested and len(ids) == requested, result["label"]
    assert all(isinstance(x, int) and x >= 0 for x in ids)
    logs = meta["output_token_logprobs"]
    assert len(logs) == requested and [x[1] for x in logs] == ids
    assert all(math.isfinite(x[0]) and x[0] <= .001 for x in logs)
    tops = meta["output_top_logprobs"]
    assert len(tops) == requested
    for log, top in zip(logs, tops):
        assert len(top) == 3 and all(math.isfinite(x[0]) for x in top)
        assert all(a[0] >= b[0] - .001 for a, b in zip(top, top[1:]))
        for item in top:
            if item[1] == log[1]:
                assert abs(item[0] - log[0]) < .02
    if expect_spec:
        assert verify_widths, "Explicit expected verify widths are required for speculative qualification"
        verify = meta["spec_verify_ct"]
        accepted, proposed = meta["spec_num_correct_drafts"], meta["spec_num_proposed_drafts"]
        hist = meta["spec_cap_lens_histogram"]
        assert all(isinstance(n, int) and n >= 0 for n in hist)
        assert verify > 0 and sum(hist) == verify
        assert all(width in verify_widths for width, n in enumerate(hist) if n), (hist, verify_widths)
        assert sum((width - 1) * n for width, n in enumerate(hist)) == proposed
        assert 0 <= accepted <= proposed
        assert abs(meta["spec_accept_rate"] - (accepted / proposed if proposed else 0)) < 1e-6
        if result["label"] == "single-sampled":
            width = verify_widths[0]
            assert len(hist) > width and hist[width] == verify, (hist, verify_widths)
    return {"label": result["label"], "seconds": result["end"] - result["start"],
            "tokens": len(ids), "spec": {k: v for k, v in meta.items() if k.startswith("spec_")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://strix-3:31041")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-spec", action="store_true", help="Require 0048 actual verified-proposal accounting and a nonempty runtime width-policy env")
    parser.add_argument("--verify-widths", type=parse_widths,
        help="Explicit per-running-request target widths: NEXTN e.g. 4,4,2,2; DFlash e.g. 8,4,4,2 (includes anchor token)")
    args = parser.parse_args()
    if args.expect_spec and not args.verify_widths:
        parser.error("--expect-spec requires --verify-widths matching the runtime algorithm's width policy")
    phases = [
        [("single-sampled", 32, 1., 0.)],
        [("mixed-greedy", 48, 0., 0.), ("mixed-sampled", 64, 1., 0.)],
        [("padded-three-a", 48, 1., 0.), ("padded-three-b", 64, 1., 0.), ("padded-three-c", 80, 1., 0.)],
        [("stagger-long", 128, 1., 0.), ("stagger-short", 32, 1., .2),
         ("stagger-medium", 64, 1., .4), ("stagger-fourth", 96, 1., .6)],
    ]
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for phase in phases:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(phase)) as pool:
                futures = [pool.submit(generate, args.url, *case) for case in phase]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    args.out.write_text(json.dumps(records, indent=2))
                    print(json.dumps(validate(record, args.expect_spec, args.verify_widths)), flush=True)
    except Exception:
        args.out.write_text(json.dumps(records, indent=2))
        raise
    print("PASS: sampled C1, mixed C2, padded C3, staggered C4-to-C1, committed token/logprob alignment", flush=True)


if __name__ == "__main__":
    main()
