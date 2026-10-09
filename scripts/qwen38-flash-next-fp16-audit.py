#!/usr/bin/env python3
"""Exhaustively audit BF16 checkpoint values for native gfx1030 FP16 loading."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


EXPERT_RE = re.compile(
    r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
FUSED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)$"
)


def category(name: str) -> str:
    if ".mtp." in name or name.startswith("mtp."):
        return "mtp"
    if EXPERT_RE.search(name) or FUSED_EXPERT_RE.fullmatch(name):
        return "routed_experts"
    if ".mlp.shared_expert." in name:
        return "shared_expert"
    if ".ple." in name:
        return "ple"
    if ".visual." in name:
        return "vision"
    return "other"


def empty_metrics() -> dict[str, int | float]:
    return {
        "tensors": 0,
        "values": 0,
        "source_nonfinite": 0,
        "fp16_overflow": 0,
        "fp16_flush_to_zero": 0,
        "exact_values": 0,
        "source_energy": 0.0,
        "squared_error": 0.0,
        "max_abs_source": 0.0,
        "max_abs_error": 0.0,
    }


def merge(target: dict[str, int | float], source: dict[str, int | float]) -> None:
    for key in (
        "tensors",
        "values",
        "source_nonfinite",
        "fp16_overflow",
        "fp16_flush_to_zero",
        "exact_values",
        "source_energy",
        "squared_error",
    ):
        target[key] += source[key]
    target["max_abs_source"] = max(target["max_abs_source"], source["max_abs_source"])
    target["max_abs_error"] = max(target["max_abs_error"], source["max_abs_error"])


@torch.no_grad()
def audit_tensor(tensor: torch.Tensor) -> dict[str, int | float]:
    result = empty_metrics()
    result["tensors"] = 1
    result["values"] = tensor.numel()
    source = tensor.to(torch.float32)
    source_finite = torch.isfinite(source)
    result["source_nonfinite"] = torch.count_nonzero(~source_finite).item()
    finite_source = torch.where(source_finite, source, torch.zeros_like(source))
    converted = source.to(torch.float16)
    converted32 = converted.to(torch.float32)
    converted_finite = torch.isfinite(converted32)
    overflow = source_finite & ~converted_finite
    comparable = source_finite & converted_finite
    result["fp16_overflow"] = torch.count_nonzero(overflow).item()
    result["fp16_flush_to_zero"] = torch.count_nonzero(
        comparable & (source != 0) & (converted32 == 0)
    ).item()
    result["exact_values"] = torch.count_nonzero(comparable & (source == converted32)).item()
    difference = torch.where(comparable, converted32 - source, torch.zeros_like(source))
    result["source_energy"] = torch.sum(
        finite_source * finite_source, dtype=torch.float64
    ).item()
    result["squared_error"] = torch.sum(
        difference * difference, dtype=torch.float64
    ).item()
    result["max_abs_source"] = torch.max(torch.abs(finite_source)).item()
    result["max_abs_error"] = torch.max(torch.abs(difference)).item()
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output.resolve()
    if not model.is_dir():
        print(f"audit error: model directory does not exist: {model}", file=sys.stderr)
        return 2
    if output.is_relative_to("/tmp"):
        print("audit error: output must be durable and cannot use /tmp", file=sys.stderr)
        return 2

    index_path = model / "model.safetensors.index.json"
    if not index_path.is_file():
        print("audit error: complete safetensors index is required", file=sys.stderr)
        return 2
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted({model / value for value in index["weight_map"].values()})
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        print(f"audit error: {len(missing)} indexed shards are missing", file=sys.stderr)
        return 2

    aggregate = empty_metrics()
    categories: dict[str, dict[str, int | float]] = {}
    overflow_tensors: list[dict[str, Any]] = []
    nonfinite_tensors: list[dict[str, Any]] = []
    for shard_number, shard in enumerate(shards, start=1):
        print(f"[{shard_number}/{len(shards)}] {shard.name}", flush=True)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if not tensor.is_floating_point():
                    continue
                measured = audit_tensor(tensor)
                merge(aggregate, measured)
                bucket = categories.setdefault(category(name), empty_metrics())
                merge(bucket, measured)
                if measured["fp16_overflow"]:
                    overflow_tensors.append(
                        {"name": name, "count": measured["fp16_overflow"]}
                    )
                if measured["source_nonfinite"]:
                    nonfinite_tensors.append(
                        {"name": name, "count": measured["source_nonfinite"]}
                    )

    for measured in [aggregate, *categories.values()]:
        energy = float(measured["source_energy"])
        error = float(measured["squared_error"])
        measured["relative_frobenius_error"] = (
            math.sqrt(error / energy) if energy > 0 else 0.0
        )
        measured["exact_fraction"] = (
            int(measured["exact_values"]) / int(measured["values"])
            if measured["values"]
            else 1.0
        )

    passed = not overflow_tensors and not nonfinite_tensors
    report = {
        "model": str(model),
        "passed": passed,
        "target_dtype": "float16",
        "aggregate": aggregate,
        "categories": categories,
        "overflow_tensors": overflow_tensors,
        "nonfinite_tensors": nonfinite_tensors,
    }
    write_json_atomic(output, report)
    print(
        f"passed={passed} values={aggregate['values']} "
        f"overflow={aggregate['fp16_overflow']} "
        f"flush_to_zero={aggregate['fp16_flush_to_zero']} "
        f"relative_error={aggregate['relative_frobenius_error']:.10g}"
    )
    print(f"report: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
