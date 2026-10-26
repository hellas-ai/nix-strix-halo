#!/usr/bin/env python3
"""Verify and inventory a sharded Qwen3.8-Flash-Next checkpoint.

This intentionally parses safetensors headers without importing torch.  It is
safe to run while diagnosing a partial download and cheap enough to run as the
final staging gate for all 131 shards.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
EXPERT_RE = re.compile(
    r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight(?:_|$)"
)
FUSED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)$"
)


class InventoryError(RuntimeError):
    pass


def tensor_category(name: str) -> str:
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


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise InventoryError(f"truncated safetensors file: {path}")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8:
            raise InventoryError(
                f"invalid header length {header_length} for {path} ({file_size} bytes)"
            )
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InventoryError(f"invalid safetensors JSON header in {path}: {exc}")

    payload_size = file_size - 8 - header_length
    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = entry["dtype"]
            shape = entry["shape"]
            start, end = entry["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise InventoryError(f"malformed tensor {name!r} in {path}: {exc}")
        if dtype not in DTYPE_BYTES:
            raise InventoryError(f"unknown dtype {dtype!r} for {name!r} in {path}")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise InventoryError(f"invalid shape for {name!r} in {path}: {shape!r}")
        expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > payload_size:
            raise InventoryError(
                f"invalid data offsets [{start}, {end}] for {name!r} in {path}"
            )
        if end - start != expected_bytes:
            raise InventoryError(
                f"size mismatch for {name!r} in {path}: offsets contain "
                f"{end - start} bytes, shape/dtype require {expected_bytes}"
            )
        ranges.append((start, end, name))
        tensors[name] = entry

    previous_end = 0
    previous_name = "<payload-start>"
    for start, end, name in sorted(ranges):
        if start < previous_end:
            raise InventoryError(
                f"overlapping tensors {previous_name!r} and {name!r} in {path}"
            )
        previous_end = end
        previous_name = name
    return tensors


def inspect(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    model_dir = args.model.resolve()
    errors: list[str] = []
    if not model_dir.is_dir():
        raise InventoryError(f"model directory does not exist: {model_dir}")

    config_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen4_exp":
            errors.append(
                f"config model_type is {config.get('model_type')!r}, expected 'qwen4_exp'"
            )
        architectures = config.get("architectures") or []
        if "Qwen4ExpForConditionalGeneration" not in architectures:
            errors.append(
                "config does not declare Qwen4ExpForConditionalGeneration"
            )
    else:
        errors.append("missing config.json")

    index_path = model_dir / "model.safetensors.index.json"
    index: dict[str, Any] | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    elif args.require_complete:
        errors.append("missing model.safetensors.index.json")

    present_shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    headers: dict[str, dict[str, dict[str, Any]]] = {}
    for shard in present_shards:
        try:
            headers[shard.name] = read_safetensors_header(shard)
        except (OSError, InventoryError) as exc:
            errors.append(str(exc))

    expected_shards: set[str] = set()
    weight_map: dict[str, str] = {}
    index_total_size: int | None = None
    if index is not None:
        weight_map = index.get("weight_map") or {}
        if not isinstance(weight_map, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
        ):
            errors.append("index weight_map is not a string-to-string mapping")
            weight_map = {}
        expected_shards = set(weight_map.values())
        index_total_size = (index.get("metadata") or {}).get("total_size")

        missing_shards = expected_shards - set(headers)
        extra_shards = set(headers) - expected_shards
        if missing_shards:
            errors.append(
                f"missing {len(missing_shards)} indexed shard(s): "
                + ", ".join(sorted(missing_shards)[:5])
            )
        if extra_shards:
            errors.append(
                f"found {len(extra_shards)} shard(s) absent from index: "
                + ", ".join(sorted(extra_shards)[:5])
            )

        for shard_name, shard_header in headers.items():
            indexed_names = {
                name for name, mapped_shard in weight_map.items() if mapped_shard == shard_name
            }
            actual_names = set(shard_header)
            missing_names = indexed_names - actual_names
            extra_names = actual_names - indexed_names
            if missing_names:
                errors.append(
                    f"{shard_name} lacks {len(missing_names)} indexed tensor(s): "
                    + ", ".join(sorted(missing_names)[:3])
                )
            if extra_names:
                errors.append(
                    f"{shard_name} contains {len(extra_names)} unindexed tensor(s): "
                    + ", ".join(sorted(extra_names)[:3])
                )

    shard_goal = len(expected_shards) if expected_shards else args.expected_shards
    if args.require_complete and len(present_shards) != shard_goal:
        errors.append(
            f"found {len(present_shards)} model shards, expected {shard_goal}"
        )

    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_bytes: Counter[str] = Counter()
    for shard_header in headers.values():
        for name, entry in shard_header.items():
            dtype = entry["dtype"]
            logical_bytes = math.prod(entry["shape"]) * DTYPE_BYTES[dtype]
            category = tensor_category(name)
            dtype_counts[dtype] += 1
            dtype_bytes[dtype] += logical_bytes
            category_counts[category] += 1
            category_bytes[category] += logical_bytes

    file_bytes = sum(path.stat().st_size for path in present_shards)
    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "complete": not errors and len(present_shards) == shard_goal,
        "present_shards": len(present_shards),
        "expected_shards": shard_goal,
        "indexed_tensors": len(weight_map),
        "inspected_tensors": sum(dtype_counts.values()),
        "model_file_bytes": file_bytes,
        "index_total_size": index_total_size,
        "dtypes": {
            key: {"tensors": dtype_counts[key], "bytes": dtype_bytes[key]}
            for key in sorted(dtype_counts)
        },
        "categories": {
            key: {"tensors": category_counts[key], "bytes": category_bytes[key]}
            for key in sorted(category_counts)
        },
        "errors": errors,
    }
    return report, errors


def gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 2**30:.3f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=131)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    try:
        report, errors = inspect(args)
    except (OSError, InventoryError, json.JSONDecodeError) as exc:
        print(f"inventory error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"model: {report['model_dir']}")
        print(
            f"shards: {report['present_shards']}/{report['expected_shards']}  "
            f"tensors: {report['inspected_tensors']}/{report['indexed_tensors'] or '?'}"
        )
        print(
            f"model shard bytes: {gib(report['model_file_bytes'])}  "
            f"index total: {gib(report['index_total_size'])}"
        )
        for category, values in report["categories"].items():
            print(
                f"  {category:16s} {values['tensors']:7d} tensors  "
                f"{gib(values['bytes'])}"
            )
        if errors:
            print("errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        elif report["complete"]:
            print("checkpoint inventory: complete and internally consistent")
        else:
            print("checkpoint inventory: partial but present shards are consistent")

    return 1 if errors and args.require_complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
