#!/usr/bin/env python3
"""Flatten per-row JSON snapshots produced by run-row.sh into a wide CSV.

Each input JSON has shape:
  {
    "row": {<config attrs from the matrix row>},
    "training": {"train_runtime_s": ..., "train_loss": ..., ...},
    "delta": {
      "master": {"<key>": <int>, ...},   # all numeric keys whose pre/post moved
      "worker": {"<key>": <int>, ...},
    },
    "log": {"master": <path>, "worker": <path>}
  }

We merge the union of all keys across input files and emit a single CSV.
Missing values are blank; the schema is deliberately sparse so adding a
new transport or capturing new debugfs counters doesn't require touching
the aggregator.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for k, v in value.items():
            sub = f"{prefix}.{k}" if prefix else k
            flatten(sub, v, out)
    else:
        out[prefix] = value


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: aggregate.py <out.csv> <row.json> [<row.json> ...]", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    rows = []
    keys: list[str] = []
    seen_keys: set[str] = set()
    for path in argv[2:]:
        with Path(path).open() as f:
            doc = json.load(f)
        flat: dict[str, object] = {}
        flatten("", doc, flat)
        for k in flat:
            if k not in seen_keys:
                seen_keys.add(k)
                keys.append(k)
        rows.append(flat)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
