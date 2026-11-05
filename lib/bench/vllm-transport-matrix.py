#!/usr/bin/env python3
"""Deterministic planning for the vLLM transport matrix benchmark.

The plan is the Cartesian product of the requested transports, models and
concurrencies.  A deterministic subset can be sampled with ``--sample-cases``
and previously completed cases can be removed with ``--resume``.

Only rows whose ``status`` is ``ok`` count as complete.  ``skipped``,
``failed`` and ``partial`` rows are retried on the next run.

The sample is always drawn from the full Cartesian product *before* completed
cases are removed.  That keeps ``SAMPLE_CASES``/``RANDOM_SEED`` selecting the
same original subset across resumes instead of drifting towards a new subset
each time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path


def read_completed(csv_path: Path) -> set[tuple[str, str, int, int]]:
    """Return the cases already recorded as successful in *csv_path*.

    Only ``status == "ok"`` rows are considered complete.  Missing files and
    malformed rows are ignored so a partial or corrupt results file never
    prevents a run from being planned.
    """
    done: set[tuple[str, str, int, int]] = set()
    if not csv_path.exists():
        return done
    raw = csv_path.read_bytes().replace(b"\0", b"")
    text = raw.decode("utf-8", "replace")
    rows = csv.DictReader(line for line in text.splitlines() if line.strip())
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            done.add(
                (
                    row["transport"],
                    row["model"],
                    int(row["concurrency"]),
                    int(row["max_tokens"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return done


def build_plan(
    transports,
    models,
    concurrencies,
    max_tokens,
    sample_cases,
    seed,
    done,
):
    """Return the ordered list of cases that still need to run.

    ``sample_cases`` of ``0`` (or a value greater than or equal to the full
    matrix) keeps the full Cartesian product.  Sampling happens before
    ``done`` is removed so resuming with the same seed selects the same
    original subset.
    """
    cases = [
        (transport, model, conc, max_tokens)
        for transport, model, conc in itertools.product(
            transports, models, concurrencies
        )
    ]
    if sample_cases > 0 and sample_cases < len(cases):
        rnd = random.Random(seed)
        cases = rnd.sample(cases, sample_cases)
    if done:
        cases = [case for case in cases if case not in done]

    transport_order = {name: i for i, name in enumerate(transports)}
    model_order = {name: i for i, name in enumerate(models)}
    cases.sort(key=lambda c: (transport_order[c[0]], model_order[c[1]], c[2]))
    return cases


def write_plan(path: Path, cases) -> None:
    """Write *cases* to *path* as tab-separated rows."""
    with path.open("w") as fh:
        for case in cases:
            fh.write("\t".join(map(str, case)) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--sample-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", required=True)
    parser.add_argument("--transports", required=True)
    parser.add_argument("--concurrencies", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    done = set()
    if args.resume and args.csv is not None:
        done = read_completed(args.csv)

    cases = build_plan(
        transports=args.transports.split(),
        models=args.models.split(),
        concurrencies=[int(x) for x in args.concurrencies.split()],
        max_tokens=args.max_tokens,
        sample_cases=args.sample_cases,
        seed=args.seed,
        done=done,
    )
    write_plan(args.plan, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
