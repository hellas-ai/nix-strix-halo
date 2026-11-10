#!/usr/bin/env python3
"""Tests for the vLLM transport matrix planner (stdlib only).

These tests only exercise the planning logic.  They never start vLLM, open an
SSH connection, or touch a GPU; the "results" they read are small temporary
CSV files.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "vllm-transport-matrix.py"
spec = importlib.util.spec_from_file_location("vllm_transport_matrix_plan", MODULE_PATH)
plan = importlib.util.module_from_spec(spec)
sys.modules["vllm_transport_matrix_plan"] = plan
spec.loader.exec_module(plan)


FIELDNAMES = ["transport", "model", "concurrency", "max_tokens", "status"]


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row(transport, model, concurrency, max_tokens, status):
    return {
        "transport": transport,
        "model": model,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "status": status,
    }


class ReadCompletedTests(unittest.TestCase):
    def test_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plan.read_completed(Path(tmp) / "missing.csv"), set())

    def test_only_ok_rows_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            write_csv(
                path,
                [
                    row("solo", "m", 1, 8, "ok"),
                    row("solo", "m", 2, 8, "skipped"),
                    row("solo", "m", 3, 8, "failed"),
                    row("solo", "m", 4, 8, "partial"),
                ],
            )
            self.assertEqual(plan.read_completed(path), {("solo", "m", 1, 8)})

    def test_malformed_rows_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text(
                "transport,model,concurrency,max_tokens,status\n"
                "solo,m,not-an-int,8,ok\n"
                "solo,m,2,8,ok\n"
                "solo,m,3,8,ok\n"
            )
            self.assertEqual(
                plan.read_completed(path), {("solo", "m", 2, 8), ("solo", "m", 3, 8)}
            )

    def test_null_bytes_are_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_bytes(
                b"transport,model,concurrency,max_tokens,status\nsolo,m,1,8,ok\n\x00"
            )
            self.assertEqual(plan.read_completed(path), {("solo", "m", 1, 8)})


class BuildPlanTests(unittest.TestCase):
    def build(self, **overrides):
        kwargs = {
            "transports": ["solo", "lan_tcp"],
            "models": ["a", "b"],
            "concurrencies": [1, 2],
            "max_tokens": 8,
            "sample_cases": 0,
            "seed": 1,
            "done": set(),
        }
        kwargs.update(overrides)
        return plan.build_plan(**kwargs)

    def test_sample_zero_keeps_full_matrix(self):
        cases = self.build()
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(set(cases)), 8)

    def test_sample_larger_than_matrix_keeps_full_matrix(self):
        self.assertEqual(len(self.build(sample_cases=100)), 8)

    def test_sampling_is_deterministic(self):
        self.assertEqual(self.build(sample_cases=3), self.build(sample_cases=3))

    def test_done_cases_are_removed(self):
        full = self.build()
        done = {full[0], full[3]}
        remaining = self.build(done=done)
        self.assertEqual(set(remaining), set(full) - done)

    def test_resume_keeps_original_sample(self):
        # The regression: sampling after removing completed cases would pick a
        # different subset on every resume.
        full = self.build(sample_cases=5)
        self.assertEqual(len(full), 5)
        done = {full[0], full[2]}
        resumed = self.build(sample_cases=5, done=done)
        self.assertEqual(resumed, [case for case in full if case not in done])
        self.assertTrue(set(resumed).issubset(set(full)))

    def test_resume_never_adds_unsampled_cases(self):
        full = set(self.build(sample_cases=5))
        for done in ([case] for case in full):
            resumed = set(self.build(sample_cases=5, done=set(done)))
            self.assertTrue(resumed.issubset(full))

    def test_plan_is_sorted(self):
        cases = self.build()
        self.assertEqual(
            cases,
            sorted(
                cases,
                key=lambda c: (
                    ["solo", "lan_tcp"].index(c[0]),
                    ["a", "b"].index(c[1]),
                    c[2],
                ),
            ),
        )


class CliTests(unittest.TestCase):
    def run_plan(self, tmp: Path, *extra: str) -> list[str]:
        plan_path = tmp / "plan.tsv"
        argv = [
            "--plan",
            str(plan_path),
            "--sample-cases",
            "0",
            "--seed",
            "1",
            "--models",
            "a",
            "--transports",
            "solo",
            "--concurrencies",
            "1 2",
            "--max-tokens",
            "8",
            *extra,
        ]
        self.assertEqual(plan.main(argv), 0)
        return plan_path.read_text().splitlines()

    def test_writes_tab_separated_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = self.run_plan(Path(tmp))
            self.assertEqual(lines, ["solo\ta\t1\t8", "solo\ta\t2\t8"])

    def test_resume_retries_skipped_but_skips_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "results.csv"
            write_csv(
                csv_path,
                [
                    row("solo", "a", 1, 8, "ok"),
                    row("solo", "a", 2, 8, "skipped"),
                ],
            )
            lines = self.run_plan(tmp_path, "--resume", "--csv", str(csv_path))
            self.assertEqual(lines, ["solo\ta\t2\t8"])

    def test_resume_retries_failed_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "results.csv"
            write_csv(
                csv_path,
                [
                    row("solo", "a", 1, 8, "failed"),
                    row("solo", "a", 2, 8, "partial"),
                ],
            )
            lines = self.run_plan(tmp_path, "--resume", "--csv", str(csv_path))
            self.assertEqual(lines, ["solo\ta\t1\t8", "solo\ta\t2\t8"])

    def test_without_resume_ignores_existing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "results.csv"
            write_csv(csv_path, [row("solo", "a", 1, 8, "ok")])
            lines = self.run_plan(tmp_path, "--csv", str(csv_path))
            self.assertEqual(lines, ["solo\ta\t1\t8", "solo\ta\t2\t8"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
