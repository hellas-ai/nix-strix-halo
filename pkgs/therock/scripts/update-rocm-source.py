#!/usr/bin/env python3
"""
Refresh TheRock ROCm source pins used by this flake.

This intentionally does not fetch the full source graph. It records the TheRock
revision and source-fetch policy in JSON. update-source-tree.py turns that
policy into explicit flake inputs for the root checkout and enabled submodules.
"""

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://github.com/ROCm/TheRock.git"
DEFAULT_TARGET = "gfx1151"
DEFAULT_SERIES = "7.15"
DEFAULT_FETCH_ARGS: list[str] = []
DEFAULT_DEEP_NESTED_SUBMODULES: list[dict[str, object]] = []


def parse_deep_nested_submodule(value: str) -> dict[str, object]:
    parent, paths = value.split(":", 1)
    path_list = [path.strip() for path in paths.split(",") if path.strip()]
    return {"parent": parent, "paths": path_list}


def git_rev(url: str, ref: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", url, ref, f"{ref}^{{}}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"no revision found for {url} {ref}")
    peeled_suffix = f"{ref}^{{}}"
    for line in lines:
        rev, resolved_ref = line.split()[:2]
        if resolved_ref == peeled_suffix:
            return rev
    for line in lines:
        rev, resolved_ref = line.split()[:2]
        if resolved_ref == ref:
            return rev
    raise SystemExit(f"no exact revision found for {url} {ref}:\n{result.stdout}")


def load_sources(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def pinned_series(output: str) -> str | None:
    """Series of the currently pinned source, so the default follows the
    checked-in pin instead of going stale when the release train rolls."""
    try:
        version = json.loads(Path(output).read_text()).get("version", "")
        match = re.match(r"(\d+\.\d+)", version)
        if match:
            return match.group(1)
    except (OSError, ValueError):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="pkgs/therock/sources/rocm-source.json")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--series",
        help="Release series to track; defaults to the currently pinned series",
    )
    parser.add_argument("--ref", help="Git ref to pin; defaults to refs/heads/release/therock-${series}")
    parser.add_argument("--rev", help="Exact TheRock commit; otherwise resolved with git ls-remote")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--version", help="Package version to record; defaults to --series")
    parser.add_argument(
        "--fetch-arg",
        action="append",
        dest="fetch_args",
        help="fetch_sources.py argument; repeat to replace the default fetch policy",
    )
    parser.add_argument(
        "--deep-nested-submodule",
        action="append",
        dest="deep_nested_submodules",
        type=parse_deep_nested_submodule,
        help="second-level submodule fetch in parent:path1,path2 form; repeat to replace the default policy",
    )
    args = parser.parse_args()

    if not args.series:
        args.series = pinned_series(args.output) or DEFAULT_SERIES

    output = Path(args.output)
    sources = load_sources(output)
    ref = args.ref or f"refs/heads/release/therock-{args.series}"
    version = args.version or args.series
    rev = args.rev or git_rev(args.url, ref)
    fetch_args = args.fetch_args if args.fetch_args is not None else DEFAULT_FETCH_ARGS
    deep_nested_submodules = (
        args.deep_nested_submodules
        if args.deep_nested_submodules is not None
        else DEFAULT_DEEP_NESTED_SUBMODULES
    )

    targets = sources.setdefault("targets", {})
    targets[args.target] = {
        "url": args.url,
        "ref": ref,
        "rev": rev,
        "version": version,
        "fetchArgs": fetch_args,
        "deepNestedSubmodules": deep_nested_submodules,
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    output.write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
