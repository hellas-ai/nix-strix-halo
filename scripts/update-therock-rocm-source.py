#!/usr/bin/env python3
"""
Refresh TheRock ROCm source pins used by this flake.

This intentionally does not fetch the full source graph. It records the TheRock
revision and source-fetch policy in JSON. The fixed-output Nix source package
then fetches the complete staged source tree and reports the recursive hash to
pin.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://github.com/ROCm/TheRock.git"
DEFAULT_REF = "refs/tags/therock-7.13"
DEFAULT_KEY = "therock-7.13-gfx1151-vllm"
DEFAULT_TARGET = "gfx1151"
DEFAULT_VERSION = "7.13"
FAKE_HASH = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
DEFAULT_FETCH_ARGS = [
    "--no-include-debug-tools",
    "--no-include-media-libs",
    "--no-include-iree-libs",
    "--no-include-ml-frameworks",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="therock-rocm-source-sources.json")
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--rev", help="Exact TheRock commit; otherwise resolved with git ls-remote")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--fetch-arg",
        action="append",
        dest="fetch_args",
        help="fetch_sources.py argument; repeat to replace the default fetch policy",
    )
    parser.add_argument(
        "--hash",
        help="Pinned recursive source hash. Omit to preserve the old hash when rev is unchanged, otherwise reset.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    sources = load_sources(output)
    old = sources.get(args.key, {})
    rev = args.rev or git_rev(args.url, args.ref)
    fetch_args = args.fetch_args if args.fetch_args is not None else DEFAULT_FETCH_ARGS

    if args.hash:
        source_hash = args.hash
    elif old.get("rev") == rev:
        source_hash = old.get("hash", FAKE_HASH)
    else:
        source_hash = FAKE_HASH

    sources[args.key] = {
        "url": args.url,
        "ref": args.ref,
        "rev": rev,
        "hash": source_hash,
        "target": args.target,
        "version": args.version,
        "fetchArgs": fetch_args,
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    output.write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
