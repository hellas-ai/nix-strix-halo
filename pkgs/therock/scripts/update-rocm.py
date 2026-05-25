#!/usr/bin/env python3
"""
Refresh TheRock ROCm tarball pins used by this flake.

The TheRock nightly index is intentionally kept out of Nix evaluation. Run this
script manually when bumping the opt-in preview SDK, review the diff, then
commit the updated JSON.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://rocm.nightlies.amd.com/tarball-multi-arch/"
TARGET_SLUGS = {
    "gfx1010": "gfx101X-dgpu",
    "gfx1036": "gfx103X-all",
    "gfx1103": "gfx110X-all",
    "gfx110X": "gfx110X-dgpu",
    "gfx120X": "gfx120X-all",
}


def target_slug(target: str) -> str:
    return TARGET_SLUGS.get(target, target)


def fetch_index() -> str:
    with urllib.request.urlopen(BASE_URL, timeout=60) as response:
        return response.read().decode("utf-8")


def find_version(index: str, target: str, series: str | None) -> str:
    slug = re.escape(target_slug(target))
    pattern = re.compile(rf"therock-dist-linux-{slug}-([0-9][^\"<>/]+)\.tar\.gz")
    versions = sorted(set(pattern.findall(index)))
    if series is not None:
        versions = [version for version in versions if version.startswith(series)]
    if not versions:
        suffix = f" in series {series}" if series else ""
        raise SystemExit(f"no TheRock tarball found for {target}{suffix}")
    return versions[-1]


def prefetch(url: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "nix",
            "store",
            "prefetch-file",
            "--json",
            "--hash-type",
            "sha256",
            url,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", default=[], help="GPU target to pin")
    parser.add_argument("--series", default="7.13", help="Version prefix to select")
    parser.add_argument("--version", help="Exact TheRock version to pin")
    parser.add_argument("--output", default="pkgs/therock/sources/rocm.json")
    args = parser.parse_args()

    targets = args.target or ["gfx1151"]
    index = "" if args.version else fetch_index()

    sources = {"linux": {}}
    for target in targets:
        version = args.version or find_version(index, target, args.series)
        slug = target_slug(target)
        filename = f"therock-dist-linux-{slug}-{version}.tar.gz"
        url = f"{BASE_URL}{filename}"
        print(f"prefetching {target}: {version}", file=sys.stderr)
        fetched = prefetch(url)
        sources["linux"][target] = {
            "url": url,
            "hash": fetched["hash"],
            "version": version,
            "filename": filename,
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    Path(args.output).write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
