#!/usr/bin/env python3
"""Refresh all checked-in targets, using each target's release series and Python ABI.

Run in the development shell. A source refresh includes generated flake inputs,
their lock entries, and third-party hashes before the caller validates or publishes.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SOURCES = SCRIPTS.parent / "sources"


def run_script(name: str, *args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / name), *args], check=True)


def refresh(kind: str) -> None:
    filename, key = {
        "rocm": ("rocm.json", "linux"),
        "python-wheels": ("python-wheels.json", "targets"),
        "therock-source": ("rocm-source.json", "targets"),
    }[kind]
    targets = json.loads((SOURCES / filename).read_text())[key]
    if not targets:
        raise ValueError(f"no targets configured in {filename}")
    if kind == "therock-source" and len(targets) != 1:
        raise ValueError("the generated TheRock source graph currently requires exactly one target")
    for target, pin in sorted(targets.items()):
        print(f"Updating {kind} for {target} using its checked-in settings", flush=True)
        if kind == "rocm":
            # Pass each target's series explicitly: rocm.json may contain
            # targets on different release trains.
            series = ".".join(pin["version"].split(".")[:2])
            run_script("update-rocm.py", "--target", target, "--series", series)
        elif kind == "python-wheels":
            run_script("update-python-wheels.py", "--target", target, "--series", pin["series"])
        else:
            run_script("update-rocm-source.py", "--target", target)
            run_script("update-source-tree.py", "--target", target)
            subprocess.run(["nix", "flake", "lock", "--accept-flake-config"], check=True)
            staged = subprocess.check_output(
                [
                    "nix", "build", "--accept-flake-config", "--no-link", "--print-out-paths",
                    f".#legacyPackages.x86_64-linux.{target}.therock-rocm-source-{target}",
                ],
                text=True,
            ).strip()
            run_script("update-rocm-third-party.py", "--source", staged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["rocm", "python-wheels", "therock-source"])
    args = parser.parse_args()
    refresh(args.kind)


if __name__ == "__main__":
    main()
