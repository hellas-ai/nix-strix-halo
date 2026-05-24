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
DEFAULT_KEY = "therock-7.13-gfx1151-full"
DEFAULT_TARGET = "gfx1151"
DEFAULT_VERSION = "7.13"
FAKE_HASH = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
DEFAULT_FETCH_ARGS: list[str] = [
    "--nested-submodules",
    "iree:third_party/flatcc,third_party/benchmark,third_party/llvm-project,third_party/torch-mlir",
    (
        "rocm-systems:"
        "experimental/python/perfxpert/opencode,"
        "projects/rocprofiler/perfetto,"
        "projects/rocprofiler-sdk/external/googletest,"
        "projects/rocprofiler-register/external/glog,"
        "projects/rocprofiler-sdk/external/fmt,"
        "projects/rocprofiler-register/external/fmt,"
        "projects/rocprofiler-sdk/source/docs/doxygen-awesome-css,"
        "projects/rocprofiler-sdk/external/ptl,"
        "projects/rocprofiler-sdk/external/cereal,"
        "projects/rocprofiler-sdk/external/filesystem,"
        "projects/rocprofiler-sdk/external/perfetto,"
        "projects/rocprofiler-sdk/external/elfio,"
        "projects/rocprofiler-sdk/external/yaml-cpp,"
        "projects/rocprofiler-sdk/external/json,"
        "projects/rocprofiler-sdk/external/sqlite,"
        "projects/rocprofiler-sdk/external/pybind11,"
        "projects/rocprofiler-sdk/external/gotcha,"
        "projects/rocprofiler-systems/external/timemory,"
        "projects/rocprofiler-systems/external/perfetto,"
        "projects/rocprofiler-systems/external/elfio,"
        "projects/rocprofiler-systems/external/dyninst,"
        "projects/rocprofiler-systems/external/kokkos,"
        "projects/rocprofiler-systems/external/papi,"
        "projects/rocprofiler-systems/external/pybind11,"
        "projects/rocprofiler-systems/external/sqlite,"
        "projects/rocprofiler-systems/examples/openmp/external/ompvv,"
        "projects/rocprofiler-systems/external/googletest,"
        "projects/rocprofiler-systems/external/filesystem,"
        "projects/rocprofiler-systems/external/spdlog,"
        "projects/rocprofiler-systems/external/json,"
        "projects/rocprofiler-compute/src/vendored/pyyaml,"
        "projects/rocprofiler-systems/external/onetbb,"
        "projects/rocprofiler-sdk/external/abseil-cpp"
    ),
    "rocm-libraries:fin,HostLibraryTests/googletest",
]
DEFAULT_DEEP_NESTED_SUBMODULES: list[dict[str, object]] = [
    {
        "parent": "rocm-systems/projects/rocprofiler-systems/external/timemory",
        "paths": [
            "external/yaml-cpp",
            "external/libunwind",
            "external/gotcha",
        ],
    },
]


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
        "--deep-nested-submodule",
        action="append",
        dest="deep_nested_submodules",
        type=parse_deep_nested_submodule,
        help="second-level submodule fetch in parent:path1,path2 form; repeat to replace the default policy",
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
    deep_nested_submodules = (
        args.deep_nested_submodules
        if args.deep_nested_submodules is not None
        else DEFAULT_DEEP_NESTED_SUBMODULES
    )

    if args.hash:
        source_hash = args.hash
    elif (
        old.get("rev") == rev
        and old.get("fetchArgs") == fetch_args
        and old.get("deepNestedSubmodules", []) == deep_nested_submodules
    ):
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
        "deepNestedSubmodules": deep_nested_submodules,
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    output.write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
