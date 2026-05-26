#!/usr/bin/env python3
"""
Refresh third-party source pins used by the TheRock source build.

TheRock's own CMake build describes several archive downloads with URL_HASH
entries. In Nix those must be fetched outside the build sandbox and exposed as
a local mirror, so this script records those URLs and hashes in JSON.
"""

import argparse
import base64
import json
import re
import subprocess
from pathlib import Path


DEFAULT_OUTPUT = "pkgs/therock/sources/rocm-third-party.json"
SPIRV_HEADERS_HASH = "sha256-NECkIWUCrMdsmMfzmAT6i11h3rLSZeDcItKaSuqPXhw="
ESMI_IB_LIBRARY_URL = "https://github.com/amd/esmi_ib_library.git"
ESMI_IB_LIBRARY_HASH = "sha256-I09JTi6I6Ny2Oso7Uitu6EgrtkTEfHmH45jYWdr1cpk="


def sri(algo: str, hex_digest: str) -> str:
    return f"{algo.lower()}-{base64.b64encode(bytes.fromhex(hex_digest)).decode()}"


def cmake_vars(text: str) -> dict[str, str]:
    vars: dict[str, str] = {}
    for match in re.finditer(r'^\s*set\((\w+)\s+"([^"]+)"\)', text, re.MULTILINE):
        vars[match.group(1)] = match.group(2)
    return vars


def expand(value: str, vars: dict[str, str]) -> str:
    value = value.strip().strip('"')
    if value.startswith("${") and value.endswith("}"):
        value = vars.get(value[2:-1], value)

    def repl(match: re.Match[str]) -> str:
        return vars.get(match.group(1), match.group(0))

    return re.sub(r"\$\{(\w+)\}", repl, value)


def scan_archives(source: Path) -> dict[str, dict[str, str]]:
    archives: dict[str, dict[str, str]] = {}
    for path in sorted((source / "third-party").rglob("CMakeLists.txt")):
        text = path.read_text()
        vars = cmake_vars(text)
        pending_url: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            url_match = re.match(r"URL\s+(.+)$", line)
            if url_match:
                pending_url = expand(url_match.group(1), vars)
                continue
            hash_match = re.match(r'URL_HASH\s+"?(SHA(256|512))=([0-9A-Fa-f]+)"?', line)
            if hash_match and pending_url and pending_url.startswith("https://"):
                algo = hash_match.group(1).lower()
                digest = hash_match.group(3)
                name = Path(pending_url).name
                archives[name] = {
                    "url": pending_url,
                    "hash": sri(algo, digest),
                }
                pending_url = None
    return archives


def ls_remote_peeled_tag(url: str, tag: str) -> str:
    output = subprocess.check_output(
        ["git", "ls-remote", url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        text=True,
    )
    revs: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        rev, ref = line.split(None, 1)
        revs[ref] = rev
    return revs.get(f"refs/tags/{tag}^{{}}", revs[f"refs/tags/{tag}"])


def scan_esmi_ib_library(source: Path, hash_: str) -> dict[str, str]:
    cmake = (source / "rocm-systems/projects/amdsmi/CMakeLists.txt").read_text()
    match = re.search(r'set\(current_esmi_tag\s+"([^"]+)"\)', cmake)
    if not match:
        raise RuntimeError("could not find amdsmi current_esmi_tag")
    tag = match.group(1)
    return {
        "url": ESMI_IB_LIBRARY_URL,
        "ref": f"refs/tags/{tag}",
        "rev": ls_remote_peeled_tag(ESMI_IB_LIBRARY_URL, tag),
        "hash": hash_,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Staged TheRock source tree")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--spirv-headers-hash", default=SPIRV_HEADERS_HASH)
    parser.add_argument("--esmi-ib-library-hash", default=ESMI_IB_LIBRARY_HASH)
    args = parser.parse_args()

    source = Path(args.source)
    tag = (source / "compiler/spirv-llvm-translator/spirv-headers-tag.conf").read_text().strip()
    data = {
        "archives": scan_archives(source),
        "esmiIbLibrary": scan_esmi_ib_library(source, args.esmi_ib_library_hash),
        "spirvHeaders": {
            "url": f"https://github.com/KhronosGroup/SPIRV-Headers/archive/{tag}.tar.gz",
            "rev": tag,
            "hash": args.spirv_headers_hash,
        },
    }
    Path(args.output).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
