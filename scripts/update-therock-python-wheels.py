#!/usr/bin/env python3
"""
Refresh pinned TheRock Python wheels used by this flake.

The TheRock wheel index is intentionally kept out of Nix evaluation. Run this
script manually when bumping the opt-in binary PyTorch/ROCm stack, review the
diff, then commit the updated JSON.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://rocm.nightlies.amd.com/v2"
DEFAULT_TARGET = "gfx1151"
DEFAULT_PYTHON_TAG = "cp312"
DEFAULT_SERIES = "7.12"
BASE_PACKAGES = [
    "rocm",
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "rocm-sdk-core",
    "rocm-sdk-devel",
]
TARGET_SLUGS = {
    "gfx1010": "gfx101X-dgpu",
    "gfx1036": "gfx103X-all",
    "gfx1103": "gfx110X-all",
}


def target_slug(target: str) -> str:
    return TARGET_SLUGS.get(target, target)


def default_packages(target: str) -> list[str]:
    return BASE_PACKAGES + [f"rocm-sdk-libraries-{target_slug(target).lower()}"]


@dataclass(frozen=True)
class Distribution:
    project: str
    filename: str
    url: str
    package_version: str
    rocm_version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    kind: str


def normalize_project(project: str) -> str:
    return project.replace("_", "-").lower()


def fetch_index(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8")
        except Exception as error:  # pragma: no cover - updater resilience
            last_error = error
            if attempt == 3:
                break
            time.sleep(2 * attempt)
    raise SystemExit(f"failed to fetch {url}: {last_error}")


def parse_distribution(project: str, href: str, index_url: str) -> Distribution | None:
    if not (href.endswith(".whl") or href.endswith(".tar.gz")):
        return None

    filename = Path(urllib.parse.unquote(href)).name
    if filename.endswith(".whl"):
        stem = filename[:-4]
        parts = stem.rsplit("-", 4)
        if len(parts) != 5:
            return None

        raw_name, package_version, python_tag, abi_tag, platform_tag = parts
        kind = "wheel"
        if normalize_project(raw_name) != normalize_project(project):
            return None
    else:
        stem = filename[:-7]
        raw_name, sep, package_version = stem.rpartition("-")
        if sep == "" or normalize_project(raw_name) != normalize_project(project):
            return None
        python_tag = "source"
        abi_tag = "source"
        platform_tag = "source"
        kind = "sdist"

    rocm_match = re.search(r"(?:\+|\.|-)rocm(7\.[^-+]+)", package_version)
    if rocm_match is None and (project.startswith("rocm-sdk") or project == "rocm"):
        rocm_match = re.search(r"^(7\.[^-+]+)$", package_version)
    if rocm_match is None:
        return None

    return Distribution(
        project=project,
        filename=filename,
        url=urllib.parse.urljoin(index_url, href),
        package_version=package_version,
        rocm_version=rocm_match.group(1),
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        kind=kind,
    )


def list_distributions(target: str, project: str) -> list[Distribution]:
    target_url = f"{BASE_URL}/{target_slug(target)}"
    index_url = f"{target_url}/{project}/"
    text = fetch_index(index_url)
    hrefs = re.findall(r'href="([^"]+)"', text)
    distributions = [parse_distribution(project, href, index_url) for href in hrefs]
    return [dist for dist in distributions if dist is not None]


def dist_matches_python(dist: Distribution, python_tag: str) -> bool:
    if dist.python_tag in { "py3", "source" }:
        return True
    return dist.python_tag == python_tag and dist.abi_tag == python_tag


def dist_matches_platform(dist: Distribution) -> bool:
    return dist.platform_tag in {
        "linux_x86_64",
        "any",
        "source",
    }


def choose_rocm_version(
    distributions_by_project: dict[str, list[Distribution]],
    *,
    series: str,
    python_tag: str,
) -> str:
    common_versions: set[str] | None = None
    for project, distributions in distributions_by_project.items():
        versions = {
            dist.rocm_version
            for dist in distributions
            if dist.rocm_version.startswith(series)
            and dist_matches_python(dist, python_tag)
            and dist_matches_platform(dist)
        }
        if not versions:
            raise SystemExit(
            f"no linux distribution found for {project} in ROCm series {series} "
                f"with Python tag {python_tag}"
            )
        common_versions = versions if common_versions is None else common_versions & versions

    if not common_versions:
        projects = ", ".join(wheels_by_project)
        raise SystemExit(
            f"no common ROCm version for {projects} in series {series} "
            f"with Python tag {python_tag}"
        )

    return sorted(common_versions)[-1]


def choose_distribution(
    distributions: list[Distribution],
    *,
    rocm_version: str,
    python_tag: str,
) -> Distribution:
    candidates = [
        dist
        for dist in distributions
        if dist.rocm_version == rocm_version
        and dist_matches_python(dist, python_tag)
        and dist_matches_platform(dist)
    ]
    if not candidates:
        raise SystemExit(
            f"no distribution found for {distributions[0].project}: ROCm {rocm_version}, Python {python_tag}"
        )
    return sorted(candidates, key=lambda dist: (dist.kind == "wheel", dist.package_version))[-1]


def prefetch(url: str) -> dict[str, str]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        try:
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
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(2 * attempt)
    raise SystemExit(f"failed to prefetch {url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--python-tag", default=DEFAULT_PYTHON_TAG)
    parser.add_argument("--series", default=DEFAULT_SERIES)
    parser.add_argument("--rocm-version", help="Exact ROCm wheel version to pin")
    parser.add_argument("--package", action="append", default=[], help="Project to pin")
    parser.add_argument("--output", default="therock-python-wheel-sources.json")
    args = parser.parse_args()

    projects = args.package or default_packages(args.target)
    distributions_by_project = {project: list_distributions(args.target, project) for project in projects}
    rocm_version = args.rocm_version or choose_rocm_version(
        distributions_by_project,
        series=args.series,
        python_tag=args.python_tag,
    )

    sources = {
        "target": args.target,
        "series": args.series,
        "rocmVersion": rocm_version,
        "pythonTag": args.python_tag,
        "index": f"{BASE_URL}/{target_slug(args.target)}/",
        "packages": {},
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    for project in projects:
        dist = choose_distribution(
            distributions_by_project[project],
            rocm_version=rocm_version,
            python_tag=args.python_tag,
        )
        print(f"prefetching {project}: {dist.filename}", file=sys.stderr)
        fetched = prefetch(dist.url)
        sources["packages"][project] = {
            "url": dist.url,
            "hash": fetched["hash"],
            "filename": dist.filename,
            "packageVersion": dist.package_version,
            "rocmVersion": dist.rocm_version,
            "pythonTag": dist.python_tag,
            "abiTag": dist.abi_tag,
            "platformTag": dist.platform_tag,
            "kind": dist.kind,
        }

    Path(args.output).write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
