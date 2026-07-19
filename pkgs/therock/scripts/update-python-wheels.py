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


BASE_URL = "https://rocm.nightlies.amd.com/whl-multi-arch"
DEFAULT_TARGET = "gfx1151"
DEFAULT_SERIES = "7.15"
BASE_PACKAGES = [
    "rocm",
    "rocm-bootstrap",
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-libraries",
]
TORCH_DEVICE_FAMILIES = {
    "gfx1100": "gfx110x",
    "gfx1101": "gfx110x",
    "gfx1102": "gfx110x",
    "gfx1103": "gfx110x",
    "gfx1150": "gfx115x",
    "gfx1151": "gfx115x",
    "gfx1152": "gfx115x",
    "gfx1153": "gfx115x",
    "gfx1200": "gfx12-0",
    "gfx1201": "gfx12-0",
}


def default_packages(target: str) -> list[str]:
    packages = BASE_PACKAGES + [
        f"amd-torch-device-{target}",
        f"amd-torchvision-device-{target}",
        f"rocm-sdk-device-{target}",
    ]
    family = TORCH_DEVICE_FAMILIES.get(target)
    if family is not None:
        packages.append(f"amd-torch-device-{family}")
    return packages


@dataclass(frozen=True)
class Distribution:
    project: str
    filename: str
    url: str
    package_version: str
    rocm_version: str | None
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
    return Distribution(
        project=project,
        filename=filename,
        url=urllib.parse.urljoin(index_url, href),
        package_version=package_version,
        rocm_version=rocm_match.group(1) if rocm_match is not None else None,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        kind=kind,
    )


def list_distributions(target: str, project: str) -> list[Distribution]:
    del target
    index_url = f"{BASE_URL}/{project}/"
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


def package_version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    public_version = version.split("+", 1)[0]
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.findall(r"\d+|[A-Za-z]+", public_version)
    )


def public_package_version(version: str) -> str:
    return version.split("+", 1)[0]


def dist_matches_package_version(dist: Distribution, package_version: str | None) -> bool:
    return package_version is None or public_package_version(dist.package_version) == package_version


def choose_rocm_version(
    distributions_by_project: dict[str, list[Distribution]],
    *,
    series: str,
    python_tag: str,
    package_versions: dict[str, str],
) -> str:
    common_versions: set[str] | None = None
    for project, distributions in distributions_by_project.items():
        versions = {
            dist.rocm_version
            for dist in distributions
            if dist.rocm_version is not None
            and dist.rocm_version.startswith(series)
            and dist_matches_python(dist, python_tag)
            and dist_matches_platform(dist)
            and dist_matches_package_version(dist, package_versions.get(normalize_project(project)))
        }
        if not any(dist.rocm_version is not None for dist in distributions):
            continue
        if not versions:
            raise SystemExit(
            f"no linux distribution found for {project} in ROCm series {series} "
                f"with Python tag {python_tag}"
            )
        common_versions = versions if common_versions is None else common_versions & versions

    if not common_versions:
        projects = ", ".join(distributions_by_project)
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
    package_version: str | None,
) -> Distribution:
    candidates = [
        dist
        for dist in distributions
        if (dist.rocm_version is None or dist.rocm_version == rocm_version)
        and dist_matches_python(dist, python_tag)
        and dist_matches_platform(dist)
        and dist_matches_package_version(dist, package_version)
    ]
    if not candidates:
        raise SystemExit(
            f"no distribution found for {distributions[0].project}: ROCm {rocm_version}, Python {python_tag}"
        )
    return sorted(
        candidates,
        key=lambda dist: (
            dist.kind == "wheel",
            package_version_key(dist.package_version),
        ),
    )[-1]


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


def load_sources(path: Path) -> dict:
    if not path.exists():
        return {"targets": {}}

    data = json.loads(path.read_text())
    if "targets" in data:
        data.setdefault("targets", {})
        return data

    if "target" in data:
        target = data["target"]
        return {
            "targets": {
                target: data,
            },
            "updated": data.get("updated"),
        }

    return {"targets": data}


def pinned_series(output: str) -> str | None:
    """Series (major.minor) of the currently pinned rocmVersion, so the
    default follows the checked-in pin instead of going stale when the
    nightly train rolls to a new series."""
    try:
        text = Path(output).read_text()
        match = re.search(r'"rocmVersion":\s*"(\d+\.\d+)\.', text)
        if match:
            return match.group(1)
    except OSError:
        pass
    return None


def pinned_python_tag(sources: dict, target: str) -> str | None:
    target_sources = sources.get("targets", {}).get(target)
    if isinstance(target_sources, dict):
        python_tag = target_sources.get("pythonTag")
        if isinstance(python_tag, str):
            return python_tag
    return None


def parse_package_version(value: str) -> tuple[str, str]:
    project, separator, version = value.partition("=")
    if not separator or not project or not version:
        raise argparse.ArgumentTypeError("expected PROJECT=VERSION")
    return normalize_project(project), version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--python-tag", help="Python ABI tag; defaults to the current pin")
    parser.add_argument(
        "--series",
        help="Version prefix to select; defaults to the currently pinned series",
    )
    parser.add_argument("--rocm-version", help="Exact ROCm wheel version to pin")
    parser.add_argument("--package", action="append", default=[], help="Project to pin")
    parser.add_argument(
        "--package-version",
        action="append",
        default=[],
        type=parse_package_version,
        metavar="PROJECT=VERSION",
        help="Select an exact public package version; repeat for multiple projects",
    )
    parser.add_argument("--output", default="pkgs/therock/sources/python-wheels.json")
    args = parser.parse_args()
    sources = load_sources(Path(args.output))
    python_tag = args.python_tag or pinned_python_tag(sources, args.target)
    if python_tag is None:
        parser.error("--python-tag is required when the output has no current pin for this target")

    if not args.series and not args.rocm_version:
        args.series = pinned_series(args.output) or DEFAULT_SERIES
        print(f"series from current pin: {args.series}", file=sys.stderr)

    projects = args.package or default_packages(args.target)
    package_versions = dict(args.package_version)

    # Device wheels must match the Python frontend wheel exactly. Derive their
    # constraints from the selected torch/torchvision versions so callers only
    # need to name the public stack versions once.
    torch_version = package_versions.get("torch")
    torchvision_version = package_versions.get("torchvision")
    for project in projects:
        normalized = normalize_project(project)
        if normalized.startswith("amd-torch-device-") and torch_version is not None:
            package_versions.setdefault(normalized, torch_version)
        if normalized.startswith("amd-torchvision-device-") and torchvision_version is not None:
            package_versions.setdefault(normalized, torchvision_version)

    distributions_by_project = {project: list_distributions(args.target, project) for project in projects}
    rocm_version = args.rocm_version or choose_rocm_version(
        distributions_by_project,
        series=args.series,
        python_tag=python_tag,
        package_versions=package_versions,
    )

    target_sources = {
        "target": args.target,
        "series": args.series,
        "rocmVersion": rocm_version,
        "pythonTag": python_tag,
        "index": f"{BASE_URL}/",
        "packages": {},
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    for project in projects:
        dist = choose_distribution(
            distributions_by_project[project],
            rocm_version=rocm_version,
            python_tag=python_tag,
            package_version=package_versions.get(normalize_project(project)),
        )
        print(f"prefetching {project}: {dist.filename}", file=sys.stderr)
        fetched = prefetch(dist.url)
        target_sources["packages"][project] = {
            "url": dist.url,
            "hash": fetched["hash"],
            "filename": dist.filename,
            "packageVersion": dist.package_version,
            "rocmVersion": dist.rocm_version or rocm_version,
            "pythonTag": dist.python_tag,
            "abiTag": dist.abi_tag,
            "platformTag": dist.platform_tag,
            "kind": dist.kind,
        }

    output = Path(args.output)
    sources["targets"][args.target] = target_sources
    sources["updated"] = target_sources["updated"]
    output.write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
