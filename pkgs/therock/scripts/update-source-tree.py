#!/usr/bin/env python3
"""
Refresh TheRock source-tree inputs.

TheRock's fetch_sources.py stages a root checkout plus a selected set of
submodules and nested submodules. This script records that source graph as
flake inputs so Nix owns each fetched repository independently.
"""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_SOURCES = Path("pkgs/therock/sources/rocm-source.json")
DEFAULT_FLAKE = Path("flake.nix")
DEFAULT_TREE_OUTPUT = Path("pkgs/therock/sources/source-tree.nix")
BEGIN_INPUTS = "    # BEGIN generated TheRock source inputs"
END_INPUTS = "    # END generated TheRock source inputs"


@dataclass(frozen=True)
class GitSource:
    url: str
    rev: str


@dataclass(frozen=True)
class SourceEntry:
    path: str
    url: str
    rev: str
    input_name: str


def run(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def nix_string(value: str) -> str:
    return json.dumps(value)


def fetch_repo(repo_dir: Path, source: GitSource, *, checkout: bool = False) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        run(["git", "init", "-q"], cwd=repo_dir)
        run(["git", "remote", "add", "origin", source.url], cwd=repo_dir)

    fetch_cmd = [
        "git",
        "fetch",
        "-q",
        "--filter=blob:none",
        "--depth",
        "1",
        "origin",
        source.rev,
    ]
    result = run(fetch_cmd, cwd=repo_dir, check=False)
    if result.returncode != 0:
        run(["git", "fetch", "-q", "--filter=blob:none", "origin", source.rev], cwd=repo_dir)

    run(["git", "update-ref", "refs/heads/source", "FETCH_HEAD"], cwd=repo_dir)
    run(["git", "symbolic-ref", "HEAD", "refs/heads/source"], cwd=repo_dir)

    if checkout:
        run(["git", "checkout", "-q", "--detach", "source"], cwd=repo_dir)


def git_config_blob(repo_dir: Path, key_regex: str) -> dict[str, str]:
    result = run(
        ["git", "config", "--blob", "HEAD:.gitmodules", "--get-regexp", key_regex],
        cwd=repo_dir,
        check=False,
    )
    if result.returncode != 0:
        return {}
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        key, value = line.split(None, 1)
        values[key] = value
    return values


def submodules(repo_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    paths = git_config_blob(repo_dir, r"submodule\..*\.path")
    urls = git_config_blob(repo_dir, r"submodule\..*\.url")
    by_name: dict[str, dict[str, str]] = {}
    for key, path in paths.items():
        name = key.split(".")[1]
        url = urls.get(f"submodule.{name}.url")
        if url is None:
            continue
        by_name[name] = {
            "name": name,
            "path": path,
            "url": url,
        }
    by_path = {meta["path"]: meta for meta in by_name.values()}
    return by_name, by_path


def tree_commit(repo_dir: Path, path: str) -> str:
    result = run(["git", "ls-tree", "HEAD", "--", path], cwd=repo_dir)
    line = result.stdout.strip()
    if not line:
        raise RuntimeError(f"no git tree entry for {path} in {repo_dir}")
    fields = line.split(None, 3)
    if len(fields) < 3 or fields[1] != "commit":
        raise RuntimeError(f"{path} is not a submodule gitlink in {repo_dir}: {line}")
    return fields[2]


def enabled_root_paths(root_dir: Path, fetch_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "build_tools/fetch_sources.py",
        "--no-update-submodules",
        "--no-apply-patches",
    ] + fetch_args
    result = run(cmd, cwd=root_dir)
    marker = "$ git update-index --no-skip-worktree -- "
    for line in result.stdout.splitlines():
        if marker not in line:
            continue
        shell_cmd = line.split("$ ", 1)[1]
        parts = shlex.split(shell_cmd)
        separator = parts.index("--")
        return parts[separator + 1 :]
    raise RuntimeError(f"could not find enabled source paths in fetch_sources.py output:\n{result.stdout}")


def nested_specs(fetch_args: list[str]) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(fetch_args):
        arg = fetch_args[index]
        index += 1
        if arg != "--nested-submodules":
            continue
        while index < len(fetch_args) and not fetch_args[index].startswith("--"):
            project, values = fetch_args[index].split(":", 1)
            specs.append(
                (
                    project,
                    [value.strip() for value in values.split(",") if value.strip()],
                )
            )
            index += 1
    return specs


def input_name(series: str, target: str, path: str) -> str:
    if path == "":
        return f"therock-src-{series.replace('.', '-')}-{target}-root"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-").lower()
    digest = hashlib.sha1(path.encode()).hexdigest()[:8]
    if len(slug) > 64:
        slug = slug[:64].rstrip("-")
    return f"therock-src-{series.replace('.', '-')}-{target}-{slug}-{digest}"


def input_url(url: str, rev: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "github.com":
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = path.split("/")
        if len(parts) == 2:
            owner, repo = parts
            return f"github:{owner}/{repo}/{rev}"
    return f"git+{url}?rev={rev}&shallow=1"


def resolve_submodule(
    repo_dir: Path,
    selector: str,
    *,
    prefer_name: bool,
) -> dict[str, str]:
    by_name, by_path = submodules(repo_dir)
    if prefer_name and selector in by_name:
        return by_name[selector]
    if selector in by_path:
        return by_path[selector]
    if selector in by_name:
        return by_name[selector]
    raise RuntimeError(f"could not resolve submodule {selector!r} in {repo_dir}")


def source_key(series: str, target: str) -> str:
    return f"therock-{series}-{target}-full"


def build_graph(root_dir: Path, source: dict[str, object]) -> tuple[SourceEntry, list[SourceEntry]]:
    series = str(source["version"])
    target = str(source["target"])
    root_source = GitSource(url=str(source["url"]), rev=str(source["rev"]))
    root_entry = SourceEntry(
        path="",
        url=root_source.url,
        rev=root_source.rev,
        input_name=input_name(series, target, ""),
    )

    repos: dict[GitSource, Path] = {root_source: root_dir}
    entries: dict[str, SourceEntry] = {}

    def repo_for(git_source: GitSource) -> Path:
        if git_source not in repos:
            repo_dir = root_dir.parent / ("repo-" + hashlib.sha1(f"{git_source.url}@{git_source.rev}".encode()).hexdigest())
            fetch_repo(repo_dir, git_source)
            repos[git_source] = repo_dir
        return repos[git_source]

    def add_entry(path: str, url: str, rev: str) -> SourceEntry:
        if path in entries:
            return entries[path]
        entry = SourceEntry(
            path=path,
            url=url,
            rev=rev,
            input_name=input_name(series, target, path),
        )
        entries[path] = entry
        return entry

    fetch_args = [str(arg) for arg in source.get("fetchArgs", [])]
    for path in enabled_root_paths(root_dir, fetch_args):
        meta = resolve_submodule(root_dir, path, prefer_name=False)
        add_entry(path, meta["url"], tree_commit(root_dir, path))

    for parent_name, nested_paths in nested_specs(fetch_args):
        parent_meta = resolve_submodule(root_dir, parent_name, prefer_name=True)
        parent_path = parent_meta["path"]
        parent_entry = entries.get(parent_path)
        if parent_entry is None:
            continue
        parent_repo = repo_for(GitSource(parent_entry.url, parent_entry.rev))
        for nested_path in nested_paths:
            nested_meta = resolve_submodule(parent_repo, nested_path, prefer_name=True)
            add_entry(
                f"{parent_path}/{nested_meta['path']}",
                nested_meta["url"],
                tree_commit(parent_repo, nested_meta["path"]),
            )

    for item in source.get("deepNestedSubmodules", []):
        parent_path = str(item["parent"])
        parent_entry = entries.get(parent_path)
        if parent_entry is None:
            raise RuntimeError(f"deep nested parent {parent_path!r} was not fetched")
        parent_repo = repo_for(GitSource(parent_entry.url, parent_entry.rev))
        for nested_path in item["paths"]:
            nested_path = str(nested_path)
            nested_meta = resolve_submodule(parent_repo, nested_path, prefer_name=False)
            add_entry(
                f"{parent_path}/{nested_meta['path']}",
                nested_meta["url"],
                tree_commit(parent_repo, nested_meta["path"]),
            )

    sorted_entries = sorted(entries.values(), key=lambda entry: (entry.path.count("/"), entry.path))
    return root_entry, sorted_entries


def input_lines(root: SourceEntry, submodules: list[SourceEntry]) -> list[str]:
    entries = [root] + submodules
    lines = [
        BEGIN_INPUTS,
        "    # Generated by pkgs/therock/scripts/update-source-tree.py; do not edit by hand.",
    ]
    for entry in entries:
        lines += [
            f"    {nix_string(entry.input_name)} = {{",
            f"      url = {nix_string(input_url(entry.url, entry.rev))};",
            "      flake = false;",
            "    };",
            "",
        ]
    lines.append(END_INPUTS)
    return lines


def update_flake_inputs(path: Path, root: SourceEntry, submodules: list[SourceEntry]) -> None:
    lines = path.read_text().splitlines()
    try:
        begin = lines.index(BEGIN_INPUTS)
        end = lines.index(END_INPUTS)
    except ValueError as error:
        raise RuntimeError(f"missing generated TheRock source input markers in {path}") from error
    if begin >= end:
        raise RuntimeError(f"invalid generated TheRock source input marker order in {path}")
    replacement = input_lines(root, submodules)
    path.write_text("\n".join(lines[:begin] + replacement + lines[end + 1 :]) + "\n")


def write_tree(path: Path, target: str, version: str, root: SourceEntry, submodules: list[SourceEntry]) -> None:
    lines = [
        "# Generated by pkgs/therock/scripts/update-source-tree.py; do not edit by hand.",
        "{ inputs }:",
        "{",
        f"  {nix_string(target)} = {{",
        f"    version = {nix_string(version)};",
        f"    root = inputs.{nix_string(root.input_name)};",
        "    submodules = [",
    ]
    for entry in submodules:
        lines += [
            "      {",
            f"        path = {nix_string(entry.path)};",
            f"        source = inputs.{nix_string(entry.input_name)};",
            "      }",
        ]
    lines += [
        "    ];",
        "  };",
        "}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--series", default="7.13")
    parser.add_argument("--target", default="gfx1151")
    parser.add_argument("--flake", type=Path, default=DEFAULT_FLAKE)
    parser.add_argument("--tree-output", type=Path, default=DEFAULT_TREE_OUTPUT)
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text())
    source = sources[source_key(args.series, args.target)]

    with tempfile.TemporaryDirectory(prefix="therock-source-tree-") as temp:
        root_dir = Path(temp) / "root"
        fetch_repo(
            root_dir,
            GitSource(url=str(source["url"]), rev=str(source["rev"])),
            checkout=True,
        )
        root, submodules = build_graph(root_dir, source)

    update_flake_inputs(args.flake, root, submodules)
    write_tree(args.tree_output, args.target, str(source["version"]), root, submodules)
    print(f"updated {args.flake} and {args.tree_output} with {1 + len(submodules)} inputs")


if __name__ == "__main__":
    main()
