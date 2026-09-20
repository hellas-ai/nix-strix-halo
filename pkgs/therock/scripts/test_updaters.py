#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).parent


def load_script(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update_rocm = load_script("update-rocm.py")
update_rocm_source = load_script("update-rocm-source.py")
update_wheels = load_script("update-python-wheels.py")
update_third_party = load_script("update-rocm-third-party.py")
update_stack = load_script("update-rocm-stack.py")


class RocmTarballUpdaterTests(unittest.TestCase):
    def test_selects_latest_stable_version_numerically(self):
        index = """
          therock-dist-linux-gfx1151-10.0.9.tar.gz
          therock-dist-linux-gfx1151-10.0.10.tar.gz
          therock-dist-linux-gfx1151-tests-10.0.11.tar.gz
          therock-dist-linux-gfx1151-10.1.0.tar.gz
        """
        self.assertEqual(update_rocm.find_version(index, "gfx1151", "10.0"), "10.0.10")

    def test_prefers_release_over_prerelease(self):
        index = """
          therock-dist-linux-gfx1151-10.0.0rc10.tar.gz
          therock-dist-linux-gfx1151-10.0.0.tar.gz
        """
        self.assertEqual(update_rocm.find_version(index, "gfx1151", "10.0"), "10.0.0")

    def test_maps_grouped_tarball_target(self):
        self.assertEqual(update_rocm.target_slug("gfx1030"), "gfx103X-all")


class PythonWheelUpdaterTests(unittest.TestCase):
    def parse(self, project: str, href: str):
        return update_wheels.parse_distribution(
            project,
            href,
            f"{update_wheels.BASE_URL}/{project}/",
        )

    def test_parses_rocm_10_frontend_and_sdk_versions(self):
        torch = self.parse(
            "torch",
            "torch-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl",
        )
        sdk = self.parse(
            "rocm-sdk-core",
            "rocm_sdk_core-10.0.0-py3-none-linux_x86_64.whl",
        )
        self.assertIsNotNone(torch)
        self.assertIsNotNone(sdk)
        self.assertEqual(torch.rocm_version, "10.0.0")
        self.assertEqual(sdk.rocm_version, "10.0.0")

    def test_parses_triton_git_version(self):
        triton = self.parse(
            "triton",
            "triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp313-cp313-linux_x86_64.whl",
        )
        self.assertIsNotNone(triton)
        self.assertEqual(triton.rocm_version, "10.0.0")

    def test_sorts_patch_and_prerelease_versions(self):
        self.assertGreater(update_wheels.version_key("10.0.10"), update_wheels.version_key("10.0.9"))
        self.assertGreater(update_wheels.version_key("10.0.0"), update_wheels.version_key("10.0.0rc4"))

    def test_gfx1151_closure_includes_split_device_wheels(self):
        packages = update_wheels.default_packages("gfx1151")
        self.assertIn("rocm-sdk-device-gfx1151", packages)
        self.assertIn("amd-torch-device-gfx1151", packages)
        self.assertIn("amd-torch-device-gfx115x", packages)
        self.assertIn("amd-torchvision-device-gfx1151", packages)


class RocmThirdPartyUpdaterTests(unittest.TestCase):
    def test_reuses_hash_only_for_identical_source(self):
        source = {"url": "https://example.test/source.tar.gz", "rev": "v2"}
        previous = source | {"hash": "sha256-existing"}
        with patch.object(update_third_party.subprocess, "check_output") as fetch:
            self.assertEqual(update_third_party.source_hash(source, previous, git=False), "sha256-existing")
            fetch.assert_not_called()

    def test_changed_archive_gets_fresh_unpacked_hash(self):
        source = {"url": "https://example.test/new.tar.gz", "rev": "v2"}
        previous = {"url": "https://example.test/old.tar.gz", "rev": "v1", "hash": "stale"}
        with patch.object(update_third_party.subprocess, "check_output", return_value='{"hash":"sha256-new"}') as fetch:
            self.assertEqual(update_third_party.source_hash(source, previous, git=False), "sha256-new")
            self.assertIn("--unpack", fetch.call_args.args[0])

    def test_changed_git_revision_gets_fresh_hash(self):
        source = {"url": "https://example.test/repo.git", "rev": "new"}
        previous = source | {"rev": "old", "hash": "stale"}
        with patch.object(update_third_party.subprocess, "check_output", side_effect=['{"sha256":"base32"}', 'sha256-new\n']) as fetch:
            self.assertEqual(update_third_party.source_hash(source, previous, git=True), "sha256-new")
            self.assertEqual(fetch.call_args_list[0].args[0][-1], "new")

    def test_parses_rocm_10_esmi_commit_pin(self):
        rev = "d494a3194ceb4cc4dbb2debf9fcbe8773c6d3bef"
        self.assertEqual(
            update_third_party.parse_esmi_pin(f'set(ESMI_GIT_HASH "{rev}")'),
            {"ref": rev, "rev": rev},
        )


class RocmSourceUpdaterTests(unittest.TestCase):
    def test_reads_series_from_selected_target(self):
        sources = {
            "targets": {
                "gfx1151": {"version": "10.0.7"},
                "gfx1100": {"version": "9.2.3"},
            }
        }
        self.assertEqual(update_rocm_source.pinned_series(sources, "gfx1151"), "10.0")
        self.assertEqual(update_rocm_source.pinned_series(sources, "gfx1100"), "9.2")

    def test_preserves_checked_in_fetch_policy(self):
        nested = [{"parent": "rocm-systems", "paths": ["projects/example"]}]
        sources = {
            "targets": {
                "gfx1151": {
                    "fetchArgs": ["--example"],
                    "deepNestedSubmodules": nested,
                }
            }
        }
        self.assertEqual(
            update_rocm_source.source_fetch_policy(sources, "gfx1151", None, None),
            (["--example"], nested),
        )


class RocmStackUpdaterTests(unittest.TestCase):
    def test_updates_all_binary_targets_using_their_own_series(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "rocm.json").write_text(json.dumps({"linux": {
                "gfx1151": {"version": "10.0.0"}, "gfx1030": {"version": "9.2.4"},
            }}))
            with patch.object(update_stack, "SOURCES", path), patch.object(update_stack, "run_script") as run:
                update_stack.refresh("rocm")
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[0].args[-3:], ("gfx1030", "--series", "9.2"))
                self.assertEqual(run.call_args_list[1].args[-3:], ("gfx1151", "--series", "10.0"))

    def test_wheel_updates_do_not_override_the_pinned_python_abi(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "python-wheels.json").write_text(json.dumps({"targets": {
                "gfx1151": {"series": "10.0", "pythonTag": "cp313"},
                "gfx1030": {"series": "10.0", "pythonTag": "cp313"},
            }}))
            with patch.object(update_stack, "SOURCES", path), patch.object(update_stack, "run_script") as run:
                update_stack.refresh("python-wheels")
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    self.assertNotIn("--python-tag", call.args)

    def test_source_refresh_includes_third_party_after_lock_and_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "rocm-source.json").write_text('{"targets":{"gfx1151":{"version":"10.0"}}}')
            events = []
            with patch.object(update_stack, "SOURCES", path), \
                 patch.object(update_stack, "run_script", side_effect=lambda *args: events.append(args)), \
                 patch.object(update_stack.subprocess, "run", side_effect=lambda cmd, **kw: events.append(tuple(cmd))), \
                 patch.object(update_stack.subprocess, "check_output", return_value='/nix/store/staged-source\n'):
                update_stack.refresh("therock-source")
            self.assertEqual([event[0] for event in events], [
                "update-rocm-source.py", "update-source-tree.py", "nix", "update-rocm-third-party.py",
            ])
            self.assertEqual(events[-1][-1], "/nix/store/staged-source")


if __name__ == "__main__":
    unittest.main()
