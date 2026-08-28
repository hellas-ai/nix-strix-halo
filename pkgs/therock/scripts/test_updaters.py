#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
