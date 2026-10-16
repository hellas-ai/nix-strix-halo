#!/usr/bin/env python3
"""Synthetic, read-only storage checks for the V4.1 node launcher."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("ds41-model-check.py")
spec = importlib.util.spec_from_file_location("ds41_model_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "mount" / "DeepSeek-V4.1-Flash-hf-dba1be0a"
        self.model.mkdir(parents=True)
        self.device = root / "dev" / "nvme2n1"
        self.device.parent.mkdir()
        self.device.touch()
        self.by_id = root / "by-id"
        self.by_id.mkdir()
        (self.by_id / ("nvme-uuid." + check.SNAPSHOT_UUID)).symlink_to(self.device)
        self.block = root / "class-block"
        namespace = self.block / self.device.name
        (namespace / "device").mkdir(parents=True)
        (namespace / "uuid").write_text(check.SNAPSHOT_UUID + "\n")
        (namespace / "device" / "subsysnqn").write_text(check.NQN + "\n")
        self.nvme = root / "class-nvme"
        self.nvme.mkdir()
        # Native multipath can assign a controller number unrelated to nvme2n1.
        self.controller = self.add_controller("nvme8", "rdma", "live")
        self.mount = f"{self.device} xfs ro,norecovery {self.model.parent}"
        mocked = patch.object(
            Path, "is_block_device", lambda path: path.name.startswith("nvme-uuid.")
        )
        mocked.start()
        self.addCleanup(mocked.stop)

    def add_controller(self, name, transport, state, nqn=None):
        controller = self.nvme / name
        controller.mkdir()
        (controller / "subsysnqn").write_text((nqn or check.NQN) + "\n")
        (controller / "transport").write_text(transport + "\n")
        (controller / "state").write_text(state + "\n")
        return controller

    def verify(self, mount=None):
        return check.verify_storage(
            self.model,
            mount_line=self.mount if mount is None else mount,
            by_id=self.by_id,
            block_class=self.block,
            nvme_class=self.nvme,
        )

    def test_reconnected_namespace_and_rdma_controller(self):
        self.assertEqual(self.verify()["transport"], "rdma")

    def test_rejects_non_snapshot_or_writable_mount(self):
        for mount in (
            f"{self.device} nfs4 ro,norecovery {self.model.parent}",
            f"{self.device} xfs rw,norecovery {self.model.parent}",
            f"{self.device} xfs ro,norecovery {self.model.parent.parent}",
        ):
            with self.subTest(mount=mount), self.assertRaises(RuntimeError):
                self.verify(mount)

    def test_rejects_wrong_namespace_uuid_or_nqn(self):
        namespace = self.block / self.device.name
        (namespace / "uuid").write_text("00000000-0000-0000-0000-000000000000\n")
        with self.assertRaisesRegex(RuntimeError, "UUID"):
            self.verify()
        (namespace / "uuid").write_text(check.SNAPSHOT_UUID + "\n")
        (namespace / "device" / "subsysnqn").write_text("nqn.other\n")
        with self.assertRaisesRegex(RuntimeError, "NQN"):
            self.verify()

    def test_rejects_tcp_or_mixed_live_paths(self):
        (self.controller / "transport").write_text("tcp\n")
        with self.assertRaisesRegex(RuntimeError, "non-RDMA"):
            self.verify()
        (self.controller / "transport").write_text("rdma\n")
        self.add_controller("nvme9", "tcp", "live")
        with self.assertRaisesRegex(RuntimeError, "non-RDMA"):
            self.verify()

    def test_requires_live_matching_controller(self):
        (self.controller / "state").write_text("dead\n")
        self.add_controller("nvme9", "rdma", "live", nqn="nqn.other")
        with self.assertRaisesRegex(RuntimeError, "no live"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
