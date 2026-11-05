#!/usr/bin/env python3
"""Read-only identity check for the published V4.1 NVMe/RDMA model snapshot."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SNAPSHOT_UUID = "c609331f-926a-4791-8ac4-b143a66a9af5"
NQN = "nqn.2026-07.link.satanic.trex:models-c609331f-926a-4791-8ac4-b143a66a9af5"
MANIFEST_SHA256 = "b249f1037dd3ce64927bb25b122ba341a2e995570aa980c2bef8fec2caa9040a"
CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
FILE_COUNT = 88
TOTAL_BYTES = 510313353565


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_storage(
    model_path,
    *,
    mount_line=None,
    by_id=Path("/dev/disk/by-id"),
    block_class=Path("/sys/class/block"),
    nvme_class=Path("/sys/class/nvme"),
):
    require(
        model_path.is_dir() and not model_path.is_symlink(),
        "model path must be a direct directory on the snapshot",
    )
    if mount_line is None:
        mount_line = subprocess.check_output(
            [
                "findmnt",
                "-rn",
                "-T",
                str(model_path),
                "-o",
                "SOURCE,FSTYPE,OPTIONS,TARGET",
            ],
            text=True,
        ).strip()
    fields = mount_line.split()
    require(len(fields) == 4, "expected one unambiguous model mount")
    source, fs_type, options, target = fields
    require(
        Path(target) == model_path.parent,
        "model must be immediately under its snapshot mountpoint",
    )
    require(
        fs_type == "xfs" and {"ro", "norecovery"} <= set(options.split(",")),
        "model mount is not read-only XFS with norecovery",
    )
    expected = by_id / ("nvme-uuid." + SNAPSHOT_UUID)
    require(expected.is_block_device(), "published snapshot UUID block device missing")
    device = Path(source).resolve(strict=True)
    require(
        device == expected.resolve(strict=True),
        "model mount does not use the pinned snapshot UUID",
    )
    match = re.fullmatch(r"nvme[0-9]+n[0-9]+", device.name)
    require(match is not None, "model mount is not a whole NVMe namespace")
    namespace = block_class / device.name
    require(namespace.is_dir(), "snapshot namespace sysfs entry is absent")
    require(
        (namespace / "uuid").read_text().strip().lower() == SNAPSHOT_UUID,
        "snapshot namespace UUID differs from published snapshot",
    )
    require(
        (namespace / "device" / "subsysnqn").read_text().strip() == NQN,
        "snapshot namespace belongs to a different NQN",
    )
    live = []
    for controller in nvme_class.iterdir():
        if not re.fullmatch(r"nvme[0-9]+", controller.name) or not controller.is_dir():
            continue
        if (controller / "subsysnqn").read_text().strip() != NQN:
            continue
        if (controller / "state").read_text().strip() == "live":
            live.append(controller)
    require(live, "snapshot NQN has no live NVMe controller")
    require(
        all(
            (controller / "transport").read_text().strip() == "rdma"
            for controller in live
        ),
        "snapshot NQN has a live non-RDMA path",
    )
    return {
        "mount": target,
        "device": str(device),
        "snapshot_uuid": SNAPSHOT_UUID,
        "nqn": NQN,
        "transport": "rdma",
    }


def verify_manifest(model_path):
    manifest_path = model_path / ".staging-integrity.json"
    require(
        sha256(manifest_path) == MANIFEST_SHA256,
        "model manifest does not match the published snapshot",
    )
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("repository") == "deepseek-ai/DeepSeek-V4.1-Flash"
        and manifest.get("revision") == REVISION
        and manifest.get("complete") is True
        and manifest.get("total_files") == FILE_COUNT
        and manifest.get("verified_files") == FILE_COUNT
        and manifest.get("total_bytes") == TOTAL_BYTES
        and manifest.get("verified_bytes") == TOTAL_BYTES,
        "model manifest revision or completion is wrong",
    )
    files = manifest.get("files")
    require(
        isinstance(files, list) and len(files) == FILE_COUNT,
        "model manifest file list is incomplete",
    )
    total = 0
    for entry in files:
        rel = Path(entry["path"])
        require(
            not rel.is_absolute() and ".." not in rel.parts,
            "model manifest contains an unsafe path",
        )
        path = model_path / rel
        require(
            path.is_file() and not path.is_symlink(),
            f"model file missing or linked: {rel}",
        )
        actual = path.stat().st_size
        require(actual == entry["bytes"], f"model file size mismatch: {rel}")
        total += actual
    require(total == TOTAL_BYTES, "model file total differs from manifest")
    require(
        sha256(model_path / "config.json") == CONFIG_SHA256,
        "model config digest differs from published snapshot",
    )
    require(
        sha256(model_path / "model.safetensors.index.json") == INDEX_SHA256,
        "model index digest differs from published snapshot",
    )
    return {
        "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "files": FILE_COUNT,
        "bytes": total,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: ds41-model-check.py MODEL_PATH")
    model = Path(sys.argv[1])
    require(model.is_absolute(), "model path must be absolute")
    # Establish exact RDMA snapshot identity before touching model metadata.
    storage = verify_storage(model)
    manifest = verify_manifest(model)
    print(
        json.dumps(
            {"verified": True, "model_path": str(model), **storage, **manifest},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
