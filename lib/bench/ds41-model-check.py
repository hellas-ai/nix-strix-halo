#!/usr/bin/env python3
"""Read-only identity check for the published V4.1 NVMe/RDMA model snapshot."""

import argparse
import hashlib
import json
import re
import struct
import subprocess
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


def verify_dspark(model_path, gamma):
    """Check bundled draft metadata and account storage bytes without loading weights.

    The TP4 average is a lower bound, not an allocation budget: replicated
    tensors, draft KV, activations and graph workspaces consume additional memory.
    Runtime pool sizing and the post-target draft graph memory check remain active.
    """
    require(gamma in (1, 3), "DSpark gamma must be 1 or 3 on this launcher")
    config = json.loads((model_path / "config.json").read_text())["text_config"]
    require(
        config.get("num_nextn_predict_layers") == 3
        and config.get("dspark_block_size") == 5
        and config.get("dspark_target_layer_ids") == [37, 38, 39]
        and config.get("dspark_markov_rank") == 256
        and config.get("dspark_n_routed_experts") == 128
        and config.get("dspark_num_experts_per_tok") == 3,
        "bundled DSpark geometry differs from the DS4.1 profile",
    )
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    draft = {
        name: shard for name, shard in weight_map.items() if name.startswith("mtp.")
    }
    require(
        {name.split(".")[1] for name in draft} == {"0", "1", "2"},
        "bundled DSpark must contain all three mtp layers",
    )
    shards = {}
    for name, shard in draft.items():
        require(
            Path(shard).name == shard and shard not in (".", ".."),
            "unsafe draft shard path",
        )
        shards.setdefault(shard, []).append(name)
    total = 0
    for shard, names in sorted(shards.items()):
        path = model_path / shard
        require(
            path.is_file() and not path.is_symlink(), "draft shard is absent or linked"
        )
        with path.open("rb") as source:
            length_bytes = source.read(8)
            require(len(length_bytes) == 8, "truncated draft shard header")
            length = struct.unpack("<Q", length_bytes)[0]
            require(2 <= length <= 16 << 20, "invalid draft shard header length")
            raw = source.read(length)
        require(len(raw) == length, "truncated draft shard metadata")
        header = json.loads(raw)
        payload_size = path.stat().st_size - 8 - length
        for name in names:
            require(name in header, f"draft tensor missing from shard: {name}")
            begin, end = header[name]["data_offsets"]
            require(
                0 <= begin < end <= payload_size, f"invalid draft tensor extent: {name}"
            )
            total += end - begin
    return {
        "algorithm": "DSPARK",
        "gamma": gamma,
        "verify_width": gamma + 1,
        "max_running_requests": 8 // (gamma + 1),
        "draft_tensors": len(draft),
        "draft_shards": sorted(shards),
        "draft_storage_bytes": total,
        "tp4_weight_bytes_lower_bound": (total + 3) // 4,
        "memory_qualification": "pending: replicated weights, draft KV and graph workspace must also fit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--dspark-gamma", type=int, choices=(1, 3))
    args = parser.parse_args()
    model = args.model_path
    require(model.is_absolute(), "model path must be absolute")
    # Establish exact RDMA snapshot identity before touching model metadata.
    storage = verify_storage(model)
    manifest = verify_manifest(model)
    draft = (
        {"dspark": verify_dspark(model, args.dspark_gamma)} if args.dspark_gamma else {}
    )
    print(
        json.dumps(
            {
                "verified": True,
                "model_path": str(model),
                **storage,
                **manifest,
                **draft,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
