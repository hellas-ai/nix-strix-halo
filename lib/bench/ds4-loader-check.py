#!/usr/bin/env python3
"""Check owned pread loading preserves native scale and packed-weight bytes."""

import json
import struct
import tempfile
from pathlib import Path

import torch
from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator


def main():
    cases = {
        "scale": (
            "F8_E8M0",
            torch.float8_e8m0fnu,
            bytes([0, 1, 126, 127, 128, 254, 255]),
        ),
        "packed": ("I8", torch.int8, bytes(range(256))),
        "dense": ("BF16", torch.bfloat16, struct.pack("<HH", 0x3F80, 0xC000)),
    }
    header, data = {}, bytearray()
    for name, (kind, dtype, raw) in cases.items():
        start = len(data)
        data.extend(raw)
        header[name] = {
            "dtype": kind,
            "shape": [len(raw) // dtype.itemsize],
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "native.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        for disable_mmap in (False, True):
            loaded = dict(
                safetensors_weights_iterator([str(path)], disable_mmap=disable_mmap)
            )
            # Check after the file context has closed, including NaN payloads.
            for name, (_, dtype, raw) in cases.items():
                assert loaded[name].dtype == dtype
                assert bytes(loaded[name].view(torch.uint8).tolist()) == raw
    print("Native E8M0, packed E2M1 storage, and BF16 bytes survive mmap/pread loading")


if __name__ == "__main__":
    main()
