"""Bounded, eager-only tensor capture for first-token MTP localization.

Diagnostic package only. Reads tensors on the forward stream, never changes
them. Each rank/layer/site is saved once at an explicit absolute position.
"""
import os
from pathlib import Path

ROOT = os.environ.get("GLM53_MTP_CAPTURE_ROOT")
POSITION = int(os.environ.get("GLM53_MTP_CAPTURE_POSITION", "-1"))
_seen = set()


def record(layer, site, batch, **tensors):
    if not ROOT or layer >= 45 or (site.startswith("dsa_") and layer % 4 != 3):
        return
    import torch
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    mode = batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MTP localization requires --disable-cuda-graph")
    key = (int(layer), site)
    if key in _seen:
        return
    positions = batch.positions.detach().cpu()
    if positions.numel() == 0 or int(positions.flatten()[0]) != POSITION:
        return
    rank = get_tensor_model_parallel_rank()
    destination = Path(ROOT) / f"rank{rank}-layer{layer:02d}-{site}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer": layer, "site": site, "rank": rank, "mode": str(mode),
        "positions": positions, "input_ids": batch.input_ids.detach().cpu(),
        "tensors": {},
    }
    def copy_tensor(name, value):
        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                copy_tensor(name + "." + str(index), child)
        elif value is None:
            payload["tensors"][name] = None
        else:
            copied = value.detach().cpu().clone()
            # Float8 CPU comparison operators are not all implemented. Widen
            # only the diagnostic copy, exactly, after the bounded capture.
            if str(copied.dtype).startswith("torch.float8"):
                copied = copied.float()
            payload["tensors"][name] = copied
    for name, value in tensors.items():
        copy_tensor(name, value)
    total = sum(t.numel() * t.element_size() for t in payload["tensors"].values() if t is not None)
    if total > 16 * 1024**2:
        raise RuntimeError("MTP localization tensor packet exceeds 16 MiB")
    with destination.open("xb") as stream:
        torch.save(payload, stream)
    _seen.add(key)
