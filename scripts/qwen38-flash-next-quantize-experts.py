#!/usr/bin/env python3
"""Stream Qwen3.8-Flash-Next routed experts into W4A16.

The output follows compressed-tensors' pack-quantized contract consumed by the
matching SGLang ROCm Triton MoE implementation.  Only routed expert gate/up/down
checkpoint weights are changed.  Routers, the shared expert, PLE, GDN, QSA,
mHC, norms, embeddings, lm_head, MTP, and vision checkpoint weights remain at
their source precision.  The runtime manifest requests lossy FP8 PLE storage so
the offloaded tables fit in a 128 GiB Strix Halo host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils import calculate_qparams
from safetensors import safe_open
from safetensors.torch import save_file


FUSED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
TARGET_RE = r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
IGNORE_NON_TARGET_RE = (
    r"re:^(?!.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$).*"
)
PLE_RUNTIME_DTYPE = "float8_e4m3fn"
STATE_NAME = ".quantization-state.json"
PLAN_VERSION = "qwen38-flash-next-fused-expert-w4a16-rtn-v2"
SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


class QuantizationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.chmod(0o644)
    os.replace(partial, path)


def publish_output_modes(output: Path) -> None:
    """Make a completed checkpoint readable through the read-only model export."""
    output.chmod(0o755)
    for item in output.iterdir():
        if item.is_file():
            item.chmod(0o644)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuantizationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QuantizationError(f"expected JSON object in {path}")
    return value


def make_quant_args(group_size: int) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=group_size,
        dynamic=False,
        scale_dtype="float16",
    )


@torch.no_grad()
def quantize_weight(
    weight: torch.Tensor, args: QuantizationArgs
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    if weight.ndim != 2:
        raise QuantizationError(f"expert weight must be 2-D, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    group_size = int(args.group_size)
    if columns % group_size:
        raise QuantizationError(
            f"weight shape {tuple(weight.shape)} is not divisible by group_size={group_size}"
        )
    if not weight.is_floating_point():
        raise QuantizationError(f"expert weight has non-floating dtype {weight.dtype}")

    source = weight.to(torch.float32)
    grouped = source.reshape(rows, columns // group_size, group_size)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    scale, zero_point = calculate_qparams(minimum, maximum, args)
    stored_scale = scale.to(torch.float16)
    if not torch.isfinite(stored_scale).all() or torch.count_nonzero(stored_scale) != stored_scale.numel():
        raise QuantizationError("FP16 group scales overflowed or underflowed")

    qweight = quantize(
        x=source,
        scale=stored_scale,
        zero_point=zero_point,
        args=args,
        dtype=torch.int8,
    )
    packed = pack_to_int32(qweight, num_bits=4).contiguous()
    dequantized = qweight.reshape_as(grouped).to(torch.float32) * stored_scale.unsqueeze(-1)
    error = dequantized - grouped
    metrics: dict[str, float | int] = {
        "values": source.numel(),
        "source_energy": torch.sum(grouped * grouped, dtype=torch.float64).item(),
        "squared_error": torch.sum(error * error, dtype=torch.float64).item(),
        "max_abs_error": torch.max(torch.abs(error)).item(),
        "clipped_low": torch.count_nonzero(qweight == -8).item(),
        "clipped_high": torch.count_nonzero(qweight == 7).item(),
    }
    tensors = {
        "weight_packed": packed,
        "weight_scale": stored_scale.contiguous(),
        "weight_shape": torch.tensor(weight.shape, dtype=torch.int64),
    }
    return tensors, metrics


def merge_metrics(target: dict[str, float | int], source: dict[str, float | int]) -> None:
    for key in ("values", "source_energy", "squared_error", "clipped_low", "clipped_high"):
        target[key] = target.get(key, 0) + source.get(key, 0)
    target["max_abs_error"] = max(
        float(target.get("max_abs_error", 0.0)),
        float(source.get("max_abs_error", 0.0)),
    )


def quantize_shard(
    source_path: Path,
    output_path: Path,
    args: QuantizationArgs,
) -> tuple[int, int, dict[str, float | int]]:
    output: dict[str, torch.Tensor] = {}
    metrics: dict[str, float | int] = {}
    quantized_count = 0
    untouched_count = 0

    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name in source.keys():
            tensor = source.get_tensor(name)
            fused_match = FUSED_EXPERT_RE.fullmatch(name)
            if fused_match:
                if tensor.ndim != 3:
                    raise QuantizationError(
                        f"fused expert tensor {name} must be 3-D, got {tuple(tensor.shape)}"
                    )
                layer = int(fused_match.group(1))
                projection = fused_match.group(2)
                expert_count = tensor.shape[0]
                for expert in range(expert_count):
                    expert_weight = tensor[expert]
                    if projection == "gate_up_proj":
                        if expert_weight.shape[0] % 2:
                            raise QuantizationError(
                                f"fused gate/up tensor has odd projection axis: {name}"
                            )
                        projections = zip(
                            ("gate_proj", "up_proj"),
                            expert_weight.chunk(2, dim=0),
                            strict=True,
                        )
                    else:
                        projections = (("down_proj", expert_weight),)
                    for projection_name, projection_weight in projections:
                        converted, tensor_metrics = quantize_weight(
                            projection_weight, args
                        )
                        prefix = (
                            f"model.language_model.layers.{layer}.mlp.experts."
                            f"{expert}.{projection_name}"
                        )
                        for suffix, value in converted.items():
                            output[f"{prefix}.{suffix}"] = value
                        merge_metrics(metrics, tensor_metrics)
                        quantized_count += 1
            else:
                output[name] = tensor
                untouched_count += 1

    partial = output_path.with_name(f".{output_path.name}.partial")
    if partial.exists():
        partial.unlink()
    save_file(output, partial, metadata=metadata)
    partial.chmod(0o644)
    os.replace(partial, output_path)
    return quantized_count, untouched_count, metrics


def copy_ancillary_files(source: Path, output: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        STATE_NAME,
    }
    for item in source.iterdir():
        if item.name in excluded or item.name.startswith("model-"):
            continue
        if item.name == ".cache" or item.is_dir():
            continue
        destination = output / item.name
        shutil.copy2(item, destination)
        destination.chmod(0o644)


def make_output_config(source_config: dict[str, Any], group_size: int) -> dict[str, Any]:
    config = json.loads(json.dumps(source_config))
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise QuantizationError("config text_config must be a JSON object")
    # SGLang constructs each TP-sharded, pinned PLE table in this dtype and
    # downcasts the source BF16 shard while loading.  Keeping BF16 here needs
    # about 102 GiB of host RAM and OOMs once four workers and ROCm bookkeeping
    # are included; FP8 needs about 51 GiB and gathers back to BF16 for compute.
    text_config["ple_embedding_dtype"] = PLE_RUNTIME_DTYPE
    quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "routed_experts": {
                "targets": [TARGET_RE],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                    "dynamic": False,
                    "scale_dtype": "float16",
                },
                "input_activations": None,
            }
        },
        # compressed-tensors config groups are closed-world: every linear layer
        # must either match a quantized target or be explicitly ignored.  Keep
        # the routed expert projections quantized while leaving every other
        # source-precision layer on SGLang's unquantized path.
        "ignore": [IGNORE_NON_TARGET_RE],
    }
    config["quantization_config"] = quantization_config
    config["compression_config"] = quantization_config
    return config


def build_index(output: Path) -> tuple[dict[str, Any], int]:
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in sorted(output.glob("model-*-of-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in weight_map:
                    raise QuantizationError(f"duplicate output tensor {name}")
                tensor = handle.get_slice(name)
                dtype = tensor.get_dtype()
                shape = tensor.get_shape()
                if dtype not in SAFETENSORS_DTYPE_BYTES:
                    raise QuantizationError(f"unhandled output dtype {dtype} for {name}")
                total_size += math.prod(shape) * SAFETENSORS_DTYPE_BYTES[dtype]
                weight_map[name] = shard.name
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}, total_size


def validate_layout(source: Path, index: dict[str, Any], config: dict[str, Any]) -> int:
    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        raise QuantizationError("source index has no valid weight_map")
    fused_names = sorted(name for name in weight_map if FUSED_EXPERT_RE.fullmatch(name))
    text_config = config.get("text_config") or config
    layers = int(text_config.get("num_hidden_layers", 0))
    experts = int(text_config.get("num_experts", 0))
    expected_projections = layers * experts * 3
    expected_fused = layers * 2
    if expected_projections <= 0:
        raise QuantizationError("config does not declare a positive layer/expert count")
    expected_names = {
        f"model.language_model.layers.{layer}.mlp.experts.{projection}"
        for layer in range(layers)
        for projection in ("gate_up_proj", "down_proj")
    }
    if set(fused_names) != expected_names:
        missing = sorted(expected_names - set(fused_names))
        extra = sorted(set(fused_names) - expected_names)
        raise QuantizationError(
            f"fused routed-expert layout differs: found {len(fused_names)}, expected "
            f"{expected_fused}; missing={missing[:3]}, extra={extra[:3]}"
        )
    missing_shards = {value for value in weight_map.values() if not (source / value).is_file()}
    if missing_shards:
        raise QuantizationError(
            f"source is incomplete; {len(missing_shards)} indexed shards are absent"
        )
    return expected_projections


def make_synthetic_source(root: Path) -> tuple[Path, Path]:
    if root.exists() and any(root.iterdir()):
        raise QuantizationError(f"self-test directory must be empty: {root}")
    source = root / "source"
    output = root / "output"
    source.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(380032)
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(
            64, 64, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
    }
    tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"] = torch.randn(
        2, 128, 64, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    tensors["model.language_model.layers.0.mlp.experts.down_proj"] = torch.randn(
        2, 64, 64, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    # The serving launcher disables MTP initially; its experts must remain exact.
    tensors["mtp.layers.0.mlp.experts.down_proj"] = torch.randn(
        2, 64, 64, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    embedding_name = "model.language_model.embed_tokens.weight"
    embedding = tensors.pop(embedding_name)
    save_file(
        {embedding_name: embedding},
        source / shard_names[0],
        metadata={"format": "pt"},
    )
    save_file(tensors, source / shard_names[1], metadata={"format": "pt"})
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 1,
            "num_experts": 2,
        },
    }
    index = {
        "metadata": {
            "total_size": embedding.numel() * embedding.element_size()
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors.values()
            )
        },
        "weight_map": {
            embedding_name: shard_names[0],
            **{name: shard_names[1] for name in tensors},
        },
    }
    write_json_atomic(source / "config.json", config)
    write_json_atomic(source / "model.safetensors.index.json", index)
    return source, output


def verify_synthetic_output(source: Path, output: Path, group_size: int) -> None:
    if stat.S_IMODE(output.stat().st_mode) != 0o755:
        raise QuantizationError("self-test output directory is not mode 0755")
    bad_modes = {
        item.name: oct(stat.S_IMODE(item.stat().st_mode))
        for item in output.iterdir()
        if item.is_file() and stat.S_IMODE(item.stat().st_mode) != 0o644
    }
    if bad_modes:
        raise QuantizationError(f"self-test output file modes are not 0644: {bad_modes}")

    with safe_open(
        source / "model-00002-of-00002.safetensors", framework="pt", device="cpu"
    ) as original, safe_open(
        output / "model-00002-of-00002.safetensors", framework="pt", device="cpu"
    ) as converted:
        converted_names = set(converted.keys())
        for name in original.keys():
            fused_match = FUSED_EXPERT_RE.fullmatch(name)
            if fused_match is None:
                if not torch.equal(original.get_tensor(name), converted.get_tensor(name)):
                    raise QuantizationError(f"self-test changed untouched tensor {name}")
                continue
            layer = int(fused_match.group(1))
            source_tensor = original.get_tensor(name)
            for expert in range(source_tensor.shape[0]):
                if fused_match.group(2) == "gate_up_proj":
                    projection_names = ("gate_proj", "up_proj")
                else:
                    projection_names = ("down_proj",)
                for projection_name in projection_names:
                    prefix = (
                        f"model.language_model.layers.{layer}.mlp.experts."
                        f"{expert}.{projection_name}"
                    )
                    packed_name = f"{prefix}.weight_packed"
                    scale_name = f"{prefix}.weight_scale"
                    shape_name = f"{prefix}.weight_shape"
                    if not {packed_name, scale_name, shape_name}.issubset(converted_names):
                        raise QuantizationError(
                            f"self-test output is incomplete for {prefix}"
                        )
                    shape = converted.get_tensor(shape_name).tolist()
                    packed = converted.get_tensor(packed_name)
                    scale = converted.get_tensor(scale_name)
                    unpacked = unpack_from_int32(packed, 4, torch.Size(shape))
                    reconstructed = (
                        unpacked.to(torch.float32).reshape(
                            shape[0], shape[1] // group_size, group_size
                        )
                        * scale.to(torch.float32).unsqueeze(-1)
                    ).reshape(shape)
                    if not torch.isfinite(reconstructed).all():
                        raise QuantizationError(
                            f"self-test produced non-finite values for {prefix}"
                        )
    with safe_open(
        source / "model-00001-of-00002.safetensors", framework="pt", device="cpu"
    ) as original, safe_open(
        output / "model-00001-of-00002.safetensors", framework="pt", device="cpu"
    ) as converted:
        name = "model.language_model.embed_tokens.weight"
        if list(converted.keys()) != [name]:
            raise QuantizationError("self-test changed the untouched-only shard layout")
        if not torch.equal(original.get_tensor(name), converted.get_tensor(name)):
            raise QuantizationError("self-test changed the untouched embedding tensor")
    output_config = load_json(output / "config.json")
    quant_config = output_config.get("quantization_config") or {}
    actual_group = (
        quant_config.get("config_groups", {})
        .get("routed_experts", {})
        .get("weights", {})
        .get("group_size")
    )
    if actual_group != group_size:
        raise QuantizationError(
            f"self-test config group_size={actual_group}, expected {group_size}"
        )
    output_text_config = output_config.get("text_config") or output_config
    if output_text_config.get("ple_embedding_dtype") != PLE_RUNTIME_DTYPE:
        raise QuantizationError(
            "self-test config did not request memory-bounded FP8 PLE storage"
        )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16TritonMoE,
    )

    runtime_config = CompressedTensorsConfig.from_config(quant_config)
    untouched_scheme = runtime_config.get_scheme_dict(
        torch.nn.Linear(64, 64),
        layer_name="model.language_model.layers.0.linear_attn.in_proj_qkvz",
    )
    if untouched_scheme is not None:
        raise QuantizationError(
            "self-test config did not ignore an untouched linear layer"
        )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        expert_scheme = runtime_config.get_scheme_dict(
            torch.nn.Linear(64, 64),
            layer_name=(
                "model.language_model.layers.0.mlp.experts.0."
                f"{projection}"
            ),
        )
        if expert_scheme is None:
            raise QuantizationError(
                f"self-test config incorrectly ignored routed expert {projection}"
            )
    runtime_scheme = runtime_config.get_moe_scheme(
        torch.nn.Module(), layer_name="model.layers.0.mlp.experts"
    )
    if not isinstance(runtime_scheme, CompressedTensorsWNA16TritonMoE):
        raise QuantizationError(
            "self-test config did not resolve to CompressedTensorsWNA16TritonMoE"
        )
    if runtime_scheme.group_size != group_size:
        raise QuantizationError(
            f"runtime resolved group_size={runtime_scheme.group_size}, expected {group_size}"
        )

    import sglang

    mapping_source = (
        Path(sglang.__file__).parent
        / "srt/layers/moe/fused_moe_triton/layer.py"
    ).read_text(encoding="utf-8")
    for contract in (
        '"experts.w13_"',
        'else "experts.w2_"',
        'f"experts.{expert_id}.{weight_name}."',
    ):
        if contract not in mapping_source:
            raise QuantizationError(
                f"self-test SGLang expert mapping contract changed: {contract}"
            )
    packed_names = sorted(
        name
        for name in converted_names
        if ".mlp.experts." in name
        and name.endswith(("weight_packed", "weight_scale", "weight_shape"))
    )
    for checkpoint_name in packed_names:
        match = re.fullmatch(
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
            r"(gate_proj|up_proj|down_proj)\."
            r"(weight_packed|weight_scale|weight_shape)",
            checkpoint_name,
        )
        if match is None:
            raise QuantizationError(
                f"self-test output cannot map through FusedMoE: {checkpoint_name}"
            )
        target = "w2" if match.group(1) == "down_proj" else "w13"
        mapped_suffix = f"{target}_{match.group(2)}"
        if mapped_suffix not in {
            "w13_weight_packed",
            "w13_weight_scale",
            "w13_weight_shape",
            "w2_weight_packed",
            "w2_weight_scale",
            "w2_weight_shape",
        }:
            raise QuantizationError(
                f"self-test loader mapping is invalid for {checkpoint_name}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument(
        "--allow-non-spdk",
        action="store_true",
        help="permit an output outside /mnt/optane/models (only for synthetic tests)",
    )
    parser.add_argument(
        "--self-test-dir",
        type=Path,
        help="create and convert a tiny synthetic checkpoint in this empty directory",
    )
    parsed = parser.parse_args()

    try:
        self_test = parsed.self_test_dir is not None
        if self_test:
            source, output = make_synthetic_source(parsed.self_test_dir.resolve())
            parsed.allow_non_spdk = True
        else:
            if parsed.source is None or parsed.output is None:
                parser.error("--source and --output are required outside --self-test-dir")
            source = parsed.source.resolve()
            output = parsed.output.resolve()
        if source == output:
            raise QuantizationError("source and output directories must differ")
        if not source.is_dir():
            raise QuantizationError(f"source directory does not exist: {source}")
        if not parsed.allow_non_spdk and not output.is_relative_to("/mnt/optane/models"):
            raise QuantizationError("output must be under the fast /mnt/optane/models mount")

        config_path = source / "config.json"
        index_path = source / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise QuantizationError("source requires config.json and model.safetensors.index.json")
        config = load_json(config_path)
        index = load_json(index_path)
        expected_expert_tensors = validate_layout(source, index, config)

        plan = {
            "version": PLAN_VERSION,
            "source": str(source),
            "source_index_sha256": sha256(index_path),
            "group_size": parsed.group_size,
            "scale_dtype": "float16",
            "expected_expert_tensors": expected_expert_tensors,
        }
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / STATE_NAME
        if state_path.exists():
            state = load_json(state_path)
            if state.get("plan") != plan:
                raise QuantizationError(
                    f"existing output has a different immutable plan: {state_path}"
                )
        else:
            unexpected = [item.name for item in output.iterdir()]
            if unexpected:
                raise QuantizationError(
                    f"output is non-empty but has no resumable {STATE_NAME}: "
                    + ", ".join(sorted(unexpected)[:5])
                )
            state = {"plan": plan, "completed_shards": {}, "metrics": {}}
            write_json_atomic(state_path, state)

        quant_args = make_quant_args(parsed.group_size)
        source_shards = sorted({source / value for value in index["weight_map"].values()})
        for shard_number, source_shard in enumerate(source_shards, start=1):
            name = source_shard.name
            source_size = source_shard.stat().st_size
            recorded = state["completed_shards"].get(name)
            output_shard = output / name
            if recorded is not None:
                if (
                    recorded.get("source_size") != source_size
                    or not output_shard.is_file()
                    or output_shard.stat().st_size != recorded.get("output_size")
                ):
                    raise QuantizationError(f"resume record does not match shard {name}")
                print(f"[{shard_number}/{len(source_shards)}] resume {name}", flush=True)
                continue

            print(f"[{shard_number}/{len(source_shards)}] quantize {name}", flush=True)
            quantized, untouched, shard_metrics = quantize_shard(
                source_shard, output_shard, quant_args
            )
            state["completed_shards"][name] = {
                "source_size": source_size,
                "output_size": output_shard.stat().st_size,
                "quantized_tensors": quantized,
                "untouched_tensors": untouched,
                "metrics": shard_metrics,
            }
            aggregate: dict[str, float | int] = {}
            for completed in state["completed_shards"].values():
                merge_metrics(aggregate, completed["metrics"])
            state["metrics"] = aggregate
            write_json_atomic(state_path, state)

        actual_quantized = sum(
            int(value["quantized_tensors"])
            for value in state["completed_shards"].values()
        )
        if actual_quantized != expected_expert_tensors:
            raise QuantizationError(
                f"quantized {actual_quantized} expert tensors, expected {expected_expert_tensors}"
            )

        copy_ancillary_files(source, output)
        write_json_atomic(
            output / "config.json", make_output_config(config, parsed.group_size)
        )
        output_index, total_size = build_index(output)
        write_json_atomic(output / "model.safetensors.index.json", output_index)

        metrics = state["metrics"]
        relative_rmse = math.sqrt(
            float(metrics["squared_error"]) / float(metrics["source_energy"])
        )
        state["complete"] = True
        state["output_total_size"] = total_size
        state["relative_rmse"] = relative_rmse
        write_json_atomic(state_path, state)
        publish_output_modes(output)
        print(
            f"complete: {actual_quantized} routed expert tensors, "
            f"{total_size / 2**30:.3f} GiB logical, relative_rmse={relative_rmse:.8f}"
        )
        if self_test:
            verify_synthetic_output(source, output, parsed.group_size)
            print("synthetic pack/config/untouched-tensor self-test: passed")
        return 0
    except (OSError, QuantizationError, RuntimeError, ValueError) as exc:
        print(f"quantization error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
