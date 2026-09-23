#!/usr/bin/env python3
"""Lower the production Qwen3.8-Flash-Next QSA decode kernel for gfx1030.

This is a compiler admission test, not a device-correctness or performance
claim.  It catches unsupported Triton IR and records the exact resource use
before the V620 host is powered.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

from sglang.srt.layers.attention.qsa.sparse_attn import (
    _sparse_gqa_decode_scores_physical,
    _sparse_gqa_decode_values_physical,
)


ARCH = "gfx1030"
HEAD_DIM = 256
NUM_KV_HEADS = 1
GROUP_SIZE = 6  # Qwen TP4: 24 query heads / four ranks, replicated KV head.
TOPK = 2048


def parse_resource(assembly: str, field: str) -> int:
    values = re.findall(rf"\.amdhsa_{re.escape(field)}\s+(\d+)", assembly)
    if len(values) != 1:
        raise RuntimeError(f"expected one .amdhsa_{field}, found {values!r}")
    return int(values[0])


def score_signature() -> dict[str, str]:
    signature = {
        "q": "*fp16",
        "k": "*fp16",
        "scores": "*fp32",
        "slots": "*i32",
        "scale": "fp32",
        "topk": "i32",
    }
    signature.update(
        {
            name: "constexpr"
            for name in (
                "sq_m",
                "sq_h",
                "sq_d",
                "sk_n",
                "sk_h",
                "sk_d",
                "sx_m",
                "sx_h",
                "sx_n",
                "ss_m",
                "ss_n",
                "GROUP_SIZE",
                "BLOCK_M",
                "BLOCK_N",
                "HEAD_DIM",
            )
        }
    )
    if tuple(_sparse_gqa_decode_scores_physical.arg_names) != tuple(signature):
        raise RuntimeError("QSA score kernel signature differs from AOT contract")
    return signature


def score_constants(block_n: int) -> dict[str, int]:
    # Contiguous [rows, heads, dim] q/out, [slots, kv_heads, dim] k/v,
    # and [rows, topk] physical-slot index tensors.
    return {
        "sq_m": 6 * HEAD_DIM,
        "sq_h": HEAD_DIM,
        "sq_d": 1,
        "sk_n": HEAD_DIM,
        "sk_h": HEAD_DIM,
        "sk_d": 1,
        "sx_m": GROUP_SIZE * TOPK,
        "sx_h": TOPK,
        "sx_n": 1,
        "ss_m": TOPK,
        "ss_n": 1,
        "GROUP_SIZE": GROUP_SIZE,
        "BLOCK_M": 16,
        "BLOCK_N": block_n,
        "HEAD_DIM": HEAD_DIM,
    }


def value_signature() -> dict[str, str]:
    signature = {
        "v": "*fp16",
        "out": "*fp16",
        "scores": "*fp32",
        "slots": "*i32",
        "topk": "i32",
    }
    signature.update(
        {
            name: "constexpr"
            for name in (
                "sv_n",
                "sv_h",
                "sv_d",
                "so_m",
                "so_h",
                "so_d",
                "sx_m",
                "sx_h",
                "sx_n",
                "ss_m",
                "ss_n",
                "GROUP_SIZE",
                "BLOCK_N",
                "BLOCK_D",
                "HEAD_DIM",
            )
        }
    )
    if tuple(_sparse_gqa_decode_values_physical.arg_names) != tuple(signature):
        raise RuntimeError("QSA value kernel signature differs from AOT contract")
    return signature


def value_constants(block_n: int, block_d: int) -> dict[str, int]:
    return {
        "sv_n": HEAD_DIM,
        "sv_h": HEAD_DIM,
        "sv_d": 1,
        "so_m": GROUP_SIZE * HEAD_DIM,
        "so_h": HEAD_DIM,
        "so_d": 1,
        "sx_m": GROUP_SIZE * TOPK,
        "sx_h": TOPK,
        "sx_n": 1,
        "ss_m": TOPK,
        "ss_n": 1,
        "GROUP_SIZE": GROUP_SIZE,
        "BLOCK_N": block_n,
        "BLOCK_D": block_d,
        "HEAD_DIM": HEAD_DIM,
    }


def compile_variant(
    kind: str, block_n: int, num_warps: int, num_stages: int, block_d: int = 0
) -> dict:
    if kind == "score":
        kernel = _sparse_gqa_decode_scores_physical
        signature = score_signature()
        constant_values = score_constants(block_n)
    elif kind == "value":
        kernel = _sparse_gqa_decode_values_physical
        signature = value_signature()
        constant_values = value_constants(block_n, block_d)
    else:
        raise ValueError(f"unknown kernel kind: {kind}")
    source = triton.compiler.ASTSource(
        fn=kernel,
        signature=signature,
        constexprs=constant_values,
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", ARCH, 32),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
            "waves_per_eu": 1,
        },
    )
    assembly = compiled.asm["amdgcn"]
    resources = {
        field: parse_resource(assembly, field)
        for field in (
            "next_free_vgpr",
            "next_free_sgpr",
            "group_segment_fixed_size",
            "private_segment_fixed_size",
        )
    }
    if resources["private_segment_fixed_size"] != 0:
        raise RuntimeError(f"{kind} kernel spills to scratch: {resources}")
    return {
        "kind": kind,
        "block_n": block_n,
        "block_d": block_d or None,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "kernel_name": compiled.metadata.name,
        "compiler_hash": compiled.metadata.hash,
        "resources": resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # These are the exact no-spill configurations used by the HIP wrapper.
    variants = [
        compile_variant("score", 32, 8, 2),
        compile_variant("value", 64, 4, 2, 64),
    ]
    manifest = {
        "schema": 1,
        "kind": "qwen38-flash-next-qsa-gfx1030-aot-admission",
        "device_execution": False,
        "correctness_claim": False,
        "performance_claim": False,
        "target": {"backend": "hip", "arch": ARCH, "warp_size": 32},
        "contract": {
            "head_dim": HEAD_DIM,
            "num_kv_heads_per_rank": NUM_KV_HEADS,
            "query_group_size": GROUP_SIZE,
            "indexer_budget": TOPK,
            "weight_dtype": "float16",
            "accumulator_dtype": "float32",
        },
        "triton": triton.__version__,
        "variants": variants,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
