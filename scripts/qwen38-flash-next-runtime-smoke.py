#!/usr/bin/env python3
"""Import the actual Qwen4-Exp MoE runtime from the packaged closure."""

from __future__ import annotations

import importlib
import json
import os
import sys

import torch


def main() -> None:
    os.environ.setdefault("SGLANG_USE_AITER", "0")

    imported = []
    for module_name in (
        "sglang.srt.layers.activation",
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
        "sglang.srt.layers.moe.fused_moe_triton.layer",
        "sglang.srt.constrained.xgrammar_backend",
        "sglang.srt.models.qwen4_exp",
    ):
        importlib.import_module(module_name)
        imported.append(module_name)

    if "sgl_kernel" in sys.modules:
        raise RuntimeError("gfx1030 runtime smoke unexpectedly imported sgl_kernel")

    from sglang.srt.layers.moe.topk import fused_topk_torch_native

    hidden_states = torch.zeros((2, 4), dtype=torch.float16)
    router_logits = torch.tensor(
        [[1.0, 3.0, 2.0, -1.0], [4.0, 0.0, 2.0, 1.0]], dtype=torch.float16
    )
    topk_weights, topk_ids = fused_topk_torch_native(
        hidden_states,
        router_logits,
        topk=2,
        renormalize=True,
        scoring_func="softmax",
    )
    if topk_weights.shape != (2, 2) or topk_ids.shape != (2, 2):
        raise RuntimeError("torch-native MoE top-k returned the wrong shape")
    if not torch.allclose(topk_weights.sum(dim=-1), torch.ones(2)):
        raise RuntimeError("torch-native MoE top-k did not renormalize")

    print(
        json.dumps(
            {
                "status": "ok",
                "aiter": os.environ["SGLANG_USE_AITER"],
                "imported": imported,
                "moe_topk": "torch_native",
                "sgl_kernel_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
