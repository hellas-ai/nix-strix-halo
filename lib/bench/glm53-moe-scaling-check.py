#!/usr/bin/env python3
"""Compare the complete ROCm W4 MoE module to independent CPU math.

Run with the SGLang Python environment and SGLANG_USE_AITER=0. This covers
routing, clamping, quantized experts, the shared expert and output scaling;
it does not load checkpoint weights or establish full-model quality.
"""

import argparse
import json

import torch
import torch.nn.functional as F
from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model-path",
    default="/mnt/glm53-fabric/GLM-5.3-Flash-AWQ-W4A16",
    help="Local model configuration; no checkpoint weights are loaded",
)
cli = parser.parse_args()
assert torch.version.hip, "This regression exercises ROCm"
torch.set_num_threads(4)
torch.manual_seed(53)
args = ServerArgs(
    model_path=cli.model_path,
    attention_backend="dsa",
    dsa_prefill_backend="triton",
    dsa_decode_backend="triton",
    kv_cache_dtype="bfloat16",
    linear_attn_backend="triton",
    moe_runner_backend="triton",
    disable_shared_experts_fusion=True,
    disable_cuda_graph=True,
    disable_radix_cache=True,
)
publish(args, role="test")
initialize_moe_config()
init_distributed_environment(
    world_size=1,
    rank=0,
    local_rank=0,
    distributed_init_method="tcp://127.0.0.1:29559",
    backend="nccl",
)
initialize_model_parallel(tensor_model_parallel_size=1)
e, k, n, g = 16, 4096, 512, 128
config = Glm5NextTextConfig(
    hidden_size=k,
    moe_intermediate_size=n,
    n_routed_experts=e,
    num_experts_per_tok=8,
    n_shared_experts=1,
    swiglu_limit=10,
    routed_scaling_factor=2.5,
)
qc = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp.experts.*"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
        }
    },
    "ignore": ["re:.*shared_experts.*"],
}
quant = CompressedTensorsConfig.from_config(qc)
with torch.device("cuda"):
    torch.set_default_dtype(torch.bfloat16)
    m = DeepseekV2MoE(
        config, 3, quant, prefix="model.layers.3.mlp", alt_stream=torch.cuda.Stream()
    )
    torch.set_default_dtype(torch.float32)


def weight(rows, cols):
    vals = torch.randint(-8, 8, (e, rows, cols), dtype=torch.int32)
    scales = (torch.rand(e, rows, cols // g) * 0.006 + 0.002).bfloat16()
    packed = (
        (((vals + 8).reshape(e, rows, cols // 8, 8).long()) << (torch.arange(8) * 4))
        .sum(-1)
        .int()
    )
    ref = (vals.float() * scales.float().repeat_interleave(g, -1)).bfloat16().float()
    return packed.transpose(1, 2).contiguous(), scales.transpose(1, 2).contiguous(), ref


p1, s1, w1 = weight(n * 2, k)
p2, s2, w2 = weight(k, n)
layer = m.experts
with torch.no_grad():
    layer.w13_weight_packed.copy_(p1)
    layer.w2_weight_packed.copy_(p2)
    layer.w13_weight_scale.copy_(s1)
    layer.w2_weight_scale.copy_(s2)
    router = (torch.randn(e, k) * 0.01).bfloat16()
    bias = torch.randn(e) * 0.02
    shared1 = (torch.randn(n * 2, k) * 0.015).bfloat16()
    shared2 = (torch.randn(k, n) * 0.015).bfloat16()
    m.gate.weight.copy_(router)
    m.gate.e_score_correction_bias.copy_(bias)
    m.shared_experts.gate_up_proj.weight.copy_(shared1)
    m.shared_experts.down_proj.weight.copy_(shared2)
    layer.quant_method.process_weights_after_loading(layer)


def act(y):
    a, b = y.chunk(2, dim=-1)
    return (
        (F.silu(a.clamp(max=10)).bfloat16().float() * b.clamp(-10, 10))
        .bfloat16()
        .float()
    )


for rows in (1, 7, 33):
    x = (torch.randn(rows, k) * 3).bfloat16()
    logits = (x.float() @ router.float().T).bfloat16().float()
    scores = logits.sigmoid()
    ids = (scores + bias).topk(8, dim=-1).indices
    route = scores.gather(1, ids)
    route /= route.sum(-1, keepdim=True)
    ref = torch.zeros(rows, k)
    for expert in range(e):
        idx, slot = torch.where(ids == expert)
        if not len(idx):
            continue
        y = act((x[idx].float() @ w1[expert].T).bfloat16().float()) @ w2[expert].T
        ref[idx] += (y * route[idx, slot, None]).bfloat16().float()
    ref = (ref * 2.5).bfloat16().float()
    shared = (
        (act((x.float() @ shared1.float().T).bfloat16().float()) @ shared2.float().T)
        .bfloat16()
        .float()
    )
    ref = (ref + shared).bfloat16().float()
    for method in (m.forward_normal, m.forward_normal_dual_stream):
        # The multi-stream path consumes the original input for the shared
        # expert after running routed experts, requiring an out-of-place sum.
        # Serving leaves ROCm multi-stream disabled; check scaling in both
        # methods without relying on that separate input-aliasing issue.
        m.experts.moe_runner_config.inplace = method == m.forward_normal
        with torch.no_grad():
            got = method(x.cuda()).cpu().float()
        error = float((got - ref).norm() / ref.norm())
        print(
            json.dumps(
                {
                    "method": method.__name__,
                    "tokens": rows,
                    "relative_l2": error,
                    "max_abs": float((got - ref).abs().max()),
                }
            ),
            flush=True,
        )
        assert error < 0.02, (
            "Routed expert scaling, routing, shared expert, or W4 math mismatch"
        )
torch.distributed.destroy_process_group()
