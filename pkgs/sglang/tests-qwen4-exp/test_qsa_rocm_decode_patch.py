#!/usr/bin/env python3
"""Static contract for the Qwen4-Exp ROCm QSA decode boot fix."""

from __future__ import annotations

import ast
import os
import textwrap
from pathlib import Path

import torch


root = Path(os.environ["SGLANG_ROOT_UNDER_TEST"])
server_args_source = (root / "sglang/srt/server_args.py").read_text(encoding="utf-8")
assert '"Qwen4ExpForConditionalGeneration",' in server_args_source

source_path = root / "sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
source = source_path.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(source_path))

backend = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "QwenSparseAttnBackend"
)
paged = next(
    node
    for node in backend.body
    if isinstance(node, ast.FunctionDef) and node.name == "_forward_paged_attention"
)
paged_source = ast.get_source_segment(source, paged)
assert paged_source is not None
assert "if q.is_cuda and torch.version.hip is not None:" in paged_source
assert "slots = self._logical_to_physical(topk_indices, metadata)" in paged_source
assert "sparse_gqa_decode_physical_triton(" in paged_source

sparse_source = (
    root / "sglang/srt/layers/attention/qsa/sparse_attn.py"
).read_text(encoding="utf-8")
assert "def _sparse_gqa_decode_scores_physical(" in sparse_source
assert "def _sparse_gqa_decode_values_physical(" in sparse_source
assert "def sparse_gqa_decode_physical_triton(" in sparse_source
assert "scores = torch.empty(" in sparse_source
assert "accumulator = tl.zeros([BLOCK_D], tl.float32)" in sparse_source
assert "has_values = next_max > -float(\"inf\")" in sparse_source

# The CUDA implementation remains available for supported NVIDIA systems; the
# patch must not pretend that FA2/FA4 itself became portable.
assert "flash_attn_varlen_func = _resolve_flash_attn_varlen_func()" in paged_source

quant_root = root / "sglang/srt/layers/quantization"
mxfp4_source = (quant_root / "mxfp4.py").read_text(encoding="utf-8")
registry_source = (quant_root / "__init__.py").read_text(encoding="utf-8")
diffusion_mxfp4_source = (
    root / "sglang/multimodal_gen/runtime/layers/quantization/mxfp4.py"
).read_text(encoding="utf-8")
assert "if _use_aiter:\n    # import aiter" in mxfp4_source
assert '_quark_available = not (is_hip() and _rocm_arch.startswith("gfx10"))' in registry_source
assert "if _is_hip and is_gfx95_supported():" in diffusion_mxfp4_source

activation_source = (root / "sglang/srt/layers/activation.py").read_text(
    encoding="utf-8"
)
moe_runner_source = (
    root
    / "sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
).read_text(encoding="utf-8")
triton_moe_source = (
    root / "sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py"
).read_text(encoding="utf-8")
assert "elif _is_hip:\n    from sglang.kernels.ops.activation.activation import (" in activation_source
assert "def gelu_quick(input: torch.Tensor, out=None)" in activation_source
assert "elif _is_hip:\n    from sglang.kernels.ops.activation.activation import gelu_and_mul, silu_and_mul" in moe_runner_source
assert "if is_cuda() or is_hip():" in triton_moe_source

wna16_source = (
    quant_root
    / "compressed_tensors/schemes/compressed_tensors_wNa16_moe.py"
).read_text(encoding="utf-8")
assert "def transpose_parameter_on_host(" in wna16_source
assert "parameter.data = parameter.data.new_empty(0)" in wna16_source
assert 'setattr(layer, name, None)' in wna16_source
assert "torch.cuda.empty_cache()" in wna16_source
assert 'transpose_parameter_on_host("w13_weight_packed", view_uint8=True)' in wna16_source
assert 'transpose_parameter_on_host("w2_weight_scale")' in wna16_source

# Exercise the actual patched method body on CPU without importing SGLang's
# quantization package, whose import probes for a physical GPU.  Keep external
# references to the original Parameters to reproduce the loader bookkeeping
# that caused the live V620 peak: those objects must survive but their
# GPU-sized storage must not.
wna16_tree = ast.parse(wna16_source)
wna16_class = next(
    node
    for node in wna16_tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "CompressedTensorsWNA16TritonMoE"
)
wna16_method = next(
    node
    for node in wna16_class.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "process_weights_after_loading"
)
wna16_method_source = ast.get_source_segment(wna16_source, wna16_method)
assert wna16_method_source is not None
wna16_namespace = {"torch": torch}
exec(
    "class _PatchedWNA16Scheme:\n"
    + textwrap.indent(textwrap.dedent(wna16_method_source), "    "),
    wna16_namespace,
)


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w13_weight_packed = torch.nn.Parameter(
            torch.arange(24, dtype=torch.int32).reshape(2, 3, 4),
            requires_grad=False,
        )
        self.w2_weight_packed = torch.nn.Parameter(
            (torch.arange(24, dtype=torch.int32) + 100).reshape(2, 3, 4),
            requires_grad=False,
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            requires_grad=False,
        )
        self.w2_weight_scale = torch.nn.Parameter(
            (torch.arange(24, dtype=torch.float32) + 100).reshape(2, 3, 4),
            requires_grad=False,
        )


layer = _FakeLayer()
names = (
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
)
old_parameters = {name: getattr(layer, name) for name in names}
expected = {
    name: parameter.detach().transpose(1, 2).contiguous()
    for name, parameter in old_parameters.items()
}
expected["w13_weight_packed"] = expected["w13_weight_packed"].view(torch.uint8)
expected["w2_weight_packed"] = expected["w2_weight_packed"].view(torch.uint8)

empty_cache = torch.cuda.empty_cache
torch.cuda.empty_cache = lambda: None
try:
    scheme = wna16_namespace["_PatchedWNA16Scheme"]()
    scheme.process_weights_after_loading(layer)
    first_parameters = {name: getattr(layer, name) for name in names}
    scheme.process_weights_after_loading(layer)
finally:
    torch.cuda.empty_cache = empty_cache

assert layer.is_triton_converted
for name in names:
    assert old_parameters[name].numel() == 0
    assert torch.equal(first_parameters[name], expected[name])
    assert getattr(layer, name) is first_parameters[name]

xgrammar_source = (
    root / "sglang/srt/constrained/xgrammar_backend.py"
).read_text(encoding="utf-8")
assert "from sgl_kernel import apply_token_bitmask_inplace_cuda" not in xgrammar_source
assert "_is_hip = is_hip()" in xgrammar_source
assert xgrammar_source.count("apply_token_bitmask_inplace_triton(logits, vocab_mask)") == 2

hyperconnection_source = (
    root / "sglang/srt/layers/hyperconnection.py"
).read_text(encoding="utf-8")
assert "from sglang.srt.utils import is_cuda" in hyperconnection_source
assert (
    "envs.SGLANG_HC_MIX_CUDA.get()\n"
    "                and is_cuda()\n"
    "                and torch.cuda.is_available()"
) in hyperconnection_source

hc_mix_source = (
    root / "sglang/kernels/ops/elementwise/hc_mix.py"
).read_text(encoding="utf-8")
assert "def hc_mix(" in hc_mix_source
assert "flashinfer_pr4266_dense_bf16_gemm_sm100_splitk import (" in hc_mix_source

qwen4_source = (root / "sglang/srt/models/qwen4_exp.py").read_text(
    encoding="utf-8"
)
assert qwen4_source.count("params_dtype=torch.get_default_dtype(),") == 2
assert "output_dtype=torch.get_default_dtype()," in qwen4_source
assert '"output_dtype",' in qwen4_source
assert "dtype=self.output_dtype" in qwen4_source
assert "out.dtype != self.output_dtype" in qwen4_source
assert "weight_scale\", torch.ones(1, dtype=torch.get_default_dtype())" in qwen4_source

topk_source = (root / "sglang/srt/layers/moe/topk.py").read_text(encoding="utf-8")
assert "elif _is_hip:\n            topk_weights, topk_ids = fused_topk_torch_native(" in topk_source
assert "scoring_func=scoring_func," in topk_source

moe_align_source = (
    root
    / "sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py"
).read_text(encoding="utf-8")
assert "if _is_cuda or _is_xpu or _is_musa:" in moe_align_source
assert "elif _is_hip:" in moe_align_source
assert "moe_align_block_size as jit_moe_align_block_size" in moe_align_source
assert "if ignore_invalid_expert:" in moe_align_source
assert "jit_moe_align_block_size(" in moe_align_source

eagle_source = (root / "sglang/srt/speculative/eagle_utils.py").read_text(
    encoding="utf-8"
)
assert "if _is_cuda or _is_hip or _is_musa:" not in eagle_source
assert eagle_source.count("elif _is_xpu or _is_hip:") == 2

hisparse_source = (root / "sglang/srt/mem_cache/hisparse_memory_pool.py").read_text(
    encoding="utf-8"
)
assert "try:\n    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla" in hisparse_source
assert 'except ImportError:' in hisparse_source

scheduler_source = (root / "sglang/srt/managers/scheduler.py").read_text(
    encoding="utf-8"
)
assert scheduler_source.count(
    "from sglang.srt.disaggregation.decode_kvcache_offload_manager import ("
) == 1
assert (
    "\n            from sglang.srt.disaggregation."
    "decode_kvcache_offload_manager import ("
) in scheduler_source
assert "from sglang.srt.managers.hisparse_coordinator import" not in scheduler_source
assert "self.hisparse_coordinator: Optional[Any] = None" in scheduler_source

host_pool_source = (root / "sglang/srt/mem_cache/memory_pool_host.py").read_text(
    encoding="utf-8"
)
assert "except ImportError:\n    def _missing_kvcacheio" in host_pool_source
assert "transfer_kv_per_layer_mla_pf_lf = _missing_kvcacheio" in host_pool_source

hybrid_controller_source = (
    root / "sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
).read_text(encoding="utf-8")
assert hybrid_controller_source.count(
    "from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost"
) == 1
assert (
    "\n            from sglang.srt.mem_cache.pool_host.mha import "
    "MHATokenToKVPoolHost"
) in hybrid_controller_source

spec_utils_source = (root / "sglang/srt/speculative/spec_utils.py").read_text(
    encoding="utf-8"
)
assert "elif _is_hip:\n    from sgl_kernel import fast_topk" not in spec_utils_source

rope_source = (
    root / "sglang/srt/layers/rotary_embedding/base.py"
).read_text(encoding="utf-8")
assert "except ImportError:\n                    self.use_fallback_kernel = False" in rope_source
assert "self._forward_method = self.forward_native" in rope_source
assert "if rotary_embedding is not None:" in rope_source
