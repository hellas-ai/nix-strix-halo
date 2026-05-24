{
  lib,
  stdenv,
  cacert,
  vllm-rocm-lemonade,
  packageSuffix,
  hsaOverrideGfxVersion ? "11.5.1",
}:

let
  vllm = vllm-rocm-lemonade;
  rocmCore = "${vllm}/lib/python3.12/site-packages/_rocm_sdk_core";
in
stdenv.mkDerivation {
  pname = "vllm-lemonade-qwen36-27b-cache-${packageSuffix}";
  version = vllm.version;

  dontUnpack = true;
  dontConfigure = true;
  dontStrip = true;
  dontFixup = true;

  requiredSystemFeatures = [ "${packageSuffix}" ];
  preferLocalBuild = true;
  allowSubstitutes = false;

  buildPhase = ''
    runHook preBuild

    export HOME="$TMPDIR/home"
    export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
    export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"
    export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/inductor_cache"
    export HF_HOME="/models/.cache/huggingface"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export SSL_CERT_FILE="${cacert}/etc/ssl/certs/ca-bundle.crt"
    export VLLM_DISABLE_AITER=1
    export VLLM_CONFIG_ROOT="/tmp/vllm-lemonade-${packageSuffix}/config"
    export VLLM_XLA_CACHE_PATH="/tmp/vllm-lemonade-${packageSuffix}/xla_cache"
    export VLLM_LOGGING_LEVEL="INFO"
    export TRITON_CACHE_AUTOTUNING="1"
    export HSA_OVERRIDE_GFX_VERSION="${hsaOverrideGfxVersion}"
    export HSA_NO_SCRATCH_RECLAIM="1"
    export HSA_ENABLE_INTERRUPT="0"
    export HIP_PLATFORM="amd"
    export GPU_ARCHS="${packageSuffix}"
    export PYTORCH_ROCM_ARCH="${packageSuffix}"
    export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
    export ROCM_HOME="${rocmCore}"
    export ROCM_PATH="$ROCM_HOME"
    export HIP_PATH="$ROCM_HOME"
    export TRITON_LIBHIP_PATH="$ROCM_HOME/lib/libamdhip64.so"
    export DEVICE_LIB_PATH="${rocmCore}/lib/llvm/amdgcn/bitcode"
    export HIP_DEVICE_LIB_PATH="$DEVICE_LIB_PATH"
    export CC="${stdenv.cc}/bin/cc"
    export CXX="${stdenv.cc}/bin/c++"
    export PATH="${vllm}/bin:${stdenv.cc}/bin:$PATH"
    export LD_PRELOAD="${vllm}/lib/libvllm-rocm-c10-hip-compat.so''${LD_PRELOAD:+ $LD_PRELOAD}"

    mkdir -p \
      "$HOME" \
      "$XDG_CACHE_HOME" \
      "$VLLM_CACHE_ROOT" \
      "$TRITON_CACHE_DIR" \
      "$TORCHINDUCTOR_CACHE_DIR" \
      "$VLLM_CONFIG_ROOT" \
      "$VLLM_XLA_CACHE_PATH"

    ${vllm}/bin/python3 -u - <<'PY'
    import math
    import torch

    from vllm.model_executor.layers.fla.ops import (
        chunk_gated_delta_rule,
        fused_post_conv_prep,
    )
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from vllm.model_executor.layers.fla.ops.layernorm_guard import (
        layer_norm_fwd,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        triton_mrope,
    )
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode,
    )
    from vllm.v1.worker.block_table import BlockTable
    from vllm.v1.worker.utils import _zero_kv_blocks_kernel

    device = "cuda"
    dtype = torch.bfloat16
    seq_len = FLA_CHUNK_SIZE

    # Qwen3.6-27B GDN/linear-attention dimensions from config.json.
    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = 128
    head_v_dim = 128
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

    print("priming qwen3.6 gdn prefill kernels", flush=True)
    a_log = torch.zeros(num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(num_v_heads, device=device, dtype=torch.float32)
    mixed_qkv = torch.randn(seq_len, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(seq_len, num_v_heads, device=device, dtype=dtype)
    b = torch.randn(seq_len, num_v_heads, device=device, dtype=dtype)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )

    state = torch.zeros(
        1,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        device=device,
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    out, final_state = chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    print(
        "primed qwen3.6 gdn kernels",
        tuple(out.shape),
        None if final_state is None else tuple(final_state.shape),
        flush=True,
    )

    print("priming qwen3.6 recurrent decode kernel", flush=True)
    packed_out = torch.empty(
        1, 1, num_v_heads, head_v_dim, device=device, dtype=dtype
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv[:1].contiguous(),
        a[:1].contiguous(),
        b[:1].contiguous(),
        a_log,
        dt_bias,
        1.0 / math.sqrt(head_k_dim),
        torch.zeros(
            2,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device=device,
            dtype=torch.float32,
        ),
        packed_out,
        torch.tensor([1], device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )

    print("priming qwen3.6 causal-conv kernels", flush=True)
    conv_dim = qkv_dim
    conv_width = 4
    conv_weight = torch.randn(
        conv_dim, conv_width, device=device, dtype=dtype
    )
    conv_bias = torch.zeros(conv_dim, device=device, dtype=dtype)
    conv_state = torch.zeros(
        4,
        conv_dim,
        conv_width - 1,
        device=device,
        dtype=torch.float32,
    )
    cache_indices = torch.tensor([1], device=device, dtype=torch.int32)
    causal_conv1d_fn(
        torch.randn(512, conv_dim, device=device, dtype=dtype).t(),
        conv_weight,
        conv_bias,
        conv_state,
        torch.tensor([0, 512], device=device, dtype=torch.int32),
        cache_indices=cache_indices,
        has_initial_state=torch.tensor(
            [False], device=device, dtype=torch.bool
        ),
        activation="silu",
    )
    causal_conv1d_update(
        torch.randn(1, conv_dim, device=device, dtype=dtype),
        conv_state,
        conv_weight,
        conv_bias,
        activation="silu",
        conv_state_indices=cache_indices,
    )

    print("priming qwen3.6 attention prefill/decode kernels", flush=True)
    num_query_heads = 24
    num_kv_heads = 4
    attn_head_dim = 256
    physical_block = 784
    cache_x = 16
    key_cache = torch.randn(
        2,
        num_kv_heads,
        attn_head_dim // cache_x,
        physical_block,
        cache_x,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn(
        2,
        num_kv_heads,
        attn_head_dim,
        physical_block,
        device=device,
        dtype=dtype,
    )
    block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([512], device=device, dtype=torch.int32)
    k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    for query_len in (512, 1):
        query = torch.randn(
            query_len,
            num_query_heads,
            attn_head_dim,
            device=device,
            dtype=dtype,
        )
        chunked_prefill_paged_decode(
            query,
            torch.randn(
                query_len,
                num_kv_heads,
                attn_head_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                query_len,
                num_kv_heads,
                attn_head_dim,
                device=device,
                dtype=dtype,
            ),
            torch.empty_like(query),
            "auto",
            key_cache,
            value_cache,
            block_table,
            torch.tensor([0, query_len], device=device, dtype=torch.int32),
            seq_lens,
            512,
            query_len,
            k_scale,
            v_scale,
            sm_scale=1.0 / math.sqrt(attn_head_dim),
        )

    print("priming qwen3.6 worker helper kernels", flush=True)
    zero_buf = torch.empty(2048, device=device, dtype=torch.int32)
    _zero_kv_blocks_kernel[(1,)](
        torch.tensor([zero_buf.data_ptr()], device=device, dtype=torch.uint64),
        torch.tensor([0], device=device, dtype=torch.int64),
        1,
        N_SEGS=1,
        PAGE_SIZE_EL=1024,
        BLOCK_SIZE=1024,
    )
    block_table_helper = BlockTable(
        16,
        1,
        64,
        512,
        False,
        torch.device(device),
        16,
        1,
    )
    block_table_helper.add_row(list(range(32)), 0)
    block_table_helper.commit_block_table(1)
    block_table_helper.compute_slot_mapping(
        1,
        torch.tensor([0, 512], device=device, dtype=torch.int32),
        torch.arange(512, device=device, dtype=torch.int64),
    )

    print("priming qwen3.6 norm and mrope kernels", flush=True)
    layer_norm_fwd(
        torch.randn(1, 5120, device=device, dtype=dtype),
        torch.ones(5120, device=device, dtype=dtype),
        None,
        1e-6,
        z=torch.randn(1, 5120, device=device, dtype=dtype),
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
    )
    layer_norm_fwd(
        torch.randn(1, 128, device=device, dtype=dtype),
        torch.ones(128, device=device, dtype=dtype),
        None,
        1e-6,
        group_size=128,
        is_rms_norm=True,
    )
    triton_mrope(
        torch.randn(
            512,
            num_query_heads * attn_head_dim,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            512,
            num_kv_heads * attn_head_dim,
            device=device,
            dtype=dtype,
        ),
        torch.randn(3, 512, 32, device=device, dtype=dtype),
        torch.randn(3, 512, 32, device=device, dtype=dtype),
        [11, 11, 10],
        attn_head_dim,
        64,
        True,
    )
    torch.cuda.synchronize()
    print("primed qwen3.6 decode/runtime kernels", flush=True)
    PY

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -R "$VLLM_CACHE_ROOT"/. "$out"/
    find "$out" -type f | sort > "$out/manifest.txt"

    runHook postInstall
  '';

  meta = with lib; {
    description = "Primed vLLM/Triton cache for Qwen3.6-27B on Lemonade ROCm ${packageSuffix}";
    platforms = platforms.linux;
  };
}
