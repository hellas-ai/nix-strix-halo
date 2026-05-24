{
  lib,
  stdenv,
  cacert,
  gemma4-31b-it-text-config,
  vllm-rocm-lemonade,
  vllm-env-lemonade,
  packageSuffix,
  hsaOverrideGfxVersion ? "11.5.1",
}:

let
  vllm = vllm-rocm-lemonade;
  vllmCli = vllm-env-lemonade;
  rocmCore = "${vllm}/lib/python3.12/site-packages/_rocm_sdk_core";
  cacheRoot = "/tmp/vllm-lemonade-${packageSuffix}/gemma4-31b-q8";
in
stdenv.mkDerivation {
  pname = "vllm-lemonade-gemma4-31b-q8-kernel-cache-${packageSuffix}";
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
    export VLLM_CACHE_ROOT="${cacheRoot}"
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
    export PATH="${vllmCli}/bin:${vllm}/bin:${stdenv.cc}/bin:$PATH"
    export LD_PRELOAD="${vllm}/lib/libvllm-rocm-c10-hip-compat.so''${LD_PRELOAD:+ $LD_PRELOAD}"

    rm -rf "$VLLM_CACHE_ROOT"
    mkdir -p \
      "$HOME" \
      "$XDG_CACHE_HOME" \
      "$VLLM_CACHE_ROOT" \
      "$TRITON_CACHE_DIR" \
      "$TORCHINDUCTOR_CACHE_DIR" \
      "$VLLM_CONFIG_ROOT" \
      "$VLLM_XLA_CACHE_PATH"

    gguf_repo="$HF_HOME/hub/models--unsloth--gemma-4-31B-it-GGUF"
    gguf_rev="$(cat "$gguf_repo/refs/main")"
    model_path="$gguf_repo/snapshots/$gguf_rev/gemma-4-31B-it-Q8_0.gguf"
    tokenizer_repo="$HF_HOME/hub/models--unsloth--gemma-4-31B-it"
    tokenizer_rev="$(cat "$tokenizer_repo/refs/main")"
    tokenizer_path="$tokenizer_repo/snapshots/$tokenizer_rev"
    test -f "$model_path"
    test -f "$tokenizer_path/tokenizer.json"

    prime_gemma4_cache() {
      # Use the same vLLM wrapper as the runtime app. Triton launcher
      # cache keys include generated launcher source, so wrapper/env
      # drift can leave first-run launcher work outside the derivation.
      ${vllmCli}/bin/vllm bench throughput \
        --model "$model_path" \
        --tokenizer "$tokenizer_path" \
        --hf-config-path "${gemma4-31b-it-text-config}" \
        --trust-remote-code \
        --load-format gguf \
        --quantization gguf \
        --dataset-name random \
        --random-input-len 64 \
        --random-output-len 16 \
        --random-range-ratio 0 \
        --num-prompts 2 \
        --tensor-parallel-size 1 \
        --max-model-len 2048 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.85 \
        --dtype float16
    }

    prime_gemma4_cache
    prime_gemma4_cache

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
    description = "Primed vLLM/Triton kernel cache for Gemma4-31B Q8 smoke tests on Lemonade ROCm ${packageSuffix}";
    platforms = platforms.linux;
  };
}
