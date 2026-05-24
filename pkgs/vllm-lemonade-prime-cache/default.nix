{
  writeShellApplication,
  coreutils,
  vllm-rocm-lemonade,
  packageSuffix,
}:

writeShellApplication {
  name = "vllm-lemonade-prime-cache-${packageSuffix}";
  runtimeInputs = [ coreutils ];
  text = ''
    set -euo pipefail

    model="''${1:-Qwen/Qwen3.6-27B}"
    if [ "$#" -gt 0 ]; then
      shift
    fi

    cache_root="''${VLLM_LEMONADE_CACHE_ROOT:-''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/vllm/lemonade-${packageSuffix}}"
    export VLLM_CACHE_ROOT="''${VLLM_CACHE_ROOT:-$cache_root}"
    export TRITON_CACHE_DIR="''${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton_cache}"
    export TORCHINDUCTOR_CACHE_DIR="''${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/inductor_cache}"
    export HF_HOME="''${HF_HOME:-/models/.cache/huggingface}"
    export VLLM_LOGGING_LEVEL="''${VLLM_LOGGING_LEVEL:-INFO}"
    export TRITON_CACHE_AUTOTUNING="''${TRITON_CACHE_AUTOTUNING:-1}"

    mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

    echo "Priming Lemonade vLLM ROCm cache"
    echo "  model:             $model"
    echo "  VLLM_CACHE_ROOT:   $VLLM_CACHE_ROOT"
    echo "  TRITON_CACHE_DIR:  $TRITON_CACHE_DIR"
    echo "  HF_HOME:           $HF_HOME"

    exec "${vllm-rocm-lemonade}/bin/vllm" bench throughput \
      --model "$model" \
      --dataset-name random \
      --num-prompts "''${VLLM_PRIME_NUM_PROMPTS:-1}" \
      --random-input-len "''${VLLM_PRIME_INPUT_LEN:-512}" \
      --random-output-len "''${VLLM_PRIME_OUTPUT_LEN:-1}" \
      --max-model-len "''${VLLM_PRIME_MAX_MODEL_LEN:-4096}" \
      --gpu-memory-utilization "''${VLLM_PRIME_GPU_MEMORY_UTILIZATION:-0.85}" \
      --trust-remote-code \
      --enforce-eager \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      "$@"
  '';
}
