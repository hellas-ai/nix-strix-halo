{
  writeShellApplication,
  coreutils,
  gemma4-31b-it-text-config,
  vllm-lemonade-gemma4-31b-q8,
  packageSuffix,
}:

writeShellApplication {
  name = "vllm-lemonade-gemma4-31b-q8-prime-${packageSuffix}";
  runtimeInputs = [ coreutils ];
  text = ''
    set -euo pipefail

    hf_home="''${HF_HOME:-/models/.cache/huggingface}"
    model="''${VLLM_GEMMA4_MODEL:-}"
    tokenizer="''${VLLM_GEMMA4_TOKENIZER:-}"
    if [ -z "$model" ]; then
      gguf_repo="$hf_home/hub/models--unsloth--gemma-4-31B-it-GGUF"
      if [ -r "$gguf_repo/refs/main" ]; then
        gguf_rev="$(cat "$gguf_repo/refs/main")"
        candidate="$gguf_repo/snapshots/$gguf_rev/gemma-4-31B-it-Q8_0.gguf"
        if [ -f "$candidate" ]; then
          model="$candidate"
        fi
      fi
    fi
    if [ -z "$model" ]; then
      model="unsloth/gemma-4-31B-it-GGUF:Q8_0"
    fi
    if [ -z "$tokenizer" ]; then
      tokenizer_repo="$hf_home/hub/models--unsloth--gemma-4-31B-it"
      if [ -r "$tokenizer_repo/refs/main" ]; then
        tokenizer_rev="$(cat "$tokenizer_repo/refs/main")"
        candidate="$tokenizer_repo/snapshots/$tokenizer_rev"
        if [ -f "$candidate/tokenizer.json" ]; then
          tokenizer="$candidate"
        fi
      fi
    fi
    if [ -z "$tokenizer" ]; then
      tokenizer="unsloth/gemma-4-31B-it"
    fi

    num_prompts="''${VLLM_GEMMA4_PRIME_NUM_PROMPTS:-2}"
    input_len="''${VLLM_GEMMA4_PRIME_INPUT_LEN:-64}"
    output_len="''${VLLM_GEMMA4_PRIME_OUTPUT_LEN:-16}"
    tp="''${VLLM_GEMMA4_PRIME_TP:-1}"
    max_model_len="''${VLLM_GEMMA4_PRIME_MAX_MODEL_LEN:-2048}"
    max_num_seqs="''${VLLM_GEMMA4_PRIME_MAX_NUM_SEQS:-1}"
    max_batched_tokens="''${VLLM_GEMMA4_PRIME_MAX_NUM_BATCHED_TOKENS:-2048}"
    gpu_memory_utilization="''${VLLM_GEMMA4_PRIME_GPU_MEMORY_UTILIZATION:-0.85}"
    warmup_runs="''${VLLM_GEMMA4_PRIME_WARMUP_RUNS:-0}"
    unset \
      VLLM_GEMMA4_PRIME_NUM_PROMPTS \
      VLLM_GEMMA4_PRIME_INPUT_LEN \
      VLLM_GEMMA4_PRIME_OUTPUT_LEN \
      VLLM_GEMMA4_PRIME_TP \
      VLLM_GEMMA4_PRIME_MAX_MODEL_LEN \
      VLLM_GEMMA4_PRIME_MAX_NUM_SEQS \
      VLLM_GEMMA4_PRIME_MAX_NUM_BATCHED_TOKENS \
      VLLM_GEMMA4_PRIME_GPU_MEMORY_UTILIZATION \
      VLLM_GEMMA4_PRIME_WARMUP_RUNS \
      VLLM_GEMMA4_MODEL \
      VLLM_GEMMA4_TOKENIZER

    run_prime() {
      "${vllm-lemonade-gemma4-31b-q8}/bin/vllm-lemonade-gemma4-31b-q8-${packageSuffix}" \
        bench throughput \
        --model "$model" \
        --tokenizer "$tokenizer" \
        --hf-config-path "${gemma4-31b-it-text-config}" \
        --trust-remote-code \
        --load-format gguf \
        --quantization gguf \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio 0 \
        --num-prompts "$num_prompts" \
        --tensor-parallel-size "$tp" \
        --max-model-len "$max_model_len" \
        --max-num-seqs "$max_num_seqs" \
        --max-num-batched-tokens "$max_batched_tokens" \
        --gpu-memory-utilization "$gpu_memory_utilization" \
        --dtype float16 \
        "$@"
    }

    run=0
    while [ "$run" -lt "$warmup_runs" ]; do
      run_prime "$@" >/dev/null 2>&1
      run=$((run + 1))
    done

    run_prime "$@"
  '';
}
