#!/usr/bin/env bash
# Run one candidate DeepSeek V4.1 Flash TP4 rank on an already configured Strix.
set -euo pipefail

: "${DS41_NODE_RANK:?set node rank: strix-3=0, strix-4=1, strix-2=2, strix-1=3}"
: "${DS41_BINARY:?set the built sglang executable, e.g. /nix/store/.../bin/sglang}"
DS41_HEAD_ADDR=${DS41_HEAD_ADDR:-192.168.25.103:51041}
DS41_MODEL_PATH=${DS41_MODEL_PATH:-/mnt/ds41-models-rdma/DeepSeek-V4.1-Flash-hf-dba1be0a}
DS41_PORT=${DS41_PORT:-31041}
DS41_CHUNKED_PREFILL_SIZE=${DS41_CHUNKED_PREFILL_SIZE-1536}
DS41_PREFILL_DECODE_INTERVAL=${DS41_PREFILL_DECODE_INTERVAL-32}
DS41_FP4_LOGITS_BUDGET_MB=${DS41_FP4_LOGITS_BUDGET_MB-64}

case "$DS41_NODE_RANK" in 0|1|2|3) ;; *) echo 'invalid DS41_NODE_RANK' >&2; exit 2 ;; esac
[[ "$DS41_BINARY" = /* && -x "$DS41_BINARY" ]] || {
  echo 'DS41_BINARY must be an executable absolute path' >&2; exit 2;
}
[[ "$DS41_PORT" =~ ^[0-9]+$ ]] && (( 10#$DS41_PORT >= 1 && 10#$DS41_PORT <= 65535 )) || {
  echo 'DS41_PORT must be in 1..65535' >&2; exit 2;
}
[[ "$DS41_CHUNKED_PREFILL_SIZE" =~ ^[0-9]+$ ]] &&
  (( 10#$DS41_CHUNKED_PREFILL_SIZE > 0 )) || {
  echo 'DS41_CHUNKED_PREFILL_SIZE must be a positive integer' >&2; exit 2;
}
[[ "$DS41_PREFILL_DECODE_INTERVAL" =~ ^[0-9]+$ ]] &&
  (( 10#$DS41_PREFILL_DECODE_INTERVAL > 0 )) || {
  echo 'DS41_PREFILL_DECODE_INTERVAL must be a positive integer' >&2; exit 2;
}
case "$DS41_FP4_LOGITS_BUDGET_MB" in 64|128|256|384|512) ;; *)
  echo 'DS41_FP4_LOGITS_BUDGET_MB must be 64, 128, 256, 384, or 512' >&2; exit 2 ;;
esac
# Opt-in decoder SWA bounded replay (default off): after the last KV-source layer
# the remaining layers of each prefill chunk run on its last 128 rows only, which
# is the runtime's --enable-decoder-swa-bounded-replay. Late-layer sliding-window
# K/V of a chunk's earlier tail rows then differ from the full prefill, so cached
# objects written with and without it must never mix: pair it with a new Mooncake
# extra_backend_tag. Prompt logprobs (logprob_start_len below the prompt) are
# refused by the runtime under it.
DS41_DECODER_SWA_BOUNDED_REPLAY=${DS41_DECODER_SWA_BOUNDED_REPLAY-0}
case "$DS41_DECODER_SWA_BOUNDED_REPLAY" in 0|1) ;; *)
  echo 'DS41_DECODER_SWA_BOUNDED_REPLAY must be 0 or 1' >&2; exit 2 ;;
esac
replay_args=()
if [[ "$DS41_DECODER_SWA_BOUNDED_REPLAY" == 1 ]]; then
  replay_args=(--enable-decoder-swa-bounded-replay)
fi
[[ -n "$DS41_HEAD_ADDR" ]] || { echo 'DS41_HEAD_ADDR is empty' >&2; exit 2; }
DS41_CACHE_ROOT=${DS41_CACHE_ROOT:-/tmp/ds41-compiler-cache/gfx1151}
[[ "$DS41_CACHE_ROOT" = /* ]] || {
  echo 'DS41_CACHE_ROOT must be an absolute writable directory' >&2; exit 2;
}

# Opt-in, bounded DSpark profiles. Keep the ordinary decode profile unchanged.
# Request capacity is limited so target verification uses at most eight rows.
DS41_DSPARK_POLICY=${DS41_DSPARK_POLICY:-fixed}
case "$DS41_DSPARK_POLICY" in
  fixed) unset SGLANG_DSV41_DSPARK_ADAPTIVE_VERIFY ;;
  adaptive)
    [[ "${DS41_DSPARK_GAMMA:-3}" == 3 ]] || {
      echo 'adaptive DSpark requires DS41_DSPARK_GAMMA=3 (or unset)' >&2; exit 2;
    }
    DS41_DSPARK_GAMMA=3
    export SGLANG_DSV41_DSPARK_ADAPTIVE_VERIFY=1
    ;;
  *) echo 'DS41_DSPARK_POLICY must be fixed or adaptive' >&2; exit 2 ;;
esac
DS41_DSPARK_GAMMA=${DS41_DSPARK_GAMMA:-0}
spec_args=()
preflight_args=()
overlap_args=()
max_running_requests=4
verify_width=1
case "$DS41_DSPARK_GAMMA" in
  0) ;;
  1|3)
    [[ "${SGLANG_RAGGED_VERIFY_MODE:-static}" == static ]] || {
      echo 'DS41 DSpark graphs require SGLANG_RAGGED_VERIFY_MODE=static' >&2; exit 2;
    }
    export SGLANG_RAGGED_VERIFY_MODE=static
    spec_args=(--speculative-algorithm DSPARK --speculative-dspark-block-size "$DS41_DSPARK_GAMMA")
    preflight_args=(--dspark-gamma "$DS41_DSPARK_GAMMA")
    verify_width=$((DS41_DSPARK_GAMMA + 1))
    if [[ "$DS41_DSPARK_GAMMA" == 3 ]]; then
      max_running_requests=2
    fi
    ;;
  *) echo 'DS41_DSPARK_GAMMA must be 0 (off), 1, or 3' >&2; exit 2 ;;
esac
if [[ "$DS41_DSPARK_POLICY" == adaptive ]]; then
  max_running_requests=4
  verify_width=2  # B3/4 uses width 2; B1/2 uses width 4.
  # Width selection uses the current target batch; overlap schedules a step ahead.
  overlap_args=(--disable-overlap-schedule)
fi

# Decode graph buckets, for exact-versus-padded A/B on one build.
#   padded (default): power-of-two buckets; a batch between two buckets replays the next
#     larger one with masked/zeroed padding rows (C3 on [1,2,4] replays the C4 graph).
#   exact: one graph per batch size 1..max-running-requests, so no replay carries padding
#     rows. The runtime must opt in (SGLANG_DSV41_EXACT_GRAPH_BUCKETS=1, exported below);
#     a build without it refuses to start rather than falling back to eager.
# DS41_MAX_RUNNING_REQUESTS (optional) overrides the profile's request capacity, e.g. 8 for
# the eight-concurrency variant; target verification keeps at most eight token rows, so it
# cannot exceed 8 / verify width, and padded buckets need a power of two.
DS41_GRAPH_BUCKETS=${DS41_GRAPH_BUCKETS:-padded}
case "$DS41_GRAPH_BUCKETS" in padded|exact) ;; *)
  echo 'DS41_GRAPH_BUCKETS must be padded or exact' >&2; exit 2 ;;
esac
if [[ -n "${DS41_MAX_RUNNING_REQUESTS:-}" ]]; then
  [[ "$DS41_MAX_RUNNING_REQUESTS" =~ ^[0-9]+$ ]] &&
    (( 10#$DS41_MAX_RUNNING_REQUESTS >= 1 && 10#$DS41_MAX_RUNNING_REQUESTS <= 8 / verify_width )) || {
    echo "DS41_MAX_RUNNING_REQUESTS must be 1..$((8 / verify_width)) for this profile" >&2; exit 2;
  }
  max_running_requests=$((10#$DS41_MAX_RUNNING_REQUESTS))
fi
if [[ "$DS41_DSPARK_POLICY" == adaptive ]] &&
   [[ "$DS41_GRAPH_BUCKETS" != padded || "$max_running_requests" != 4 ]]; then
  echo 'adaptive DSpark requires padded graph buckets and max-running-requests=4' >&2
  exit 2
fi
graph_bs=
if [[ "$DS41_GRAPH_BUCKETS" == exact ]]; then
  for ((bucket = 1; bucket <= max_running_requests; bucket++)); do graph_bs+="${graph_bs:+,}$bucket"; done
  export SGLANG_DSV41_EXACT_GRAPH_BUCKETS=1
else
  case "$max_running_requests" in 1|2|4|8) ;; *)
    echo 'padded graph buckets need a power-of-two request capacity (1/2/4/8); use DS41_GRAPH_BUCKETS=exact' >&2
    exit 2 ;;
  esac
  for bucket in 1 2 4 8; do
    (( bucket < max_running_requests )) && graph_bs+="${graph_bs:+,}$bucket"
  done
  graph_bs+="${graph_bs:+,}$max_running_requests"
  unset SGLANG_DSV41_EXACT_GRAPH_BUCKETS
fi
graph_config='{"decode":{"backend":"full","bs":['"$graph_bs"'],"max_bs":'"$max_running_requests"'}}'
if [[ "$DS41_DSPARK_POLICY" == adaptive ]]; then
  # The scheduler captures the initial width-4 target first. DSpark then
  # captures width-2 target and gamma-3 draft runners on [1,2,4] itself.
  graph_config='{"decode":{"backend":"full","bs":[1,2],"max_bs":2}}'
fi
# Print the resolved profile and stop, before any host, model or GPU access.
if [[ -n "${DS41_PRINT_PROFILE:-}" ]]; then
  printf 'max_running_requests=%s\ngraph_config=%s\nexact_graph_buckets=%s\n' \
    "$max_running_requests" "$graph_config" "${SGLANG_DSV41_EXACT_GRAPH_BUCKETS:-unset}"
  printf 'dspark_policy=%s\ndspark_gamma=%s\nadaptive_verify=%s\nchunked_prefill_size=%s\nprefill_decode_interval=%s\ndisable_overlap_schedule=%s\n' \
    "$DS41_DSPARK_POLICY" "$DS41_DSPARK_GAMMA" "${SGLANG_DSV41_DSPARK_ADAPTIVE_VERIFY:-unset}" "$DS41_CHUNKED_PREFILL_SIZE" "$DS41_PREFILL_DECODE_INTERVAL" "${#overlap_args[@]}"
  printf 'fp4_logits_budget_mb=%s\n' "$DS41_FP4_LOGITS_BUDGET_MB"
  printf 'decoder_swa_bounded_replay=%s\n' "$DS41_DECODER_SWA_BOUNDED_REPLAY"
  exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

# A minimal Strix node need not expose python3 on PATH. Prefer the interpreter
# that ships alongside the runtime (sglang-python) so the mandatory identity/RDMA
# preflight always runs; fall back to PATH python3 only for non-Nix runtimes.
runtime_bin_dir=$(dirname -- "$(realpath -- "$DS41_BINARY")")
preflight_python=$runtime_bin_dir/sglang-python
if [[ ! -x "$preflight_python" ]]; then
  preflight_python=$(command -v python3) || {
    echo 'no runtime sglang-python or PATH python3 for the model preflight' >&2
    exit 2
  }
fi
"$preflight_python" "$script_dir/ds41-model-check.py" "$DS41_MODEL_PATH" "${preflight_args[@]}"

# Keep compiled kernels and support caches off small writable store overlays.
runtime_name=$(basename -- "$(dirname -- "$(dirname -- "$(realpath -- "$DS41_BINARY")")")")
cache_dir=$DS41_CACHE_ROOT/$runtime_name
# A caller may reuse compiled kernels after checking compiler/GPU/Python ABI
# identity. Both compiler caches default to this runtime output's namespace.
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$cache_dir/triton}
[[ "$TRITON_CACHE_DIR" = /* ]] || {
  echo 'TRITON_CACHE_DIR must be an absolute directory' >&2; exit 2;
}
SGLANG_JIT_CACHE_DIR=${SGLANG_JIT_CACHE_DIR:-$cache_dir/sglang-jit}
[[ "$SGLANG_JIT_CACHE_DIR" = /* ]] || {
  echo 'SGLANG_JIT_CACHE_DIR must be an absolute directory' >&2; exit 2;
}
export TRITON_CACHE_DIR SGLANG_JIT_CACHE_DIR
export XDG_CACHE_HOME="$cache_dir/cache" SGLANG_CACHE_DIR="$cache_dir/sglang"
export TORCHINDUCTOR_CACHE_DIR="$cache_dir/torch"
export AITER_JIT_DIR="$cache_dir/aiter-jit" AITER_ROOT_DIR="$cache_dir/aiter-root"
mkdir -p -- "$XDG_CACHE_HOME" "$SGLANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$AITER_JIT_DIR" "$AITER_ROOT_DIR"

# Bind RCCL/Gloo to the model fabric. The node's HCA name may differ by boot.
[[ -d /sys/class/net/cx5fabric0/device/infiniband ]] || {
  echo 'cx5fabric0 RDMA HCA is unavailable' >&2; exit 2;
}
shopt -s nullglob
hcas=(/sys/class/net/cx5fabric0/device/infiniband/*)
(( ${#hcas[@]} == 1 )) && [[ -d ${hcas[0]} ]] || {
  echo 'expected exactly one cx5fabric0 HCA' >&2; exit 2;
}
export NCCL_IB_HCA="=${hcas[0]##*/}:1"

# Native BF16/FP8/MXFP4 TP4/EP1 serving configuration.
export HIP_VISIBLE_DEVICES=0 SGLANG_USE_AITER=0 SGLANG_ROCM_USE_MULTI_STREAM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_DSV4_FP4_EXPERTS=1 SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_DSV4_KV_LAYOUT=v41 SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp8
export SGLANG_DSV4_UNIFIED_KV_FP8=0 SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
# The pooled FP32 score block is retained for the worker lifetime. 512 MiB
# permits a full M1536 score at up to 65536 key columns (384 MiB), avoiding
# host row splitting. Keep the ordinary 64 MiB profile unless opted in.
export SGLANG_DSV4_FP4_LOGITS_BUDGET_MB="$DS41_FP4_LOGITS_BUDGET_MB"
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=1800
export SGLANG_DSV41_REASONING_EFFORT=50
export NCCL_SOCKET_IFNAME=cx5fabric0 GLOO_SOCKET_IFNAME=cx5fabric0
export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_IB_TC=106 NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_AUTOTUNING=1 SGLANG_WARMUP_TIMEOUT=1800
export NCCL_GRAPH_MIXING_SUPPORT=1 ROCPROFILER_QUEUE_INTERPOSITION=0

# Cooperate with the component-test and serving controller's per-node lease.
exec flock --nonblock /tmp/ds41-gpu.lock "$DS41_BINARY" serve \
  --model-path "$DS41_MODEL_PATH" \
  --served-model-name deepseek-v4.1-flash \
  --load-format safetensors \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":2}' \
  --dtype bfloat16 --quantization fp8 \
  --tp-size 4 --nnodes 4 --node-rank "$DS41_NODE_RANK" \
  --dist-init-addr "$DS41_HEAD_ADDR" --dist-timeout 1800 \
  --moe-runner-backend triton --moe-a2a-backend none \
  --fp8-gemm-backend triton --bf16-gemm-backend torch \
  --disable-shared-experts-fusion \
  --attention-backend dsv4 --dsv4-attn-backend flashmla \
  --kv-cache-dtype fp8_e4m3 --page-size 256 --mm-attention-backend sdpa \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-config "$graph_config" \
  --disable-custom-all-reduce "${spec_args[@]}" "${overlap_args[@]}" "${replay_args[@]}" \
  --context-length 131072 --max-total-tokens 262144 \
  --max-running-requests "$max_running_requests" --prefill-max-requests 2 \
  --schedule-policy shortest-prefill-first \
  --chunked-prefill-size "$DS41_CHUNKED_PREFILL_SIZE" --prefill-decode-interval "$DS41_PREFILL_DECODE_INTERVAL" \
  --swa-full-tokens-ratio 1.0 --mem-fraction-static 0.82 \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --default-chat-template-kwargs '{"reasoning_effort":50}' \
  --enable-cache-report --watchdog-timeout 2100 \
  --host 127.0.0.1 --port "$DS41_PORT"
