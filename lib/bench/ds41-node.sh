#!/usr/bin/env bash
# Run one candidate DeepSeek V4.1 Flash TP4 rank on an already configured Strix.
set -euo pipefail

: "${DS41_NODE_RANK:?set node rank: strix-3=0, strix-4=1, strix-2=2, strix-1=3}"
: "${DS41_BINARY:?set the built sglang executable, e.g. /nix/store/.../bin/sglang}"
DS41_HEAD_ADDR=${DS41_HEAD_ADDR:-192.168.25.103:51041}
DS41_MODEL_PATH=${DS41_MODEL_PATH:-/mnt/ds41-models-rdma/DeepSeek-V4.1-Flash-hf-dba1be0a}
DS41_PORT=${DS41_PORT:-31041}

case "$DS41_NODE_RANK" in 0|1|2|3) ;; *) echo 'invalid DS41_NODE_RANK' >&2; exit 2 ;; esac
[[ "$DS41_BINARY" = /* && -x "$DS41_BINARY" ]] || {
  echo 'DS41_BINARY must be an executable absolute path' >&2; exit 2;
}
[[ "$DS41_PORT" =~ ^[0-9]+$ ]] && (( 10#$DS41_PORT >= 1 && 10#$DS41_PORT <= 65535 )) || {
  echo 'DS41_PORT must be in 1..65535' >&2; exit 2;
}
[[ -n "$DS41_HEAD_ADDR" ]] || { echo 'DS41_HEAD_ADDR is empty' >&2; exit 2; }
DS41_CACHE_ROOT=${DS41_CACHE_ROOT:-/tmp/ds41-compiler-cache/gfx1151}
[[ "$DS41_CACHE_ROOT" = /* ]] || {
  echo 'DS41_CACHE_ROOT must be an absolute writable directory' >&2; exit 2;
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python3 "$script_dir/ds41-model-check.py" "$DS41_MODEL_PATH"

# Keep compiled kernels and support caches off small writable store overlays.
runtime_name=$(basename -- "$(dirname -- "$(dirname -- "$(realpath -- "$DS41_BINARY")")")")
cache_dir=$DS41_CACHE_ROOT/$runtime_name
export XDG_CACHE_HOME="$cache_dir/cache" SGLANG_CACHE_DIR="$cache_dir/sglang"
export TRITON_CACHE_DIR="$cache_dir/triton" TORCHINDUCTOR_CACHE_DIR="$cache_dir/torch"
export AITER_JIT_DIR="$cache_dir/aiter-jit" AITER_ROOT_DIR="$cache_dir/aiter-root"
mkdir -p -- "$XDG_CACHE_HOME" "$SGLANG_CACHE_DIR" "$TRITON_CACHE_DIR" \
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
export SGLANG_DSV4_KV_LAYOUT=v4 SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp8
export SGLANG_DSV4_UNIFIED_KV_FP8=0 SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_DSV4_FP4_LOGITS_BUDGET_MB=64
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=1800
export SGLANG_DSV41_REASONING_EFFORT=50
export NCCL_SOCKET_IFNAME=cx5fabric0 GLOO_SOCKET_IFNAME=cx5fabric0
export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_IB_TC=106 NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_AUTOTUNING=1 SGLANG_WARMUP_TIMEOUT=1800
export NCCL_GRAPH_MIXING_SUPPORT=1 ROCPROFILER_QUEUE_INTERPOSITION=0

exec "$DS41_BINARY" serve \
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
  --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2],"max_bs":2}}' \
  --disable-custom-all-reduce \
  --context-length 131072 --max-total-tokens 262144 \
  --max-running-requests 2 --prefill-max-requests 1 \
  --chunked-prefill-size 256 --prefill-decode-interval 1 \
  --swa-full-tokens-ratio 1.0 --mem-fraction-static 0.82 \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --default-chat-template-kwargs '{"reasoning_effort":50}' \
  --enable-cache-report --watchdog-timeout 2100 \
  --host 127.0.0.1 --port "$DS41_PORT"
