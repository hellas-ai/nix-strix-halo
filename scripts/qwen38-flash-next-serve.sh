#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on strix-2's four Radeon Pro V620s.
#
# Invoke with bash: /mnt/Home is NFS on the diskless Strix nodes and is not
# mounted executable.  Persistent artifacts and JIT caches never use /tmp.
set -euo pipefail

MODEL="${MODEL:-/models/Qwen3.8-Flash-Next-W4A16-G32}"
TP="${TP:-4}"
PORT="${PORT:-30800}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
DTYPE="${DTYPE:-float16}"
SSM_DTYPE="${SSM_DTYPE:-float32}"
CTX="${CTX:-32768}"
MEM_FRAC="${MEM_FRAC:-0.78}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# The unmerged QSA HIP path first needs eager-vs-graph token parity.  Promote
# CUDA_GRAPH=1 only after that gate passes on the exact production closure.
CUDA_GRAPH="${CUDA_GRAPH:-0}"

ART="${ART:-/mnt/Home/src/nix-strix-halo-qwen38-flash-next/.bench-artifacts}"
LOGDIR="$ART/serve"
RUNDIR="$ART/runtime"
mkdir -p "$LOGDIR" "$RUNDIR"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/Home/src/.cache/qwen38-flash-next}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/hf}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-$XDG_CACHE_HOME/aiter/jit}"
export AITER_ROOT_DIR="${AITER_ROOT_DIR:-$XDG_CACHE_HOME/aiter/root}"
# Keep the allocator resilient for JIT and inference temporaries.  The larger
# WNA16 post-load transpose is separately bounded by the gfx1030 host-layout
# patch; it must not depend on allocator fragmentation for residency.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$HF_HOME" \
  "$AITER_JIT_DIR" \
  "$AITER_ROOT_DIR"

# HIP indices 0-3 are the V620s.  The Strix Halo iGPU belongs to the separate
# DS4 campaign and must never join this TP group.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export SGLANG_USE_AITER=0
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-$DTYPE}"

if [[ "$TP" != 4 ]]; then
  echo "error: the qualified V620 topology is TP=4, got TP=$TP" >&2
  exit 2
fi
if [[ "$DTYPE" != float16 ]]; then
  echo "error: gfx1030 has native FP16, not BF16/FP8 compute; use DTYPE=float16" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "error: model directory does not exist: $MODEL" >&2
  exit 2
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "error: missing model config: $MODEL/config.json" >&2
  exit 2
fi

# Require the reviewed, source-built Qwen4-Exp closure.  A real store path is
# deliberate: relinking a result symlink under a running server is unsafe.
if [[ -z "${SGLANG_QWEN38_FLASH_NEXT_CLOSURE:-}" ]]; then
  echo "error: set SGLANG_QWEN38_FLASH_NEXT_CLOSURE to the resolved Nix store path" >&2
  exit 2
fi
if [[ -L "$SGLANG_QWEN38_FLASH_NEXT_CLOSURE" ]]; then
  echo "error: SGLANG_QWEN38_FLASH_NEXT_CLOSURE must not be a symlink" >&2
  exit 2
fi
SGLANG="$SGLANG_QWEN38_FLASH_NEXT_CLOSURE/bin/sglang"
if [[ ! -x "$SGLANG" ]]; then
  echo "error: SGLang executable not found: $SGLANG" >&2
  exit 2
fi

ARGS=(
  --model-path "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --tp-size "$TP"
  --dtype "$DTYPE"
  --mamba-ssm-dtype "$SSM_DTYPE"
  --language-model-only
  --ple-offload-embedding
  --attention-backend triton
  --linear-attn-backend triton
  --bf16-gemm-backend torch
  --moe-runner-backend triton
  --disable-custom-all-reduce
  --weight-loader-drop-cache-after-load
  --context-length "$CTX"
  --mem-fraction-static "$MEM_FRAC"
  --host 0.0.0.0
  --port "$PORT"
  --log-level info
)

if [[ "$CUDA_GRAPH" == 0 ]]; then
  ARGS+=(--disable-cuda-graph)
elif [[ "$CUDA_GRAPH" != 1 ]]; then
  echo "error: CUDA_GRAPH must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/serve-$STAMP.log"
printf 'launching: model=%s tp=%s dtype=%s ctx=%s graph=%s devices=%s\n' \
  "$MODEL" "$TP" "$DTYPE" "$CTX" "$CUDA_GRAPH" "$HIP_VISIBLE_DEVICES"
printf 'log: %s\n' "$LOG"
cd "$RUNDIR"
exec "$SGLANG" serve "${ARGS[@]}" 2>&1 | tee "$LOG"
