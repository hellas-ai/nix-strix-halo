#!/usr/bin/env bash
# Serve Qwen3.8-27B on strix-2's four Radeon Pro V620s (gfx1030 / RDNA2).
#
# ISOLATION: HIP indices 0-3 are the V620s. Index 4 is the Strix Halo iGPU and
# belongs to the parallel DS4 campaign -- never expose it here.
#
# On the diskless nodes /mnt/Home is NFS and `./script` fails with EACCES, so
# invoke this as `bash qwen38-serve.sh`. Nothing is written to /tmp: it is
# tmpfs and these machines reboot.
set -euo pipefail

# NOT /models: that is an SPDK lvol *snapshot* exported over NVMe-oF, and
# republishing it to include a new model requires draining and rebooting all
# four Strix hosts (see machines/x86/trex/spdk-models-snapshot.nix). /mnt/Home
# is live NFS4 from trex and needs no such ceremony.
MODEL="${MODEL:-/mnt/Home/models/Qwen3.8-27B}"
TP="${TP:-4}"
PORT="${PORT:-30800}"
# fp16, not bf16: gfx1030 has no bf16 ALU but has native packed fp16 at 2x
# rate. fp16 carries 10 mantissa bits against bf16's 7, so the conversion is
# exact for every weight inside fp16's range -- see .bench-artifacts/fidelity/.
DTYPE="${DTYPE:-float16}"
# The config pins the DeltaNet recurrent state to fp32 (mamba_ssm_dtype); keep
# it there regardless of the weight dtype.
SSM_DTYPE="${SSM_DTYPE:-float32}"

# The GDN *conv* state is a separate dtype from the SSM state and has NO CLI
# flag: srt/environ.py:678 hardcodes SGLANG_MAMBA_CONV_DTYPE="bfloat16", and
# srt/configs/mamba_utils.py:68 falls back to torch.bfloat16. With --dtype
# float16 the activations reaching _causal_conv1d_fwd_kernel are fp16 while the
# conv_states tensor is bf16, and the Triton frontend rejects the kernel:
#   AssertionError("Mismatched type for col0 between then block
#   (<['256'], bf16>) and else block (<['256'], fp16>)")
# It must therefore track the weight dtype on any bf16-less GPU.
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-$DTYPE}"
CTX="${CTX:-32768}"
MEM_FRAC="${MEM_FRAC:-0.85}"
# sglang 0.5.14's tool-call parser registry maps "qwen" to Qwen25Detector,
# which is correct for this model family's <tool_call> JSON format. Set to
# "" to disable tool-call parsing.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
# The chat template emits <think> blocks; sglang 0.5.14's reasoning parser
# registry (srt/parser/reasoning_parser.py, DetectorMap) has a "qwen3" entry
# for them. Set to "" to disable reasoning parsing.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

ART="${ART:-/mnt/Home/src/nix-strix-halo-qwen38/.bench-artifacts}"
LOGDIR="$ART/serve"
mkdir -p "$LOGDIR"

# Caches must not land on the node's tmpfs overlay -- it fills and then
# stale-handles the diskless root.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/Home/src/.cache/qwen38}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/hf}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

# The four V620s only.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

# TP4 here is entirely switch-local P2P/IPC across one PEX880xx, so the socket
# path is only used for bootstrap. Pin it anyway: unpinned interface selection
# has previously wedged multi-rank jobs on this fleet.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# SGLANG_GFX1030_CLOSURE must be a real (non-symlink) nix store path for the
# gfx1030 sglang build. Never point it at a shared result-* symlink: re-linking
# one out from under a running server has cost this campaign real time before.
if [[ -z "${SGLANG_GFX1030_CLOSURE:-}" ]]; then
  echo "error: set SGLANG_GFX1030_CLOSURE to a gfx1030 sglang store path" >&2
  echo "  e.g. SGLANG_GFX1030_CLOSURE=/nix/store/...-sglang-rocm-gfx1030-0.5.14" >&2
  exit 2
fi
if [[ -L "$SGLANG_GFX1030_CLOSURE" ]]; then
  echo "error: SGLANG_GFX1030_CLOSURE is a symlink; pass the resolved store path" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/serve-$STAMP.log"

# Backend choice is forced, not preferred: LINEAR_ATTN_KERNEL_BACKEND_CHOICES is
# ["triton","cutedsl","flashinfer"] and the latter two are CUDA-only, so Triton
# is the ONLY backend that can drive the 48 Gated-DeltaNet layers on gfx1030.
# Likewise aiter/wave/flashinfer cannot serve the 16 full-attention layers here.
ARGS=(
  --model-path "$MODEL"
  --tp-size "$TP"
  --dtype "$DTYPE"
  --mamba-ssm-dtype "$SSM_DTYPE"
  --attention-backend triton
  --linear-attn-backend triton
  # The mamba radix cache's "extra_buffer" strategy now works on ROCm via the
  # gfx1030 patch set (it used to assert is_cuda()/is_musa()/is_npu() in
  # server_args.py:4369). Leaving the radix cache on is the biggest TTFT win
  # we have -- 74x on warm 30K-token prefixes. --disable-radix-cache remains
  # available as a manual escape hatch if you need a clean baseline.
  --context-length "$CTX"
  --mem-fraction-static "$MEM_FRAC"
  --host 0.0.0.0
  --port "$PORT"
  --log-level info
)

if [[ -n "$TOOL_CALL_PARSER" ]]; then
  ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

# Extra args passed through, e.g. --disable-cuda-graph while bisecting a hang.
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "launching: tp=$TP dtype=$DTYPE ctx=$CTX devices=$HIP_VISIBLE_DEVICES"
echo "log: $LOG"
# NOTE: sglang 0.5.14's CLI subcommands are {serve,generate,version}. The old
# `launch_server` entry point is gone; `sglang serve` is the replacement.
exec "$SGLANG_GFX1030_CLOSURE/bin/sglang" serve "${ARGS[@]}" 2>&1 | tee "$LOG"
