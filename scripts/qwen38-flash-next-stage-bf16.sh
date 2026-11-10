#!/usr/bin/env bash
# Resume and verify the immutable BF16 oracle on trex's SPDK-backed XFS mount.
set -euo pipefail

REPO="Qwen/Qwen3.8-Flash-Next"
REVISION="f5d08274bafd880402bd16f5e3e6c514136ec06c"
MODEL_DIR="${MODEL_DIR:-/mnt/optane/models/Qwen3.8-Flash-Next-BF16}"
CACHE_DIR="${CACHE_DIR:-/mnt/optane/models/.cache/huggingface/qwen38-flash-next-bf16}"
WORKTREE="${WORKTREE:-/mnt/Home/src/nix-strix-halo-qwen38-flash-next}"
HF_CLI="${HF_CLI:-hf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$MODEL_DIR" in
  /mnt/optane/models/*) ;;
  *)
    echo "error: MODEL_DIR must be below the fast /mnt/optane/models mount" >&2
    exit 2
    ;;
esac
if [[ "$(findmnt -n -o TARGET -T "$MODEL_DIR")" != "/mnt/optane/models" ]]; then
  echo "error: MODEL_DIR is not on the SPDK model filesystem" >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "error: pre-create $MODEL_DIR with ownership for $(id -un)" >&2
  exit 2
fi
if [[ ! -w "$MODEL_DIR" ]]; then
  echo "error: $MODEL_DIR is not writable by $(id -un)" >&2
  exit 2
fi
if [[ ! -f "$WORKTREE/scripts/qwen38-flash-next-inventory.py" ]]; then
  echo "error: campaign inventory script is missing from $WORKTREE" >&2
  exit 2
fi
if ! command -v "$HF_CLI" >/dev/null; then
  echo "error: Hugging Face CLI not found: $HF_CLI" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "error: Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
if [[ ! -w "$CACHE_DIR" ]]; then
  echo "error: $CACHE_DIR is not writable by $(id -un)" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/mnt/optane/models/.cache/huggingface}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_XET=0

"$HF_CLI" download "$REPO" \
  --revision "$REVISION" \
  --local-dir "$MODEL_DIR" \
  --cache-dir "$CACHE_DIR"

"$PYTHON_BIN" \
  "$WORKTREE/scripts/qwen38-flash-next-inventory.py" \
  --model "$MODEL_DIR" \
  --require-complete \
  --expected-shards 131

printf '%s\n' "$REVISION" > "$MODEL_DIR/.campaign-revision"
printf 'verified BF16 oracle: %s @ %s\n' "$REPO" "$REVISION"
