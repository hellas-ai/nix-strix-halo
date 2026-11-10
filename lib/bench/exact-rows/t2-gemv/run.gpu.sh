#!/usr/bin/env bash
# GPU runner for the t2 (row-generic GEMV / HC post) microbench. NOT RUN YET. Only after the sentinel exists, on ONE node, as grw.
# Does not touch /tmp/ds41-gpu.lock, any service, or the production Triton caches (own cache dir under this directory).
# SGLANG_OVERLAY: patched sglang source tree (default: the t2 scratch tree = baseline 0001..0045 + patch 0047); SGLANG_RT: runtime python.
set -euo pipefail
T=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/S-exact-rows/t2-gemv
SENT=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH
[ -e "$SENT" ] || { echo "sentinel absent; refusing"; exit 3; }
R=/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308
OVERLAY=${SGLANG_OVERLAY:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/tree-t2}
export PYTHONDONTWRITEBYTECODE=1 ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export SGLANG_USE_AITER=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_FLASHINFER_MHC=0
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$T/tcache-$(hostname)}
exec ${SGLANG_RT:-$R}/bin/sglang-python "$@"
