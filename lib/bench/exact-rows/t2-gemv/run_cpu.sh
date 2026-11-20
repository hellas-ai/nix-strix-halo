#!/usr/bin/env bash
# Offline (no GPU) runner for the sglang-v41 python; OVERLAY (default: the t2 scratch tree) shadows the installed sglang.
R=/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308
OVERLAY=${OVERLAY:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/tree-t2}
export PYTHONDONTWRITEBYTECODE=1 ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/tcache-t2}
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
exec $R/bin/sglang-python "$@"
