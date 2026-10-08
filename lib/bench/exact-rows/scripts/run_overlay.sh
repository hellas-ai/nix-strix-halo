#!/usr/bin/env bash
# CPU-only runner: <overlay-with-edited-sglang> <script> [args...]. No GPU is ever visible.
OV=$1; shift
RT=/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308
export PYTHONDONTWRITEBYTECODE=1 ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
export PYTHONPATH="$OV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/tcache}
exec $RT/bin/sglang-python "$@"
