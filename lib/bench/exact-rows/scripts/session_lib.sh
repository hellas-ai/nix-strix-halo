#!/usr/bin/env bash
# Shared helpers for the S-exact-rows GPU sessions (source this).  Style of K2-moe/scripts/session1.sh.
#   * refuses to start unless the GPU-FREE-FOR-MICROBENCH sentinel exists; every step re-checks it and stops if it vanished
#   * every step first waits for the GPU to be idle (/sys/class/drm/card*/device/gpu_busy_percent <= 10 % averaged over 5 s)
#   * ONE GPU (HIP_VISIBLE_DEVICES=0), <8 GB (the python scripts call benchlib.check_budget(8.0)); never touches services,
#     /tmp/ds41-gpu.lock, or systemd units.  Run it ON the node as grw (ssh strix-N), e.g. `bash t2-gemv/session_t2.sh`.
# Env: OVERLAY (default: the final integrated overlay, built by S-exact-rows/scripts/mk_overlay.sh), NODE (informational),
#      RT (installed runtime that supplies compiled parts; default runtime B of the DSpark integration).
set -uo pipefail
S_ROOT=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/S-exact-rows
SENT=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH
OVERLAY=${OVERLAY:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/overlay-final}
RT=${RT:-/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308}
V41=${V41:-/mnt/Home/src/nix-strix-halo-exactrows/pkgs/sglang/v41}   # the branch's tests/ (GPU tests take the overlay ROOT as argument)
export S_ROOT SENT OVERLAY RT V41
[ -e "$SENT" ] || { echo "sentinel $SENT absent: refusing to touch the GPU"; exit 3; }
[ -d "$OVERLAY/lib/python3.13/site-packages/sglang" ] || { echo "overlay $OVERLAY missing"; exit 2; }

idle_check() {
  for try in $(seq 1 30); do
    busy=0; n=0
    for i in 1 2 3 4 5; do
      for f in /sys/class/drm/card*/device/gpu_busy_percent; do [ -r "$f" ] && { busy=$((busy + $(cat "$f"))); n=$((n + 1)); }; done
      sleep 1
    done
    [ "$n" -gt 0 ] && [ $((busy / n)) -le 10 ] && return 0
    echo "GPU busy ($((busy / (n > 0 ? n : 1)))%), waiting"; sleep 20
  done
  echo "GPU never idle: aborting"; exit 4
}
step() { [ -e "$SENT" ] || { echo "SENTINEL GONE: stopping"; exit 3; }; idle_check; echo "=== $(date +%T) $*"; }

# gpu_py <script> [args]: one GPU, overlay first on PYTHONPATH, private Triton cache, no profiler hooks
gpu_py() {
  [ -e "$SENT" ] || { echo "SENTINEL GONE"; exit 3; }
  PYTHONDONTWRITEBYTECODE=1 ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  PYTHONPATH="$OVERLAY/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}" \
  TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$S_ROOT/tcache-gpu-$(hostname -s)} \
  SGLANG_USE_AITER=${SGLANG_USE_AITER:-0} SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_FLASHINFER_MHC=0 \
  "$RT/bin/sglang-python" "$@" 2>&1 | grep -v "^\[aiter\]\|UserWarning\|warnings.warn\|USDT"
}
