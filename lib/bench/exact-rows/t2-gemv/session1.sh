#!/usr/bin/env bash
# t2 GPU session (ONE node, default: run it on the node you were given, as grw), serially.  Aborts if the sentinel disappears.
# Order: cheapest and most decisive first (HC post bitwise at every M, GEMV bitwise old/new), then timings, then the updated tests.
# Everything < 8 GB of GPU memory.  Results land in results/.
set -uo pipefail
T=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/S-exact-rows/t2-gemv
V41=${V41:-/mnt/Home/src/nix-strix-halo-exactrows-t2/pkgs/sglang/v41}
SENT=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH
OVERLAY=${SGLANG_OVERLAY:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/tree-t2}
R=$T/results; mkdir -p "$R"
idle_check() {  # the node's GPU must be idle (a loaded but quiescent serving process is fine) before each run
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
step "HC post: native vs _mhc_post_torch at rows 1..8 and M = 9..4096 (bitwise, order probe, signed zeros, chunked launch)"
$T/run.gpu.sh $T/bench_gemv_rows.py --stage hcpost 2>&1 | tee $R/hcpost.log
step "GEMV: new vs baseline kernels (bitwise at 1,2,4,8), generic matmul, batch invariance, rows 1..8, ten shapes"
$T/run.gpu.sh $T/bench_gemv_rows.py --stage gemv 2>&1 | tee $R/gemv.log
step "GEMV: BLOCK_K 256/512/1024 sweep of the draft shapes and engram (informational)"
$T/run.gpu.sh $T/bench_gemv_rows.py --stage blockk 2>&1 | tee $R/blockk.log
step "updated GPU tests: tests/fp8.py (rows 1..8 native, 9+ generic)"
$T/run.gpu.sh $V41/tests/fp8.py "$OVERLAY" 2>&1 | tee $R/test_fp8.log
step "updated GPU tests: tests/indexer-wqb-gemv.py (rows 1..8, kill switch)"
$T/run.gpu.sh $V41/tests/indexer-wqb-gemv.py "$OVERLAY" --source "$OVERLAY" 2>&1 | tee $R/test_indexer_wqb.log
step "updated GPU tests: tests/mhc.py (HC post rows 1..8 bitwise vs the torch fallback, graph replay)"
$T/run.gpu.sh $V41/tests/mhc.py "$OVERLAY" 2>&1 | tee $R/test_mhc.log
echo "=== done $(date +%T)"
