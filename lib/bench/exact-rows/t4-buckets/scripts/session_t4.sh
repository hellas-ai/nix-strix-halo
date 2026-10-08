#!/usr/bin/env bash
# GPU session for the exact-graph-bucket evidence (strix-2 only, ONLY after the sentinel appears). Run ON strix-2 as grw, serially.
# Single GPU, < 8 GB, no model, no collectives, no service touched, does NOT take /tmp/ds41-gpu.lock.
# Measures what an extra decode graph costs (kernel-node bookkeeping, capture time, replay time vs resident graph count) on
# synthetic 1000/2500/5000-node graphs.  Real activation-pool memory and per-bucket latency need the full-model A/B (../AB-PROTOCOL.txt).
set -uo pipefail
K=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/S-exact-rows/t4-buckets
SENT=/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH
RT=${SGLANG_RT:-/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308}
R=$K/results; mkdir -p "$R"
[ -e "$SENT" ] || { echo "sentinel absent; refusing"; exit 3; }
idle_check() {  # the node's GPU must be idle (the serving process is loaded but quiescent) before each run
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
gpu() {
  PYTHONDONTWRITEBYTECODE=1 ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
    TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$K/tcache-strix} "$RT/bin/sglang-python" "$@"
}
step "graph overhead probe, 5000 nodes (a real decode graph), 1..32 resident graphs, shared vs private pool"
gpu "$K/scripts/graph_overhead_probe.py" --nodes 5000 --out "$R/probe_5000.json" 2>&1 | tee "$R/probe_5000.log"
step "linearity in node count: 2500 nodes"
gpu "$K/scripts/graph_overhead_probe.py" --nodes 2500 --graphs 1 2 4 8 16 --out "$R/probe_2500.json" 2>&1 | tee "$R/probe_2500.log"
step "linearity in node count: 1000 nodes"
gpu "$K/scripts/graph_overhead_probe.py" --nodes 1000 --graphs 1 2 4 8 16 --out "$R/probe_1000.json" 2>&1 | tee "$R/probe_1000.log"
echo "done $(date +%T)"
