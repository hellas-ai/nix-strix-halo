#!/usr/bin/env bash
# GPU session for task 2 (row-generic GEMV, HC post any M). Run ON a strix node as grw after GPU-FREE-FOR-MICROBENCH exists.
# (session1.sh in this directory is the fork's original script, with its own scratch-tree paths; this one runs the same stages
#  against the integrated overlay.)
source "$(dirname "$0")/../scripts/session_lib.sh"
T=$S_ROOT/t2-gemv; R=$T/results; mkdir -p "$R"
step "HC post: native vs _mhc_post_torch at rows 1..8 and M = 9..4096 (bitwise, order probe, signed zeros, chunked launch) + timing"
gpu_py $T/bench_gemv_rows.py --stage hcpost | tee $R/hcpost.log
step "GEMV: new vs pre-patch kernels (bitwise at 1,2,4,8), generic matmul, batch invariance, rows 1..8, ten shapes (incl. the draft's wq_a, wkv, main_proj)"
gpu_py $T/bench_gemv_rows.py --stage gemv | tee $R/gemv.log
step "GEMV: BLOCK_K 256/512/1024 sweep of the draft shapes and engram (informational)"
gpu_py $T/bench_gemv_rows.py --stage blockk | tee $R/blockk.log
step "GPU test tests/fp8.py (rows 1..8 native, 9+ generic, C1=C2=..=C8 bitwise where designed)"
gpu_py $V41/tests/fp8.py "$OVERLAY" | tee $R/test_fp8.log
step "GPU test tests/indexer-wqb-gemv.py (rows 1..8, kill switch)"
gpu_py $V41/tests/indexer-wqb-gemv.py "$OVERLAY" | tee $R/test_indexer_wqb.log
step "GPU test tests/mhc.py (HC post rows 1..8 bitwise vs the torch fallback, graph replay)"
gpu_py $V41/tests/mhc.py "$OVERLAY" | tee $R/test_mhc.log
echo "t2 session complete $(date +%T)"
