#!/usr/bin/env bash
# GPU session for task 1 (masked padding). Run ON a strix node as grw after GPU-FREE-FOR-MICROBENCH exists:
#   bash S-exact-rows/t1-mask/session_t1.sh        (OVERLAY=... NODE=strix-2 optional)
source "$(dirname "$0")/../scripts/session_lib.sh"
R=$S_ROOT/t1-mask/results; mkdir -p "$R"
T=$S_ROOT/t1-mask
step "masked chain: bitwise + graph safety + timing, random dead routing, installed FMA live rows at 2/4/8 (default policy)"
gpu_py $T/live_row_mask_gpu.py --dead-routing random --policy live --out $R/chain_random_live.json | tee $R/chain_random_live.log
step "same, dead rows all routed to the same 6 experts (the zero-hidden padded row of a real graph)"
gpu_py $T/live_row_mask_gpu.py --dead-routing fixed --policy live --cases 4:3,8:6,8:5,8:7 --out $R/chain_fixed_live.json | tee $R/chain_fixed_live.log
step "WMMA gate/up policy (opt-in) for completeness"
gpu_py $T/live_row_mask_gpu.py --dead-routing random --policy wmma --cases 4:3,8:6 --out $R/chain_random_wmma.json | tee $R/chain_random_wmma.log
echo "t1 session complete $(date +%T)"
