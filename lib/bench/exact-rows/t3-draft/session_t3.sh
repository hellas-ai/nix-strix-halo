#!/usr/bin/env bash
# GPU session for task 3 (draft path at exact rows). Run ON a strix node as grw after GPU-FREE-FOR-MICROBENCH exists.
source "$(dirname "$0")/../scripts/session_lib.sh"
T=$S_ROOT/t3-draft; R=$T/results; mkdir -p "$R"
step "draft MoE chain: stage-by-stage bitwise vs installed AITER sequence + timing, rows 1..8, 12, 16 (128 experts, top-3)"
gpu_py $T/draft_moe_gpu.py --out $R/draft_moe.json | tee $R/draft_moe.log
step "draft MoE tile sweep at the draft row counts 3 and 6 (arithmetic-neutral; bitwise vs the default tiles is implied by owner-routed WMMA)"
for gu in 32,256,2,2 64,256,4,2 32,128,2,2 64,128,4,2 16,256,1,2 32,256,2,3; do
  gpu_py $T/draft_moe_gpu.py --rows 3 6 --gu $gu --out $R/draft_moe_gu_${gu//,/_}.json | tee -a $R/draft_moe_sweep.log
done
for dn in 32,64,2,2 64,64,4,2 32,128,2,2 16,64,1,2 64,128,4,2; do
  gpu_py $T/draft_moe_gpu.py --rows 3 6 --dn $dn --out $R/draft_moe_dn_${dn//,/_}.json | tee -a $R/draft_moe_sweep.log
done
echo "t3 (MoE part) complete $(date +%T)"
