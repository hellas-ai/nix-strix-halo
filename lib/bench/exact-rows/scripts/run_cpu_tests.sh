#!/usr/bin/env bash
# usage: run_cpu_tests.sh <overlay> <worktree>   -- runs every sglang-python test registered in the worktree's default.nix postFixup (CPU)
OV=$1; WT=$2
RT=/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308
export PYTHONDONTWRITEBYTECODE=1 ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export PYTHONPATH="$OV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
rc=0
for t in $(grep -o '\./tests/[A-Za-z0-9_.-]*\.py' "$WT/pkgs/sglang/v41/default.nix" | sort -u); do
  f="$WT/pkgs/sglang/v41/${t#./}"
  out=$(cd "$WT" && "$RT/bin/sglang-python" "$f" "$OV" 2>&1); r=$?
  if [ $r -eq 0 ]; then echo "PASS $(basename $f)"; else echo "FAIL($r) $(basename $f)"; echo "$out" | tail -5; rc=1; fi
done
exit $rc
