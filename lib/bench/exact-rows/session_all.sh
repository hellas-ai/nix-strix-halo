#!/usr/bin/env bash
# Everything GPU-side of S-exact-rows, one node, one GPU, serially, in order of decisiveness.  Each part re-checks the sentinel and
# waits for an idle GPU before every step (scripts/session_lib.sh).  Run ON a strix node as grw AFTER GPU-FREE-FOR-MICROBENCH exists.
# Build the overlay first (CPU only):  scripts/mk_overlay.sh <tree-with-0046..0049> <overlay-final>  (see REPORT.md section 8).
set -uo pipefail
D=$(dirname "$(readlink -f "$0")")
bash "$D/t1-mask/session_t1.sh" 2>&1 | tee -a "$D/session_all.log"
bash "$D/t2-gemv/session_t2.sh" 2>&1 | tee -a "$D/session_all.log"
bash "$D/t3-draft/session_t3.sh" 2>&1 | tee -a "$D/session_all.log"
bash "$D/t4-buckets/scripts/session_t4.sh" 2>&1 | tee -a "$D/session_all.log"
