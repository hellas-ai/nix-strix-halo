#!/usr/bin/env bash
# Run one row of the finetune benchmark matrix.
#
# Required env vars (set by the nix wrapper):
#   ROW_JSON       — config dict for this row (transport, model, type, ...)
#   ENV_PATH       — strix-halo-finetune-env-<arch> store path
#   MASTER_HOST    — ssh host for rank 0
#   WORKER_HOST    — ssh host for rank 1
#   MASTER_ADDR    — IP for torch's --master_addr
#   MASTER_PORT    — TCP port for the master rendezvous
#   OUT_DIR        — directory to write per-row JSON + logs
#   SNAPSHOT_SH    — path to snapshot.sh that we ship to each node
#   RUN_ID         — caller-supplied tag (used in filenames + the json)
#   ROW_IDX        — caller-supplied integer row index
#
# Reads the row JSON, extracts (transport, model, type, epochs, ...),
# snapshots counters on both nodes, runs torchrun, snapshots again,
# computes deltas, writes a JSON document to $OUT_DIR/row-<idx>.json.

set -euo pipefail
: "${ROW_JSON:?}"
: "${ENV_PATH:?}"
: "${MASTER_HOST:?}"
: "${WORKER_HOST:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"
: "${OUT_DIR:?}"
: "${SNAPSHOT_SH:?}"
: "${RUN_ID:?}"
: "${ROW_IDX:?}"

mkdir -p "$OUT_DIR"
ROW_LOG_DIR="$OUT_DIR/row-$ROW_IDX-logs"
mkdir -p "$ROW_LOG_DIR"

# Extract row fields with jq.
TRANSPORT=$(jq -r '.transport'           <<<"$ROW_JSON")
MODEL=$(jq -r     '.model'               <<<"$ROW_JSON")
TYPE=$(jq -r      '.ft_type'             <<<"$ROW_JSON")
DATASET=$(jq -r   '.dataset      // "Abirate/english_quotes"' <<<"$ROW_JSON")
EPOCHS=$(jq -r    '.epochs        // 1'  <<<"$ROW_JSON")
BATCH=$(jq -r     '.batch_size    // 1'  <<<"$ROW_JSON")
MAXLEN=$(jq -r    '.max_length    // 64' <<<"$ROW_JSON")
GA=$(jq -r        '.gradient_accumulation // 1' <<<"$ROW_JSON")
MAX_STEPS=$(jq -r '.max_steps     // -1' <<<"$ROW_JSON")
NO_EVAL=$(jq -r    '.no_eval       // false' <<<"$ROW_JSON")
LR=$(jq -r        '.learning_rate // 5e-5' <<<"$ROW_JSON")
STRATEGY=$(jq -r  '.strategy      // "fsdp"' <<<"$ROW_JSON")
IFACE=$(jq -r     '.iface         // "br0.lan"' <<<"$ROW_JSON")
EXTRA_ENV=$(jq -r '.extra_env     // ""'  <<<"$ROW_JSON")

case "$TRANSPORT" in
  eth)
    TRANSPORT_ENV="export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE"
    ;;
  rdma)
    # IB-RDMA: rely on libibverbs being on each node's LD_LIBRARY_PATH
    # (the caller passes that via extra_env). NCCL_SOCKET_IFNAME pins
    # the bootstrap TCP to br0.lan so NCCL doesn't auto-pick e.g. the
    # ardma0 dummy netdev (which has a single-node-routable /32 IP).
    TRANSPORT_ENV="export NCCL_IB_DISABLE=0 NCCL_IB_HCA=usb4_rdma0 NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
    ;;
  rdma-4hca)
    # The new ibverbs module's per-rail naming is (domain * (MAX_LANES+1)
    # + lane), so a 2-domain * 2-lane setup gets usb4_rdma{0,1,5,6}, not
    # contiguous 0..3.
    TRANSPORT_ENV="export NCCL_IB_DISABLE=0 NCCL_IB_HCA=usb4_rdma0,usb4_rdma1,usb4_rdma5,usb4_rdma6 NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE NCCL_NET_MERGE_LEVEL=LOC NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
    ;;
  *)
    echo "unknown transport: $TRANSPORT" >&2; exit 2 ;;
esac

# For rdma transports, sanity-check that libibverbs is reachable and the
# expected IB device(s) are present on BOTH nodes before burning a 10-min
# training run. Silent fallback to NET/Socket is the worst-possible failure
# mode — we'd report ETH-tier numbers under an RDMA label.
case "$TRANSPORT" in
  rdma|rdma-4hca)
    REQUIRED_HCAS=$(printf '%s' "$TRANSPORT_ENV" | sed -nE 's/.*NCCL_IB_HCA=([^ ]+).*/\1/p' | tr ',' ' ')
    for h in "$MASTER_HOST" "$WORKER_HOST"; do
      missing=""
      lib_check=$(ssh -n "$h" "
        $EXTRA_ENV
        if [ -z \"\${LD_LIBRARY_PATH:-}\" ] || ! ls \"\${LD_LIBRARY_PATH%%:*}/libibverbs.so.1\" >/dev/null 2>&1; then
          echo MISSING_LIBIBVERBS
          exit 0
        fi
        for d in $REQUIRED_HCAS; do
          [ -d \"/sys/class/infiniband/\$d\" ] || missing=\"\$missing \$d\"
        done
        if [ -n \"\$missing\" ]; then
          echo MISSING_HCAS:\$missing
        else
          echo OK
        fi
      ")
      case "$lib_check" in
        OK) ;;
        MISSING_LIBIBVERBS)
          echo "FATAL: $h has no libibverbs.so.1 on extra_env LD_LIBRARY_PATH (transport=$TRANSPORT)" >&2
          echo "  extra_env: $EXTRA_ENV" >&2
          exit 3 ;;
        MISSING_HCAS:*)
          echo "FATAL: $h missing required IB HCAs for transport=$TRANSPORT: ${lib_check#MISSING_HCAS:}" >&2
          echo "  expected: $REQUIRED_HCAS" >&2
          echo "  present:  $(ssh -n "$h" 'ls /sys/class/infiniband/ 2>/dev/null | tr "\n" " "')" >&2
          exit 3 ;;
        *)
          echo "FATAL: $h precheck returned unexpected: $lib_check" >&2
          exit 3 ;;
      esac
    done
    ;;
esac

# Ship snapshot.sh to both nodes (so it can read root-owned debugfs via sudo).
for h in "$MASTER_HOST" "$WORKER_HOST"; do
  # nix-store source is read-only; rm first so re-upload across rows works.
  ssh -n "$h" 'rm -f /tmp/snapshot.sh'
  scp -q "$SNAPSHOT_SH" "$h:/tmp/snapshot.sh"
done

snap() {
  # $1 = node, $2 = role tag (master|worker), $3 = stage tag (pre|post).
  # Writes $ROW_LOG_DIR/$3.$2.snap so delta_json can read by role.
  ssh -n "$1" 'sudo bash /tmp/snapshot.sh' > "$ROW_LOG_DIR/$3.$2.snap"
}

run_rank() {
  # $1 = node, $2 = rank, $3 = log file
  ssh -n "$1" "
    $TRANSPORT_ENV
    $EXTRA_ENV
    mkdir -p /tmp/finetune-bench && cd /tmp/finetune-bench
    EXTRA_ARGS=()
    if [ "$NO_EVAL" = "true" ]; then
      EXTRA_ARGS+=(--no-eval)
    fi
    $ENV_PATH/bin/strix-halo-finetune-torchrun \
      --nnodes=2 --nproc_per_node=1 --node_rank=$2 \
      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      $ENV_PATH/share/strix-halo-finetune/train.py \
      --model '$MODEL' --type '$TYPE' --strategy '$STRATEGY' \
      --dataset '$DATASET' --learning-rate '$LR' \
      --epochs $EPOCHS --batch-size $BATCH --max-length $MAXLEN \
      --gradient-accumulation $GA --max-steps $MAX_STEPS --no-save \
      \"\${EXTRA_ARGS[@]}\"
  " > "$3" 2>&1
}

echo ">>> row $ROW_IDX: transport=$TRANSPORT model=$MODEL type=$TYPE strategy=$STRATEGY dataset=$DATASET"

snap "$MASTER_HOST" master pre &
snap "$WORKER_HOST" worker pre &
wait

MASTER_LOG="$ROW_LOG_DIR/rank0.log"
WORKER_LOG="$ROW_LOG_DIR/rank1.log"

run_rank "$WORKER_HOST" 1 "$WORKER_LOG" &
WPID=$!
sleep 2
START=$(date +%s)
run_rank "$MASTER_HOST" 0 "$MASTER_LOG" || true
END=$(date +%s)
wait $WPID || true

snap "$MASTER_HOST" master post
snap "$WORKER_HOST" worker post

# Parse training metrics from rank 0 log.
read_metric() {
  # e.g. "{'train_runtime': '541'", grab the number after the key
  grep -oE "'$1':[[:space:]]*'[0-9.eE+-]+'" "$MASTER_LOG" | tail -1 | sed -E "s/.*'([0-9.eE+-]+)'/\1/" || true
}
TR=$(read_metric train_runtime)
TSPS=$(read_metric train_samples_per_second)
TSTS=$(read_metric train_steps_per_second)
TL=$(read_metric train_loss)
EL=$(read_metric eval_loss)
PEAK=$(grep -oE 'Peak:[[:space:]]+[0-9.]+ GB' "$MASTER_LOG" | tail -1 | grep -oE '[0-9.]+' || true)

# Compute deltas: pre/post snapshot files are flat KEY=VALUE; for each
# key present in both, emit (post - pre) into a JSON object.
delta_json() {
  python3 - "$ROW_LOG_DIR/pre.$1.snap" "$ROW_LOG_DIR/post.$1.snap" <<'PY'
import sys, json
pre = {}
post = {}
def load(p):
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out
pre = load(sys.argv[1])
post = load(sys.argv[2])
keys = set(pre) & set(post)
delta = {k: post[k] - pre[k] for k in sorted(keys) if post[k] != pre[k]}
json.dump(delta, sys.stdout)
PY
}

MASTER_DELTA=$(delta_json master)
WORKER_DELTA=$(delta_json worker)

OUT_JSON="$OUT_DIR/row-$ROW_IDX.json"
jq -n \
  --argjson row "$ROW_JSON" \
  --arg     run_id          "$RUN_ID" \
  --arg     ts              "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg     master_host     "$MASTER_HOST" \
  --arg     worker_host     "$WORKER_HOST" \
  --arg     wall_s          "$((END - START))" \
  --arg     train_runtime_s "${TR:-}" \
  --arg     samples_per_s   "${TSPS:-}" \
  --arg     steps_per_s     "${TSTS:-}" \
  --arg     train_loss      "${TL:-}" \
  --arg     eval_loss       "${EL:-}" \
  --arg     peak_gpu_gb     "${PEAK:-}" \
  --argjson master_delta    "$MASTER_DELTA" \
  --argjson worker_delta    "$WORKER_DELTA" \
  --arg     master_log      "$MASTER_LOG" \
  --arg     worker_log      "$WORKER_LOG" \
  '{
     run_id: $run_id,
     timestamp_utc: $ts,
     row: $row,
     hosts: { master: $master_host, worker: $worker_host },
     wall_s: ($wall_s | tonumber),
     training: {
       train_runtime_s: (if $train_runtime_s == "" then null else ($train_runtime_s | tonumber) end),
       samples_per_s:   (if $samples_per_s   == "" then null else ($samples_per_s   | tonumber) end),
       steps_per_s:     (if $steps_per_s     == "" then null else ($steps_per_s     | tonumber) end),
       train_loss:      (if $train_loss      == "" then null else ($train_loss      | tonumber) end),
       eval_loss:       (if $eval_loss       == "" then null else ($eval_loss       | tonumber) end),
       peak_gpu_gb:     (if $peak_gpu_gb     == "" then null else ($peak_gpu_gb     | tonumber) end)
     },
     delta: { master: $master_delta, worker: $worker_delta },
     log: { master: $master_log, worker: $worker_log }
   }' > "$OUT_JSON"

echo ">>> row $ROW_IDX done: ${TR:-?}s, $OUT_JSON"
