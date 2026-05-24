# strix-halo-vllm-pair-bench
#
# Drive the vllm-transport-matrix bench restricted to two headline
# scenarios used for kernel-module / transport regression testing on a
# strix-1/strix-2 pair:
#
#   qwen-peak      Qwen3-0.6B @ concurrency 256, TP=2 RDMA vs solo
#                  (peak aggregate throughput — ~2700 tok/s on old module).
#   llama-tp2-win  Meta-Llama-3.1-8B @ concurrency 256, TP=2 RDMA vs solo
#                  (~890 tok/s RDMA vs ~565 tok/s solo).
#
# All toolchain (gcc-wrapper, rdma-core-usb4, vllm-env, openssh, rsync)
# is built via nix and either baked into this derivation or `nix copy`'d
# to the master node — nothing is hunted on the host PATH.
#
# The transport-matrix script + python client come from the
# thunderbolt-ibverbs flake input (the canonical home for that bench
# infra). Test against a different tib commit by passing
# `--override-input thunderbolt-ibverbs path:/path/to/worktree` at
# nix-run time.
{
  lib,
  writeShellApplication,
  callPackage,
  coreutils,
  openssh,
  rsync,
  nixVersions,
  gcc,
  rdma-core-usb4,
  vllm-env-lemonade,
  thunderboltIbverbsSrc,
  packageSuffix,
}:

let
  # The matrix runner expects vllm-stream-client.py to sit next to itself
  # under SCRIPT_DIR. Pin both files into a tight nix-store dir so the
  # ssh-injected script can find them deterministically.
  benchSrc = thunderboltIbverbsSrc + "/userspace/bench";
in
writeShellApplication {
  name = "strix-halo-vllm-pair-bench-${packageSuffix}";
  runtimeInputs = [
    coreutils
    openssh
    rsync
    nixVersions.stable
  ];
  text = ''
    set -euo pipefail

    SCENARIO=""
    MASTER="grw@strix-1.lan.satanic.link"
    WORKER="grw@strix-2.lan.satanic.link"
    USB4_HCA="usb4_rdma0,usb4_rdma1,usb4_rdma5,usb4_rdma6"
    OUT_DIR=""
    DRY_RUN=0
    EXTRA_TRANSPORTS=""

    while [ "$#" -gt 0 ]; do
      case "$1" in
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --master)           MASTER="$2"; shift 2 ;;
        --worker)           WORKER="$2"; shift 2 ;;
        --usb4-hca)         USB4_HCA="$2"; shift 2 ;;
        --out)              OUT_DIR="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --extra-transports) EXTRA_TRANSPORTS="$2"; shift 2 ;;
        -h|--help)
          echo "Usage: $(basename "$0") --scenario {qwen-peak|llama-tp2-win} [opts]" >&2
          echo "  --master HOST       default: grw@strix-1.lan.satanic.link" >&2
          echo "  --worker HOST       default: grw@strix-2.lan.satanic.link" >&2
          echo "  --usb4-hca CSV      default: usb4_rdma0,usb4_rdma1,usb4_rdma5,usb4_rdma6" >&2
          echo "  --out DIR           default: ./out-vllm-<scenario>-<ts>" >&2
          echo "  --extra-transports STR  e.g. \"lan_tcp tb_tcp\" to add comparisons" >&2
          echo "  --dry-run           print env + plan, don't invoke" >&2
          exit 0 ;;
        *)
          echo "unknown arg: $1" >&2; exit 2 ;;
      esac
    done

    case "$SCENARIO" in
      qwen-peak)
        MODELS="Qwen/Qwen3-0.6B"
        BASE_TRANSPORTS="solo usb4_rdma"
        CONCURRENCIES="256"
        MAX_TOKENS=64
        ;;
      llama-tp2-win)
        MODELS="unsloth/Meta-Llama-3.1-8B-Instruct"
        BASE_TRANSPORTS="solo usb4_rdma"
        CONCURRENCIES="256"
        MAX_TOKENS=64
        ;;
      *)
        echo "scenario must be one of: qwen-peak | llama-tp2-win (got '$SCENARIO')" >&2
        exit 2 ;;
    esac

    TRANSPORTS="$BASE_TRANSPORTS''${EXTRA_TRANSPORTS:+ $EXTRA_TRANSPORTS}"
    if [ -z "$OUT_DIR" ]; then
      OUT_DIR="./out-vllm-$SCENARIO-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    mkdir -p "$OUT_DIR"

    # Nix-built closures (baked at derivation time).
    GCC_PREFIX=${gcc}
    RDMA=${rdma-core-usb4}
    VLLM_ENV=${vllm-env-lemonade}
    BENCH_DIR=${benchSrc}

    for required in \
      "$GCC_PREFIX/bin/gcc" \
      "$RDMA/lib" \
      "$VLLM_ENV/bin/vllm" \
      "$BENCH_DIR/vllm-transport-matrix.sh" \
      "$BENCH_DIR/vllm-stream-client.py"
    do
      [ -e "$required" ] || { echo "FATAL: missing closure path: $required" >&2; exit 2; }
    done

    echo "scenario:        $SCENARIO"
    echo "master:          $MASTER"
    echo "worker:          $WORKER"
    echo "transports:      $TRANSPORTS"
    echo "concurrencies:   $CONCURRENCIES"
    echo "usb4_hca:        $USB4_HCA"
    echo "vllm-env:        $VLLM_ENV"
    echo "gcc-wrapper:     $GCC_PREFIX"
    echo "rdma-core-usb4:  $RDMA"
    echo "bench-dir:       $BENCH_DIR"
    echo "out:             $OUT_DIR"

    echo
    echo "=== copying closures to master ($MASTER) ==="
    nix copy --no-check-sigs --to "ssh://$MASTER" \
      "$VLLM_ENV" "$RDMA" "$GCC_PREFIX" "$BENCH_DIR"

    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$SCENARIO"
    REMOTE_LOG_DIR="/tmp/vllm-pair-$RUN_ID"
    REMOTE_CSV="$REMOTE_LOG_DIR/vllm-pair-$RUN_ID.csv"

    # The matrix script reads ~20 env vars; build them once as a single
    # block so we can dry-print and ssh-inject the same content.
    REMOTE_ENV=$(cat <<EOF
    VLLM_ENV=$VLLM_ENV
    GCCBIN=$GCC_PREFIX/bin
    RDMA=$RDMA
    SSH_HOST=$WORKER
    LAN_IFNAME=br0.lan
    TB_IFNAME=thunderbolt0
    USB4_HCA=$USB4_HCA
    MODELS="$MODELS"
    TRANSPORTS="$TRANSPORTS"
    CONCURRENCIES="$CONCURRENCIES"
    MAX_TOKENS=$MAX_TOKENS
    SAMPLE_CASES=0
    SETUP_RXE=0
    RUN_ID=$RUN_ID
    LOG_DIR=$REMOTE_LOG_DIR
    OUTPUT_CSV=$REMOTE_CSV
    EOF
    )

    echo
    echo "=== invoking matrix script on master ==="
    printf 'env: %s\n' "$REMOTE_ENV"

    if [ "$DRY_RUN" = "1" ]; then
      echo "(dry-run: not invoking)"
      exit 0
    fi

    # Word-split REMOTE_ENV (newline-separated KEY=VALUE lines) into
    # individual env args. Need to read it line-by-line into an array to
    # keep KEY="multi word value" intact.
    REMOTE_ENV_ARRAY=()
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      REMOTE_ENV_ARRAY+=("$line")
    done <<< "$REMOTE_ENV"
    # SC2029: $REMOTE_ENV_ARRAY expanding locally is intentional — those
    # are the env vars we want injected on the remote shell.
    # shellcheck disable=SC2029
    ssh "$MASTER" "env ''${REMOTE_ENV_ARRAY[*]} bash $BENCH_DIR/vllm-transport-matrix.sh"

    echo
    echo "=== fetching results ==="
    rsync -av "$MASTER:$REMOTE_LOG_DIR/" "$OUT_DIR/"
    echo
    echo "CSV: $OUT_DIR/vllm-pair-$RUN_ID.csv"
  '';

  meta = with lib; {
    description = "Two headline vLLM transport-matrix scenarios across a Strix Halo pair (for kernel-module regression)";
    platforms = platforms.linux;
  };
}
