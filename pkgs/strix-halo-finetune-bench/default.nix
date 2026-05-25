{
  lib,
  writeShellApplication,
  writeText,
  coreutils,
  openssh,
  jq,
  python3,
  nixVersions,
  strix-halo-finetune-env,
  packageSuffix,
  packageName ? "strix-halo-finetune-bench-${packageSuffix}",

  # Default cluster topology. Override with `.override { ... }` per host
  # group; the wrapper script also takes --master/--worker CLI flags.
  defaultMaster ? "grw@strix-1.lan.satanic.link",
  defaultWorker ? "grw@strix-2.lan.satanic.link",
  defaultMasterAddr ? "192.168.23.136",

  # The benchmark matrix. Each entry is a config dict for one row.
  # Override per project from the flake (e.g. blog post) to extend or
  # subset the matrix without forking the package. Default is a
  # 3-transport × full-FT × SmolLM2-135M sweep matching the headline
  # numbers we plot in the blog post.
  matrix ? [
    {
      transport = "eth";
      model = "HuggingFaceTB/SmolLM2-135M-Instruct";
      ft_type = "full";
    }
    {
      transport = "rdma";
      model = "HuggingFaceTB/SmolLM2-135M-Instruct";
      ft_type = "full";
      # libibverbs has to be on each node's LD_LIBRARY_PATH for RCCL's
      # dlopen path. Best-effort auto-detect from the nix store.
      extra_env = ''export LD_LIBRARY_PATH="$(find /nix/store -maxdepth 1 -type d -name '*-rdma-core-usb4-*[0-9]' 2>/dev/null | sort -u | tail -1)/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'';
    }
    {
      transport = "rdma-4hca";
      model = "HuggingFaceTB/SmolLM2-135M-Instruct";
      ft_type = "full";
      extra_env = ''export LD_LIBRARY_PATH="$(find /nix/store -maxdepth 1 -type d -name '*-rdma-core-usb4-*[0-9]' 2>/dev/null | sort -u | tail -1)/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'';
    }
  ],
}:

let
  rowsJsonl = writeText "finetune-bench-matrix-${packageSuffix}.jsonl" (
    lib.concatMapStrings (r: builtins.toJSON r + "\n") matrix
  );

in
writeShellApplication {
  name = packageName;
  runtimeInputs = [
    coreutils
    openssh
    jq
    python3
    nixVersions.stable
  ];
  text = ''
    # Usage: strix-halo-finetune-bench-<arch> [--master HOST] [--worker HOST] [--master-addr IP] [--out DIR] [--run-id NAME]
    MASTER="${defaultMaster}"
    WORKER="${defaultWorker}"
    MASTER_ADDR="${defaultMasterAddr}"
    OUT_DIR="''${PWD}/finetune-bench-$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${packageSuffix}"
    DRY_RUN=0

    while [ "$#" -gt 0 ]; do
      case "$1" in
        --master)      MASTER="$2"; shift 2 ;;
        --worker)      WORKER="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --out)         OUT_DIR="$2"; shift 2 ;;
        --run-id)      RUN_ID="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)
          echo "Usage: $(basename "$0") [--master HOST] [--worker HOST] [--master-addr IP] [--out DIR] [--run-id NAME] [--dry-run]"; exit 0 ;;
        *)
          echo "unknown arg: $1"; exit 2 ;;
      esac
    done

    mkdir -p "$OUT_DIR"
    echo "run_id:      $RUN_ID"
    echo "matrix:      ${rowsJsonl} ($(wc -l < ${rowsJsonl}) rows)"
    echo "master:      $MASTER"
    echo "worker:      $WORKER"
    echo "master_addr: $MASTER_ADDR"
    echo "out:         $OUT_DIR"

    if [ "$DRY_RUN" = "1" ]; then
      echo
      echo "rows:"
      jq -c . ${rowsJsonl}
      exit 0
    fi

    echo
    echo "=== copying closures to hosts ($MASTER, $WORKER) ==="
    for host in "$MASTER" "$WORKER"; do
      nix copy --no-check-sigs --to "ssh://$host" \
        "${strix-halo-finetune-env}" "${rowsJsonl}" "${./snapshot.sh}" "${./run-row.sh}" "${./aggregate.py}"
    done

    PORT=29600
    IDX=0
    while IFS= read -r ROW_JSON; do
      [ -n "$ROW_JSON" ] || continue
      ROW_JSON="$ROW_JSON" \
      ENV_PATH="${strix-halo-finetune-env}" \
      MASTER_HOST="$MASTER" \
      WORKER_HOST="$WORKER" \
      MASTER_ADDR="$MASTER_ADDR" \
      MASTER_PORT="$PORT" \
      OUT_DIR="$OUT_DIR" \
      SNAPSHOT_SH="${./snapshot.sh}" \
      RUN_ID="$RUN_ID" \
      ROW_IDX="$IDX" \
        bash ${./run-row.sh}
      IDX=$((IDX + 1))
      PORT=$((PORT + 1))
    done < ${rowsJsonl}

    # Aggregate every per-row JSON into a single flat CSV (sparse).
    python3 ${./aggregate.py} \
      "$OUT_DIR/finetune-bench-$RUN_ID.csv" \
      "$OUT_DIR"/row-*.json

    echo
    echo "CSV: $OUT_DIR/finetune-bench-$RUN_ID.csv"
    echo "per-row JSON + logs: $OUT_DIR/"
  '';

  meta = with lib; {
    description = "Run the 2-node FSDP fine-tuning benchmark matrix on a strix-halo cluster";
    platforms = platforms.linux;
  };
}
