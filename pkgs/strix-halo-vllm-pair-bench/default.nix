{
  lib,
  writeShellApplication,
  runCommand,
  coreutils,
  openssh,
  rsync,
  nixVersions,
  gcc,
  rdma-core,
  vllmPackage,
  packageSuffix,
  vllmTransportMatrix ? ../../lib/bench/vllm-transport-matrix.sh,
  vllmStreamClient ? ../../lib/bench/vllm-stream-client.py,
}:

let
  benchSrc = runCommand "strix-halo-vllm-matrix-bench-src" { } ''
    mkdir -p "$out"
    cp ${vllmTransportMatrix} "$out/vllm-transport-matrix.sh"
    cp ${vllmStreamClient} "$out/vllm-stream-client.py"
    chmod +x "$out/vllm-transport-matrix.sh" "$out/vllm-stream-client.py"
  '';
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
    PROMPT_SECTIONS=36
    MAX_MODEL_LEN=2048
    MAX_NUM_SEQS=512
    MAX_NUM_BATCHED_TOKENS=""
    VLLM_EXTRA_ARGS=""
    SCENARIO_EXTRA_ENV=""
    CLIENT_TIMEOUT_S=1800
    STARTUP_TIMEOUT_S=600
    GCC_PREFIX=${gcc}
    RDMA=${rdma-core}
    VLLM_ENV=${vllmPackage}
    BENCH_DIR=${benchSrc}

    usage() {
      cat >&2 <<USAGE
    Usage: $(basename "$0") --scenario SCENARIO [opts]

    Scenarios:
      qwen-peak
      llama-tp2-win
      qwen35-122b-awq-capacity
      qwen35-122b-awq-prime
      minimax-m27-awq-strix-2h

    Options:
      --master HOST             default: $MASTER
      --worker HOST             default: $WORKER
      --usb4-hca CSV            default: $USB4_HCA
      --out DIR                 default: ./out-vllm-<scenario>-<timestamp>
      --extra-transports LIST   e.g. "lan_tcp tb_tcp"
      --vllm-env PATH           vLLM environment to copy/use on both hosts
      --rdma-prefix PATH        RDMA prefix to copy/use on both hosts
      --gcc-prefix PATH         GCC prefix to copy/use on both hosts
      --bench-dir PATH          matrix script directory to copy/use on both hosts
      --dry-run                 print env + plan without SSH execution
      -h, --help                show this help
    USAGE
    }

    while [ "$#" -gt 0 ]; do
      case "$1" in
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --master)           MASTER="$2"; shift 2 ;;
        --worker)           WORKER="$2"; shift 2 ;;
        --usb4-hca)         USB4_HCA="$2"; shift 2 ;;
        --out)              OUT_DIR="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --extra-transports) EXTRA_TRANSPORTS="$2"; shift 2 ;;
        --vllm-env)         VLLM_ENV="$2"; shift 2 ;;
        --rdma-prefix)      RDMA="$2"; shift 2 ;;
        --gcc-prefix)       GCC_PREFIX="$2"; shift 2 ;;
        --bench-dir)        BENCH_DIR="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "unknown arg: $1" >&2; usage; exit 2 ;;
      esac
    done

    case "$SCENARIO" in
      qwen-peak)
        MODELS="Qwen/Qwen3-0.6B"
        BASE_TRANSPORTS="solo usb4_rdma"
        CONCURRENCIES="256"
        MAX_TOKENS=256
        ;;
      llama-tp2-win)
        MODELS="unsloth/Meta-Llama-3.1-8B-Instruct"
        BASE_TRANSPORTS="solo usb4_rdma"
        CONCURRENCIES="256"
        MAX_TOKENS=256
        ;;
      qwen35-122b-awq-capacity)
        MODELS="cyankiwi/Qwen3.5-122B-A10B-AWQ-8bit"
        BASE_TRANSPORTS="lan_tcp usb4_rdma"
        CONCURRENCIES="1 2 4 8 16"
        MAX_TOKENS=1024
        PROMPT_SECTIONS=72
        MAX_MODEL_LEN=8192
        MAX_NUM_SEQS=16
        MAX_NUM_BATCHED_TOKENS=16384
        VLLM_EXTRA_ARGS="--language-model-only"
        CLIENT_TIMEOUT_S=3600
        STARTUP_TIMEOUT_S=1800
        ;;
      qwen35-122b-awq-prime)
        MODELS="cyankiwi/Qwen3.5-122B-A10B-AWQ-8bit"
        BASE_TRANSPORTS="lan_tcp"
        CONCURRENCIES="1 4 16"
        MAX_TOKENS=64
        PROMPT_SECTIONS=72
        MAX_MODEL_LEN=8192
        MAX_NUM_SEQS=16
        MAX_NUM_BATCHED_TOKENS=16384
        VLLM_EXTRA_ARGS="--language-model-only"
        CLIENT_TIMEOUT_S=3600
        STARTUP_TIMEOUT_S=1800
        ;;
      minimax-m27-awq-strix-2h)
        MODELS="ayysasha/MiniMax-M2.7-AWQ-G32-STRIX-2H"
        BASE_TRANSPORTS="usb4_rdma lan_tcp"
        CONCURRENCIES="1 2"
        MAX_TOKENS=1024
        PROMPT_SECTIONS=72
        MAX_MODEL_LEN=196608
        MAX_NUM_SEQS=2
        MAX_NUM_BATCHED_TOKENS=20480
        VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.92 --dtype auto --load-format safetensors --trust-remote-code --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think --enable-auto-tool-choice"
        SCENARIO_EXTRA_ENV=$'VLLM_ROCM_USE_AITER=0\nOMP_NUM_THREADS=1\nTOKENIZERS_PARALLELISM=false\nTORCHDYNAMO_DISABLE=1\nRAY_CGRAPH_get_timeout=1800\nVLLM_SLEEP_WHEN_IDLE=1\nVLLM_USE_DEEP_GEMM=0\nVLLM_USE_FLASHINFER_SAMPLER=0\nVLLM_USE_FLASHINFER_MOE_FP16=0'
        CLIENT_TIMEOUT_S=7200
        STARTUP_TIMEOUT_S=3600
        ;;
      *)
        echo "scenario must be one of: qwen-peak | llama-tp2-win | qwen35-122b-awq-capacity | qwen35-122b-awq-prime | minimax-m27-awq-strix-2h (got '$SCENARIO')" >&2
        exit 2 ;;
    esac

    TRANSPORTS="$BASE_TRANSPORTS''${EXTRA_TRANSPORTS:+ $EXTRA_TRANSPORTS}"
    if [ -z "$OUT_DIR" ]; then
      OUT_DIR="./out-vllm-$SCENARIO-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    mkdir -p "$OUT_DIR"

    for required in \
      "$GCC_PREFIX/bin/gcc" \
      "$RDMA/lib" \
      "$VLLM_ENV/bin/vllm" \
      "$VLLM_ENV/bin/ray" \
      "$BENCH_DIR/vllm-transport-matrix.sh" \
      "$BENCH_DIR/vllm-stream-client.py"
    do
      [ -e "$required" ] || { echo "FATAL: missing path: $required" >&2; exit 2; }
    done

    echo "scenario:        $SCENARIO"
    echo "master:          $MASTER"
    echo "worker:          $WORKER"
    echo "transports:      $TRANSPORTS"
    echo "concurrencies:   $CONCURRENCIES"
    echo "max_tokens:      $MAX_TOKENS"
    echo "prompt_sections: $PROMPT_SECTIONS"
    echo "max_model_len:   $MAX_MODEL_LEN"
    echo "max_num_seqs:    $MAX_NUM_SEQS"
    echo "batched_tokens:  ''${MAX_NUM_BATCHED_TOKENS:-<default>}"
    echo "vllm_extra_args: ''${VLLM_EXTRA_ARGS:-<none>}"
    echo "usb4_hca:        $USB4_HCA"
    echo "vllm-env:        $VLLM_ENV"
    echo "gcc-wrapper:     $GCC_PREFIX"
    echo "rdma-core:       $RDMA"
    echo "bench-dir:       $BENCH_DIR"
    echo "out:             $OUT_DIR"

    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$SCENARIO"
    REMOTE_LOG_DIR="/tmp/vllm-pair-$RUN_ID"
    REMOTE_CSV="$REMOTE_LOG_DIR/vllm-pair-$RUN_ID.csv"

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
    MAX_MODEL_LEN=$MAX_MODEL_LEN
    MAX_NUM_SEQS=$MAX_NUM_SEQS
    MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
    PROMPT_SECTIONS=$PROMPT_SECTIONS
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS"
    CLIENT_TIMEOUT_S=$CLIENT_TIMEOUT_S
    STARTUP_TIMEOUT_S=$STARTUP_TIMEOUT_S
    SAMPLE_CASES=0
    SETUP_RXE=0
    RUN_ID=$RUN_ID
    LOG_DIR=$REMOTE_LOG_DIR
    OUTPUT_CSV=$REMOTE_CSV
    NCCL_IB_GID_INDEX=1
    NCCL_NET_MERGE_LEVEL=LOC
    NCCL_MIN_NCHANNELS=4
    NCCL_MAX_NCHANNELS=4
    NCCL_IB_QPS_PER_CONNECTION=1
    NCCL_IB_SPLIT_DATA_ON_QPS=0
    NCCL_DEBUG_LEVEL=INFO
    NCCL_DEBUG_SUBSYS=INIT,NET,ENV
    CUDA_VISIBLE_DEVICES=0
    HIP_VISIBLE_DEVICES=0
    ROCR_VISIBLE_DEVICES=0
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
    RAY_DEDUP_LOGS=0
    $SCENARIO_EXTRA_ENV
    EOF
    )

    echo
    echo "=== planned matrix invocation ==="
    printf '%s\n' "$REMOTE_ENV"

    if [ "$DRY_RUN" = "1" ]; then
      echo "(dry-run: not invoking)"
      exit 0
    fi

    echo
    echo "=== copying closures to hosts ($MASTER, $WORKER) ==="
    copy_paths=()
    for path in "$VLLM_ENV" "$RDMA" "$GCC_PREFIX" "$BENCH_DIR"; do
      case "$path" in
        /nix/store/*) copy_paths+=("$path") ;;
        *) echo "not a nix-store path; remote must already provide: $path" >&2 ;;
      esac
    done

    for host in "$MASTER" "$WORKER"; do
      if [ "''${#copy_paths[@]}" -gt 0 ]; then
        nix copy --no-check-sigs --to "ssh://$host" "''${copy_paths[@]}"
      fi
    done

    echo
    echo "=== invoking matrix script on master ==="
    env_file="$OUT_DIR/remote-env"
    printf '%s\n' "$REMOTE_ENV" > "$env_file"
    printf -v remote_log_dir_q '%q' "$REMOTE_LOG_DIR"
    printf -v bench_script_q '%q' "$BENCH_DIR/vllm-transport-matrix.sh"
    # shellcheck disable=SC2029
    ssh "$MASTER" "mkdir -p $remote_log_dir_q"
    scp "$env_file" "$MASTER:$REMOTE_LOG_DIR/env"
    # shellcheck disable=SC2029
    ssh "$MASTER" "set -a; . $remote_log_dir_q/env; set +a; exec bash $bench_script_q"

    echo
    echo "=== fetching results ==="
    rsync -av "$MASTER:$REMOTE_LOG_DIR/" "$OUT_DIR/"
    echo
    echo "CSV: $OUT_DIR/vllm-pair-$RUN_ID.csv"
  '';

  meta = with lib; {
    description = "vLLM transport-matrix benchmark driver for a two-host Strix Halo pair";
    mainProgram = "strix-halo-vllm-pair-bench-${packageSuffix}";
    platforms = platforms.linux;
  };
}
