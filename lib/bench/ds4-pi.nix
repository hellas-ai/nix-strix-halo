{
  pkgs,
  ds4Package,
  piWrapPackage,
  modelsRoot ? null,
  modelRoot ? null,
  modelPath ? null,
}:

let
  inherit (pkgs) lib;
  benchLib = import ./lib.nix { inherit lib; };

  resolvedModelsRoot = if modelsRoot != null then modelsRoot else benchLib.defaultModelsRoot pkgs;
  resolvedModelRoot =
    if modelRoot != null then modelRoot else benchLib.modelPath resolvedModelsRoot [ "ds4" ];
  resolvedModelPath =
    if modelPath != null then modelPath else benchLib.modelPath resolvedModelRoot [ "ds4flash.gguf" ];

  marker = "ds4-pi-smoke-ok";

  runner = pkgs.writeShellScript "ds4-metal-pi-smoke-runner" ''
    set -euo pipefail

    mkdir -p "$out" "$TMPDIR/home" "$TMPDIR/ds4-kv"
    export HOME="$TMPDIR/home"
    export DS4_METAL_NO_RESIDENCY=1
    export DS4_METAL_NO_MODEL_WARMUP=1
    export DS4_SERVER_PERFLEVEL=skip
    # Shared darwin /tmp: a stale 0600 lock from another _nixbld user kills
    # the run. Use the per-build TMPDIR.
    export DS4_LOCK_FILE="$TMPDIR/ds4.lock"

    port="$(${pkgs.python3}/bin/python3 - <<'PY'
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        print(s.getsockname()[1])
    PY
    )"

    echo "ds4-pi-smoke: host=$(${pkgs.coreutils}/bin/uname -n) port=$port lock=$DS4_LOCK_FILE" >&2

    server_log="$out/ds4-server.log"
    ${ds4Package}/bin/ds4-server \
      --metal \
      -m ${lib.escapeShellArg resolvedModelPath} \
      --host 127.0.0.1 \
      --port "$port" \
      --ctx 4096 \
      --tokens 128 \
      --kv-disk-dir "$TMPDIR/ds4-kv" \
      --kv-disk-space-mb 256 \
      > "$server_log" 2>&1 &
    server_pid=$!

    cleanup() {
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup EXIT

    ready=0
    for _ in $(seq 1 300); do
      if ${pkgs.curl}/bin/curl -fsS --connect-timeout 1 --max-time 2 "http://127.0.0.1:$port/v1/models" > "$out/models.json" 2> "$out/readiness.log"; then
        ready=1
        break
      fi
      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ds4-server exited before becoming ready" >&2
        cat "$server_log" >&2 || true
        exit 1
      fi
      sleep 1
    done

    if [ "$ready" -ne 1 ]; then
      echo "ds4-server did not become ready on port $port" >&2
      cat "$out/readiness.log" >&2 || true
      cat "$server_log" >&2 || true
      exit 1
    fi

    if ! ${pkgs.jq}/bin/jq -e '.data[] | select(.id == "deepseek-v4-flash")' "$out/models.json" >/dev/null; then
      echo "ds4-server did not report the expected OpenAI model id" >&2
      cat "$out/models.json" >&2 || true
      cat "$server_log" >&2 || true
      exit 1
    fi

    set +e
    OPENAI_BASE_URL="http://127.0.0.1:$port/v1" \
    OPENAI_API_KEY=unused \
    OPENAI_MODEL=deepseek-v4-flash \
      ${piWrapPackage}/bin/pi-wrap \
        -p \
        --no-session \
        --offline \
        --verbose \
        "Reply with exactly: ${marker}" \
        > "$out/pi-output.txt" 2> "$out/pi-stderr.txt"
    pi_status=$?
    set -e

    if [ "$pi_status" -ne 0 ]; then
      echo "pi-wrap failed with exit status $pi_status" >&2
      cat "$out/pi-output.txt" >&2 || true
      cat "$out/pi-stderr.txt" >&2 || true
      cat "$server_log" >&2 || true
      exit "$pi_status"
    fi

    if ! grep -F ${lib.escapeShellArg marker} "$out/pi-output.txt" >/dev/null; then
      echo "pi-wrap output did not contain the smoke marker" >&2
      cat "$out/pi-output.txt" >&2 || true
      cat "$out/pi-stderr.txt" >&2 || true
      cat "$server_log" >&2 || true
      exit 1
    fi
  '';

  benchmark = benchLib.mkBenchmark {
    inherit pkgs;
    name = "ds4-metal-pi-smoke";
    packages = [
      ds4Package
      piWrapPackage
      pkgs.curl
      pkgs.jq
      pkgs.python3
    ];
    command = [ runner ];
    env = {
      DS4_MODEL = resolvedModelPath;
      DS4_SERVER_PERFLEVEL = "skip";
      DS4_METAL_NO_RESIDENCY = "1";
      DS4_METAL_NO_MODEL_WARMUP = "1";
    };
    requirements = {
      systemFeatures = [ "metal" ];
      hostProfiles = [ "darwin-metal" ];
      sandboxPaths = [ resolvedModelRoot ];
    };
    metadata = {
      kind = "ds4-pi";
      suite = "ds4";
      accelerator = "metal";
      scenario = "pi-smoke";
      model = {
        name = "deepseek-v4-flash";
        path = resolvedModelPath;
        repo = "antirez/deepseek-v4-gguf";
      };
      tool = {
        backend = "metal";
        packageRole = "ds4";
        client = "pi";
      };
    };
    meta.platforms = [ "aarch64-darwin" ];
    description = "Run Pi against a local DS4 Metal server";
  };
in
{
  benchmarks.deepseek-v4-flash.ds4-metal-pi-smoke = benchmark;
}
