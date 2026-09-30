#!/usr/bin/env bash
# CPU regression for lib/bench/ds41-node.sh.
#
# The mandatory model identity/RDMA preflight must run under the interpreter
# shipped with the DS41_BINARY runtime, not a python3 from PATH. A minimal Strix
# node may have no python3 on PATH, so the launcher must still reach the
# preflight and must not start serving when that preflight fails.
#
# The fake runtime only records invocations: no GPU, no inference server and no
# host configuration is touched.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher=$script_dir/ds41-node.sh
bash_bin=${BASH:?run this check with bash}

fail() { echo "ds41-node launcher check: $*" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

runtime_bin=$tmp/runtime/bin
mkdir -p "$runtime_bin"

# A PATH with the shell tools the launcher needs but without any python3.
tools=$tmp/tools
mkdir -p "$tools"
for tool in dirname basename realpath mkdir; do
  tool_path=$(command -v "$tool") || fail "test host is missing $tool"
  ln -s -- "$tool_path" "$tools/$tool"
done
if PATH=$tools command -v python3 >/dev/null 2>&1; then
  fail "restricted PATH unexpectedly exposes python3"
fi

model=$tmp/models/DeepSeek-V4.1-Flash-hf-dba1be0a
cache=$tmp/cache
harness=$tmp/harness
mkdir -p "$model" "$harness"
preflight_log=$harness/preflight.log
served_log=$harness/served.log

cat > "$runtime_bin/sglang" <<EOF
#!$bash_bin
echo served >> "$served_log"
exit 0
EOF
chmod +x "$runtime_bin/sglang"

write_preflight() {
  cat > "$runtime_bin/sglang-python" <<EOF
#!$bash_bin
printf '%s\n' "\$*" >> "$preflight_log"
exit $1
EOF
  chmod +x "$runtime_bin/sglang-python"
}

run_launcher() {
  env -i \
    PATH="$tools" \
    HOME="$harness" \
    DS41_NODE_RANK=0 \
    DS41_BINARY="$runtime_bin/sglang" \
    DS41_MODEL_PATH="$model" \
    DS41_CACHE_ROOT="$cache" \
    "$@" \
    "$bash_bin" "$launcher"
}

# A failing preflight must use the runtime interpreter and abort before serving.
write_preflight 1
if run_launcher >/dev/null 2>&1; then
  fail "launcher served despite a failed model preflight"
fi
[[ -s "$preflight_log" ]] || fail "runtime sglang-python preflight was not invoked"
grep -Fq 'ds41-model-check.py' "$preflight_log" || fail "preflight did not run the model check"
grep -Fq "$model" "$preflight_log" || fail "preflight did not receive the model path"
[[ ! -e "$served_log" ]] || fail "serving started after a failed model preflight"

# Without any preflight interpreter at all the launcher must still fail closed.
rm -f "$runtime_bin/sglang-python" "$served_log"
if run_launcher >/dev/null 2>&1; then
  fail "launcher served without a preflight interpreter"
fi
[[ ! -e "$served_log" ]] || fail "serving started without a preflight interpreter"

# Observe the real launcher's exported paths at mkdir, then stop before any
# host RDMA/lock checks. This does not depend on the test host lacking an HCA.
write_preflight 0
cache_log=$harness/cache.log
rm "$tools/mkdir"
cat > "$tools/mkdir" <<EOF
#!$bash_bin
printf '%s\\n' "\$XDG_CACHE_HOME" "\$SGLANG_CACHE_DIR" "\$TRITON_CACHE_DIR" \\
  "\$TORCHINDUCTOR_CACHE_DIR" "\$AITER_JIT_DIR" "\$AITER_ROOT_DIR" > "$cache_log"
exit 73
EOF
chmod +x "$tools/mkdir"

check_cache_paths() {
  local expected_triton=$1 rc=0
  shift
  rm -f "$cache_log"
  run_launcher "$@" >"$harness/cache-output.log" 2>&1 || rc=$?
  [[ $rc == 73 ]] || fail "launcher did not reach the cache probe: exit $rc"
  printf '%s\n' "$cache/runtime/cache" "$cache/runtime/sglang" "$expected_triton" \
    "$cache/runtime/torch" "$cache/runtime/aiter-jit" "$cache/runtime/aiter-root" \
    > "$harness/expected-cache.log"
  cmp -s "$harness/expected-cache.log" "$cache_log" || fail "unexpected cache paths"
  [[ ! -e "$served_log" ]] || fail "cache probe started serving"
}

check_cache_paths "$cache/runtime/triton"
check_cache_paths "$cache/runtime/triton" TRITON_CACHE_DIR=
check_cache_paths "$tmp/shared triton" "TRITON_CACHE_DIR=$tmp/shared triton"

rm -f "$cache_log"
rc=0
run_launcher TRITON_CACHE_DIR=relative/triton >"$harness/cache-output.log" 2>&1 || rc=$?
[[ $rc == 2 ]] || fail "relative Triton cache was not rejected: exit $rc"
grep -Fq 'TRITON_CACHE_DIR must be an absolute directory' "$harness/cache-output.log" ||
  fail "missing invalid cache path diagnostic"
[[ ! -e "$cache_log" && ! -e "$served_log" ]] ||
  fail "invalid cache path reached cache creation or serving"

echo "ds41-node launcher check passed"
