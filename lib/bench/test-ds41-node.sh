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

echo "ds41-node launcher check passed"
