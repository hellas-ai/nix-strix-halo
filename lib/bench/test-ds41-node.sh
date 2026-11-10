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

# Valid draft profiles reach the pinned runtime preflight with their gamma.
# Invalid profiles fail before metadata reads, caches or host/GPU operations.
for gamma in 1 3; do
  : > "$preflight_log"
  if run_launcher DS41_DSPARK_GAMMA=$gamma >/dev/null 2>&1; then
    fail "launcher served despite a failed draft preflight"
  fi
  grep -Fq -- "--dspark-gamma $gamma" "$preflight_log" || fail "draft preflight omitted gamma"
done
for gamma in -1 2 4 5 nonsense; do
  : > "$preflight_log"
  rc=0
  run_launcher DS41_DSPARK_GAMMA=$gamma >"$harness/profile.log" 2>&1 || rc=$?
  [[ $rc == 2 && ! -s "$preflight_log" ]] || fail "invalid gamma reached preflight"
done
for interval in '' 0 -1 1.5 bad; do
  : > "$preflight_log"
  rc=0
  run_launcher "DS41_PREFILL_DECODE_INTERVAL=$interval" >"$harness/profile.log" 2>&1 || rc=$?
  [[ $rc == 2 && ! -s "$preflight_log" && ! -e "$served_log" ]] ||
    fail "invalid prefill/decode interval reached preflight or serving"
done
: > "$preflight_log"
rc=0
run_launcher DS41_DSPARK_GAMMA=3 SGLANG_RAGGED_VERIFY_MODE=compact >"$harness/profile.log" 2>&1 || rc=$?
[[ $rc == 2 && ! -s "$preflight_log" ]] || fail "ragged verify reached preflight"

# Decode graph buckets. Default arms print exactly the graph lists the launcher always shipped;
# the exact arm lists one graph per request count and opts the runtime in. Resolving a profile
# never reaches the model preflight, the host checks or the server.
profile_for() {
  : > "$preflight_log"
  rm -f "$served_log"
  run_launcher DS41_PRINT_PROFILE=1 "$@" 2>&1
}
expect_profile() {
  local running=$1 graphs=$2 exact=$3 got rc=0 gamma=0 chunk=1536 interval=32 replay=0 setting
  shift 3
  for setting in "$@"; do
    case "$setting" in
      DS41_DSPARK_GAMMA=*) gamma=${setting#*=} ;;
      DS41_CHUNKED_PREFILL_SIZE=*) chunk=${setting#*=} ;;
      DS41_PREFILL_DECODE_INTERVAL=*) interval=${setting#*=} ;;
      DS41_DECODER_SWA_BOUNDED_REPLAY=*) replay=${setting#*=} ;;
    esac
  done
  got=$(profile_for "$@") || rc=$?
  [[ $rc == 0 ]] || fail "profile $* failed: exit $rc: $got"
  [[ $got == "max_running_requests=$running
graph_config={\"decode\":{\"backend\":\"full\",\"bs\":[$graphs],\"max_bs\":$running}}
exact_graph_buckets=$exact
dspark_policy=fixed
dspark_gamma=$gamma
adaptive_verify=unset
chunked_prefill_size=$chunk
prefill_decode_interval=$interval
disable_overlap_schedule=0
fp4_logits_budget_mb=64
decoder_swa_bounded_replay=$replay" ]] || fail "unexpected profile for $*: $got"
  [[ ! -s "$preflight_log" && ! -e "$served_log" ]] || fail "profile $* reached the preflight or the server"
}
reject_profile() {
  local rc=0
  profile_for "$@" >"$harness/profile.log" || rc=$?
  [[ $rc == 2 ]] || fail "profile $* was not rejected: exit $rc"
  [[ ! -s "$preflight_log" && ! -e "$served_log" ]] || fail "rejected profile $* reached the preflight or the server"
}
expect_profile 4 "1,2,4" unset
expect_profile 4 "1,2,4" unset DS41_CHUNKED_PREFILL_SIZE=4096
expect_profile 4 "1,2,4" unset DS41_CHUNKED_PREFILL_SIZE=1
expect_profile 4 "1,2,4" unset DS41_PREFILL_DECODE_INTERVAL=8
expect_profile 4 "1,2,4" unset DS41_GRAPH_BUCKETS=padded
expect_profile 4 "1,2,4" unset DS41_DSPARK_GAMMA=1
expect_profile 2 "1,2" unset DS41_DSPARK_GAMMA=3
expect_profile 4 "1,2,3,4" 1 DS41_GRAPH_BUCKETS=exact
expect_profile 4 "1,2,3,4" 1 DS41_GRAPH_BUCKETS=exact DS41_DSPARK_GAMMA=1
expect_profile 2 "1,2" 1 DS41_GRAPH_BUCKETS=exact DS41_DSPARK_GAMMA=3
expect_profile 8 "1,2,4,8" unset DS41_MAX_RUNNING_REQUESTS=8
expect_profile 8 "1,2,3,4,5,6,7,8" 1 DS41_MAX_RUNNING_REQUESTS=8 DS41_GRAPH_BUCKETS=exact
expect_profile 3 "1,2,3" 1 DS41_MAX_RUNNING_REQUESTS=3 DS41_GRAPH_BUCKETS=exact
expect_profile 1 "1" unset DS41_MAX_RUNNING_REQUESTS=1
expect_profile 4 "1,2,4" unset SGLANG_DSV41_EXACT_GRAPH_BUCKETS=1   # the padded arm clears an inherited opt-in
expect_profile 4 "1,2,4" unset SGLANG_DSV41_DSPARK_ADAPTIVE_VERIFY=1
# Decoder SWA bounded replay: off unless asked, validated before any host access.
expect_profile 4 "1,2,4" unset DS41_DECODER_SWA_BOUNDED_REPLAY=0
expect_profile 4 "1,2,4" unset DS41_DECODER_SWA_BOUNDED_REPLAY=1
for settings in implicit explicit; do
  args=(DS41_DSPARK_POLICY=adaptive)
  [[ "$settings" == implicit ]] || args+=(DS41_DSPARK_GAMMA=3 DS41_MAX_RUNNING_REQUESTS=4)
  got=$(profile_for "${args[@]}") || fail 'adaptive profile failed'
  [[ "$got" == 'max_running_requests=4
graph_config={"decode":{"backend":"full","bs":[1,2],"max_bs":2}}
exact_graph_buckets=unset
dspark_policy=adaptive
dspark_gamma=3
adaptive_verify=1
chunked_prefill_size=1536
prefill_decode_interval=32
disable_overlap_schedule=1
fp4_logits_budget_mb=64
decoder_swa_bounded_replay=0' ]] || fail "unexpected adaptive profile: $got"
  [[ ! -s "$preflight_log" && ! -e "$served_log" ]] || fail 'adaptive print reached runtime'
done
reject_profile DS41_DSPARK_POLICY=invalid
reject_profile DS41_DSPARK_POLICY=adaptive DS41_DSPARK_GAMMA=0
reject_profile DS41_DSPARK_POLICY=adaptive DS41_DSPARK_GAMMA=1
reject_profile DS41_DSPARK_POLICY=adaptive DS41_GRAPH_BUCKETS=exact
reject_profile DS41_DSPARK_POLICY=adaptive DS41_MAX_RUNNING_REQUESTS=2
reject_profile DS41_DSPARK_POLICY=adaptive DS41_MAX_RUNNING_REQUESTS=8
reject_profile DS41_DSPARK_POLICY=adaptive SGLANG_RAGGED_VERIFY_MODE=compact
reject_profile 'DS41_DECODER_SWA_BOUNDED_REPLAY='
reject_profile DS41_DECODER_SWA_BOUNDED_REPLAY=2
reject_profile DS41_DECODER_SWA_BOUNDED_REPLAY=yes
reject_profile DS41_DECODER_SWA_BOUNDED_REPLAY=true
reject_profile DS41_GRAPH_BUCKETS=bogus
reject_profile DS41_GRAPH_BUCKETS=Exact
reject_profile DS41_MAX_RUNNING_REQUESTS=3                              # padded needs a power of two
reject_profile DS41_MAX_RUNNING_REQUESTS=0 DS41_GRAPH_BUCKETS=exact
reject_profile DS41_MAX_RUNNING_REQUESTS=9 DS41_GRAPH_BUCKETS=exact
reject_profile DS41_MAX_RUNNING_REQUESTS=x DS41_GRAPH_BUCKETS=exact
reject_profile 'DS41_CHUNKED_PREFILL_SIZE='
reject_profile DS41_CHUNKED_PREFILL_SIZE=0
reject_profile DS41_CHUNKED_PREFILL_SIZE=-1
reject_profile DS41_CHUNKED_PREFILL_SIZE=1.5
reject_profile DS41_CHUNKED_PREFILL_SIZE=bad
reject_profile 'DS41_PREFILL_DECODE_INTERVAL='
reject_profile DS41_PREFILL_DECODE_INTERVAL=0
reject_profile DS41_PREFILL_DECODE_INTERVAL=-1
reject_profile DS41_PREFILL_DECODE_INTERVAL=1.5
reject_profile DS41_PREFILL_DECODE_INTERVAL=bad
reject_profile DS41_DSPARK_GAMMA=3 DS41_MAX_RUNNING_REQUESTS=3 DS41_GRAPH_BUCKETS=exact   # width 4: at most two requests
reject_profile DS41_DSPARK_GAMMA=1 DS41_MAX_RUNNING_REQUESTS=5 DS41_GRAPH_BUCKETS=exact   # width 2: at most four requests

# The serve command line carries the runtime flag only through this array, which is
# empty unless the knob is 1. The real exec needs the host's RDMA device, so check it
# statically.
grep -Fq -- 'replay_args=(--enable-decoder-swa-bounded-replay)' "$launcher" ||
  fail "bounded replay knob does not select the runtime flag"
grep -Fq -- '"${replay_args[@]}"' "$launcher" || fail "bounded replay flag is not on the serve command line"

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
printf '%s\\n' "\$XDG_CACHE_HOME" "\$SGLANG_CACHE_DIR" "\$TRITON_CACHE_DIR" "\$SGLANG_JIT_CACHE_DIR" \\
  "\$TORCHINDUCTOR_CACHE_DIR" "\$AITER_JIT_DIR" "\$AITER_ROOT_DIR" > "$cache_log"
exit 73
EOF
chmod +x "$tools/mkdir"

check_cache_paths() {
  local expected_triton=$1 expected_native=$2 rc=0
  shift 2
  rm -f "$cache_log"
  run_launcher "$@" >"$harness/cache-output.log" 2>&1 || rc=$?
  [[ $rc == 73 ]] || fail "launcher did not reach the cache probe: exit $rc"
  printf '%s\n' "$cache/runtime/cache" "$cache/runtime/sglang" "$expected_triton" "$expected_native" \
    "$cache/runtime/torch" "$cache/runtime/aiter-jit" "$cache/runtime/aiter-root" \
    > "$harness/expected-cache.log"
  cmp -s "$harness/expected-cache.log" "$cache_log" || fail "unexpected cache paths"
  [[ ! -e "$served_log" ]] || fail "cache probe started serving"
}

check_cache_paths "$cache/runtime/triton" "$cache/runtime/sglang-jit"
check_cache_paths "$cache/runtime/triton" "$cache/runtime/sglang-jit" TRITON_CACHE_DIR= SGLANG_JIT_CACHE_DIR=
check_cache_paths "$tmp/shared triton" "$cache/runtime/sglang-jit" "TRITON_CACHE_DIR=$tmp/shared triton"
check_cache_paths "$cache/runtime/triton" "$tmp/shared native jit" "SGLANG_JIT_CACHE_DIR=$tmp/shared native jit"
check_cache_paths "$tmp/shared triton" "$tmp/shared native jit" \
  "TRITON_CACHE_DIR=$tmp/shared triton" "SGLANG_JIT_CACHE_DIR=$tmp/shared native jit"

rm -f "$cache_log"
rc=0
run_launcher TRITON_CACHE_DIR=relative/triton >"$harness/cache-output.log" 2>&1 || rc=$?
[[ $rc == 2 ]] || fail "relative Triton cache was not rejected: exit $rc"
grep -Fq 'TRITON_CACHE_DIR must be an absolute directory' "$harness/cache-output.log" ||
  fail "missing invalid cache path diagnostic"
[[ ! -e "$cache_log" && ! -e "$served_log" ]] ||
  fail "invalid cache path reached cache creation or serving"

rm -f "$cache_log"
rc=0
run_launcher SGLANG_JIT_CACHE_DIR=relative/native >"$harness/cache-output.log" 2>&1 || rc=$?
[[ $rc == 2 ]] || fail "relative native JIT cache was not rejected: exit $rc"
grep -Fq 'SGLANG_JIT_CACHE_DIR must be an absolute directory' "$harness/cache-output.log" ||
  fail "missing invalid native JIT cache path diagnostic"
[[ ! -e "$cache_log" && ! -e "$served_log" ]] ||
  fail "invalid native JIT cache path reached cache creation or serving"

echo "ds41-node launcher check passed"
