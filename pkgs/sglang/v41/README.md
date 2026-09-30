# DeepSeek V4.1 on gfx1151

`sglang-v41-rocm` pins the V4.1 serving source and matching native kernels to
`fdf14605e5791ad3bcb93b27a495d447f2746838`. Both Rust extensions build from that
source in a separate derivation, so Python-only patches reuse the native build.
The extensions use the runtime's exact Python and Torch and are linked into
the complete Nix package. The candidate preserves the official Flash checkpoint's packed MXFP4
experts and FP8/E8M0 dense weights; it adds no weight quantization.

The patches provide portable mHC, FP4 sparse indexing and E8M0 activation
quantization on gfx1151. File-backed Engram reads only requested native rows,
checks tensor metadata, and skips the large embedding tensors before the weight
loader materializes them. This allows the native TP4 weights to fit in the four
128 GB machines. Peak memory and available cache capacity depend on the workload.

After staging weights on Trex, publish a frozen, read-only SPDK model snapshot
and point each rank at that snapshot mounted over NVMe/RDMA. Trex's writable
`/models` and a client's pinned `/models` can contain different model generations.
Verify the namespace UUID and RDMA transport before measuring loading or Engram
lookup performance; a separate NFS export exercises a different storage path.

`lib/bench/ds41-node.sh` is a candidate foreground launcher for this native
TP4/EP1 configuration. Build `.#sglang-v41-rocm` once and make that **same Nix
output** visible on all four ranks: copy it to the writable node stores with
`nix copy --to ssh://HOST "$(readlink -f result)"`, or expose the identical
output through a configured read-only shared store. Keep the launcher and its
adjacent `ds41-model-check.py` together on each node. On Strix-3 (rank 0),
Strix-4 (rank 1), Strix-2 (rank 2), and Strix-1 (rank 3), run the matching rank:

```sh
DS41_BINARY=/nix/store/EXACT_SGLANG_OUTPUT/bin/sglang \
DS41_NODE_RANK=0 lib/bench/ds41-node.sh
```

`DS41_HEAD_ADDR` (default `192.168.25.103:51041`), `DS41_MODEL_PATH` (the
published snapshot), `DS41_PORT` (default `31041`), and `DS41_CACHE_ROOT` are
configurable. The script verifies the pinned revision, manifest, file sizes,
read-only XFS mount, snapshot UUID and live NVMe/RDMA paths before serving; it
does not mount or recover storage. It holds `/tmp/ds41-gpu.lock` while serving
to prevent another cooperating launcher from using the same GPU. The API binds
only to node-local localhost.
`TRITON_CACHE_DIR` may override the runtime-specific Triton cache with an
absolute path. Reuse a directory only after verifying matching GPU architecture,
compiler/toolchain and Python/Torch/Triton ABI identity; the launcher checks the
path, not that identity. Unset or empty values retain the runtime-specific
default, and all other caches remain runtime-specific.
For access from a client, tunnel to the head rank, for example
`ssh -N -L 31041:127.0.0.1:31041 strix-3`, then use
`http://127.0.0.1:31041` on that client. Configured 128K context and C1/C2
graphs require workload-specific validation before deployment.

To use packaged Pi or OpenCode through that tunnel, create session-local client
configuration; these commands leave your global configuration files alone.
They match the coding checks' provider settings (Pi 0.87.1, OpenCode 1.18.32).
The clients must be on `PATH`; this flake exposes Pi, but does not package
OpenCode.
Run the clients from the working tree you want them to inspect or edit.

```sh
DS41_CLIENT_CONFIG="$(mktemp -d)"
mkdir -p "$DS41_CLIENT_CONFIG/pi/sessions"
cat > "$DS41_CLIENT_CONFIG/pi/models.json" <<'JSON'
{
  "providers": {
    "strix-ds41": {
      "baseUrl": "http://127.0.0.1:31041/v1",
      "api": "openai-completions", "apiKey": "local",
      "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": true},
      "models": [{
        "id": "deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash",
        "reasoning": true, "input": ["text"],
        "contextWindow": 131072, "maxTokens": 16384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
JSON
cat > "$DS41_CLIENT_CONFIG/pi/settings.json" <<'JSON'
{"compaction": {"reserveTokens": 16384}}
JSON
cat > "$DS41_CLIENT_CONFIG/pi/sampling.ts" <<'JS'
export default function(pi) {
  pi.on('before_provider_request', event =>
    ({ ...event.payload, temperature: 1.0, top_p: 0.95 }));
}
JS
PI_CODING_AGENT_DIR="$DS41_CLIENT_CONFIG/pi" pi \
  --offline --provider strix-ds41 --model deepseek-v4.1-flash --thinking low \
  --no-extensions --extension "$DS41_CLIENT_CONFIG/pi/sampling.ts" \
  --no-skills --no-prompt-templates --no-context-files \
  --session-dir "$DS41_CLIENT_CONFIG/pi/sessions" \
  --print 'Inspect this repository and explain how to run its tests.'
```

For interactive Pi, omit `--print` and its prompt argument.
`--no-context-files` isolated the coding checks; omit it for normal repository
instructions. Pi's recipe retains the text-only model declaration used in its
coding check.
OpenCode's declaration also enables image attachments; add `--file image.png`
before `--` when needed. Its configured tool permissions allow file/shell tools
and deny external-directory and web tools; these permissions are not a process
or network sandbox:

```sh
DS41_CLIENT_CONFIG="$(mktemp -d)"
cat > "$DS41_CLIENT_CONFIG/opencode.json" <<'JSON'
{
  "$schema": "https://opencode.ai/config.json",
  "model": "strix-ds41/v41", "small_model": "strix-ds41/v41",
  "enabled_providers": ["strix-ds41"],
  "autoupdate": false, "share": "disabled", "lsp": false, "formatter": false,
  "permission": {
    "*": "allow", "external_directory": "deny", "webfetch": "deny", "websearch": "deny"
  },
  "provider": {
    "strix-ds41": {
      "npm": "@ai-sdk/openai-compatible", "name": "DeepSeek V4.1 Flash",
      "options": {"baseURL": "http://127.0.0.1:31041/v1", "apiKey": "local"},
      "models": {
        "v41": {
          "id": "deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash",
          "reasoning": true, "tool_call": true, "attachment": true,
          "modalities": {"input": ["text", "image"], "output": ["text"]},
          "limit": {"context": 131072, "output": 8192}
        }
      }
    }
  }
}
JSON
OPENCODE_CONFIG="$DS41_CLIENT_CONFIG/opencode.json" \
OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_CLAUDE_CODE=1 \
OPENCODE_DISABLE_AUTOUPDATE=1 OTEL_SDK_DISABLED=true \
  opencode run --pure --auto --variant low --model strix-ds41/v41 -- \
  'Inspect this repository and explain how to run its tests.'
```

`local` is a dummy client key for this unauthenticated loopback endpoint. Both
clients use Chat Completions and native low reasoning effort (50), which is not
a thinking-token or time limit. The 131,072-token context includes output;
Pi reserves 16,384 tokens for output, while OpenCode's output budget is 8,192.
These settings preserve the
configured capacity, not a guarantee for arbitrary 128K coding sessions; the
qualification and serial/paired numerical limitations below still apply.

Enable file-backed Engram with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and
`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file`. It requires the default safetensors
loader, mmap enabled, and checkpoint prefetch disabled. Prefill remains eager.
Decode graphs support exact capture sets `[1,2]` or `[1,2,4,8]` with TP4, DP1/CP1/PP1,
no speculation and one compute stream. Use `--cuda-graph-backend-decode full`,
`--cuda-graph-backend-prefill disabled`, `--max-running-requests 2`, and
`--cuda-graph-config '{"decode":{"backend":"full","bs":[1,2],"max_bs":2}}'`.
For capacity eight and all four capture buckets, use `--max-running-requests 8`
and `--cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8],"max_bs":8}}'`.
Other batch sizes remain eager; graph rows are never padded. Engram hashing
and bounded file reads happen before replay; captured code consumes refreshed
native row buffers. A failure after the history commit terminates the worker
so the serving controller can stop the TP group; that decode step cannot be
retried in place. Unsupported file-backed graph configurations are rejected.

Graph selection also distinguishes the indexer's sequence-length branches;
mixed-length batches select a branch that handles every request. Prefill and
uncaptured batch sizes retain their ordinary paths. Variant eligibility follows
the effective capture buckets, independently of request-pool capacity. Native
C1/C2/C4/C8 projections use the
existing activation quantizer and FP8/E8M0 weights, with FP32 reduction and
BF16 output. This applies only to marked V4.1 Engram, TP4 attention, and TP4/EP1
separate shared-expert projections. The FP32 mHC projection has its own exact
C1/C2 shape guard; RMS and Sinkhorn operations are unchanged. These
reductions can round differently from the generic matrix kernels. C4/C8 reuse
each native weight tile across the token rows with independent FP32 accumulators;
they preserve the existing C1/C2 kernels and quantizer.

Portable gfx1151 C2 mHC post-residual mixing uses one kernel for HC4/H5120,
with separate FP32 products/additions and final BF16 rounding. Explicit
FlashInfer/TileLang choices and all other shapes retain their existing paths.

`--model-loader-extra-config '{"enable_multithread_load":false}'` disables both
iterator threads and model-side weight-copy threads. With threading enabled,
the existing `num_threads` setting (default 8) also bounds outstanding copies,
so CPU dequantization cannot run arbitrarily far ahead of device copies.

Use `SGLANG_USE_AITER=0`, `SGLANG_HACK_FLASHMLA_BACKEND=triton`,
`SGLANG_DSV4_KV_LAYOUT=v4`, `SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp8`,
`--moe-runner-backend triton`, `--fp8-gemm-backend triton`,
`--disable-shared-experts-fusion`, and `--disable-custom-all-reduce`.
The V4.1 vision tower uses a model-specific
rank-3 SDPA path and computes RoPE tables on the GPU to preserve the official
BF16 rounding behavior.
Native input modalities are text and images. Audio needs a separate recognizer;
video and PDFs need conversion to frames, text, or page images. Output is text,
reasoning, and tool calls.
Disable both TileLang mHC overrides. The automatic HIP attention selection can
choose an unqualified TileLang path, so the explicit Triton setting matters.
For TP4's 16 attention heads per rank, single-scope SWA autotuning retains the
three existing 16-head tiles and skips larger masked-head tiles. This reduces
cold compilation without changing kernel arithmetic. `tests/swa.py` checks
native cache bytes and attention against independent references, including
changed-input graph replay through the SWA backend.

Run the focused checks through the built package's interpreter, passing the
package output as the first argument:

```sh
nix build .#sglang-v41-rocm
result/bin/sglang-python pkgs/sglang/v41/tests/engram.py "$(readlink -f result)"
```

For native TP4 C2/C4/C8 routed experts, an exact-shape gfx1151 kernel evaluates only
live gate/up rows and shares weight reads between routes to the same expert.
It applies to contiguous BF16 `[C, 5120]` with C in 2/4/8, six routes, 384 experts and local
intermediate size 576. C4/C8 reuse the unchanged C2 body within adjacent token
pairs; each pair owns separate input, route and output slices. Other shapes
retain the ordinary N128/N256 dispatch. Native
packed weights/scales, activation and down projection are unchanged. The FP32
reduction order differs, so bitwise equality to the fallback is not promised.
Filtered expert routes (`-1`) produce zero without reading weight memory.
Expanded subnormal weights follow the installed BF16 scaling path's flush-to-zero
behavior. Focused baseline controls check reserved scale bytes and subnormal
activations separately.
`tests/mxfp4.py` retains all 44 reference cases and checks the new reduction
against a native-operand FP64 reference and operation-derived error bound,
including cancellation, route changes, graph replay and other-shape fallbacks.
C4/C8 add shared, disjoint, duplicate and filtered routes, pairwise C2 projection
comparisons, and the same independent projection/downstream acceptance gates.
Its finite-fixture premises fail closed; it is not a full-model quality proof.
Run under an external 600-second timeout and 8 GiB host memory limit; the test
limits GPU allocation to 3% and uses about 1.8 GB of packed synthetic weights.

The other checks cover MXFP4 experts, mHC, FP4 indexing and FP8 quantization.
They use independent numerical or bit-level references and changing-input
graph replay where applicable. `ENGRAM_CPU_ONLY=1` runs the row-store checks
without a GPU. Fixtures are synthetic and small; no checkpoint or run logs are
included here.

`tests/fp8.py` checks all six guarded projection shapes with exact native-format
operands, the original quantizer, C1/C2/C4/C8 graph replay and generic fallbacks.
Dense exact-dyadic operands independently exercise every reduction term; C4/C8
also compare bitwise with separate native C2 calls. The CPU-only
`tests/graph-buckets.py` checks effective capture geometry, variant boundaries
and unpadded eligibility at package build time using explicit metadata stubs.
`tests/mhc.py` includes a FP64 projection reference with an FP32 reduction-error
bound, plus post-residual bitwise, signed-zero, reduction-order, graph and
fallback controls against installed Torch. `tests/engram.py` checks prepared-row
embedding, bounded graph input generations, and cross-stream bucket transitions
with owned output snapshots; its local ownership checks do not replace a multi-rank collective
or full-model test.

`tests/weight-loading.py` checks bounded source ownership, serial loading and
copy-error propagation using the installed loader source. Run it with
`HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''` for its CPU-only checks.

The portable MXFP4 method owns routed scaling; its model callers add the shared
output without applying the routed factor again. `tests/routed-scaling.py` runs
at package build time on CPU tensors, exercising the installed caller and quant
method with GPU math and communication stubs. It checks once-only scaling and
preserves legacy paths; it is not a full-model accuracy test.

The complete vision tower and image projection match the pinned official
reference bitwise on three image inputs. Run `tests/vision.py` with the runtime
path and checkpoint directory to repeat that comparison; it reads only the
vision shard. Full serving produced correct text/image smoke outputs and useful
Pi/OpenCode benchmark fixes that were independently reviewed and tested. These
bounded checks are not a general coding-quality guarantee.

After correcting routed scaling, a short executable-answer control still gave
two wrong answers out of eight with thinking disabled. With thinking enabled
at effort 50, all five cold, cached and paired requests passed their answer and
cache checks. These checks support using thinking for coding; they do not
establish full-model reference parity or correctness on arbitrary tasks.

With the launcher's 512-token prefill chunk, two related 130,563-token prompts
passed all ten retrieval-answer checks across cold loads, repeats, branching,
switching back and paired submissions. Repeats retained 130,560 cached tokens;
the branch reused 65,024 shared-prefix tokens. Both paired rounds retained the
full cached prompts, with actual two-request graph decode observed in one round.
This establishes capacity and cache behavior for that workload, not unrelated
concurrent cold 128K prompts or batched prefill.

Numerical differences between serial and paired runs remain unresolved, including
with the eager baseline. The eager head-boundary comparison changed both batching
and prefix-cache reuse. The 128K run failed strict logprob invariance between
serial and paired requests and between paired rounds, despite correct answers.
This reproducibility requirement is not an independently derived accuracy bound.
A full-depth text comparison against the official high-level model with CPU
mathematical adapters agrees on the top token for an eight-token prefill and
one teacher-forced decode, but the probability distributions differ substantially.
Activation and KV quantization, TP partial rounding, head dtype and backend
arithmetic differ between those paths; the cause has not been localized.
The package remains a serving candidate pending numerical qualification;
measured speed does not establish proximity to the hardware roofline.

The native top512 selector preserves strict FP32 score ordering and breaks exact
cutoff ties by smaller logical token ID, including compact candidate rows. When
a coarse score bin exceeds the native scratch capacity, it rescans the original
row rather than selecting from truncated scratch. `tests/topk.py` checks strict
winners, synthetic cutoff ties under candidate/page permutations, masked tails,
128K overflow and changed-input graph replay.

On gfx1151, level-one candidate blocks also preserve strict score order and
resolve equal-score cutoffs by smaller logical block ID, with the newest causal
block still included. `tests/candidate-blocks.py` checks this policy against an
independent reference, including partial blocks, causal masks, signed zero,
128K inputs and changing scores/lengths during graph replay. These component
checks do not qualify full-model 128K cache reuse.
