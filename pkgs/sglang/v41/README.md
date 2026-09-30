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
For access from a client, tunnel to the head rank, for example
`ssh -N -L 31041:127.0.0.1:31041 strix-3`, then use
`http://127.0.0.1:31041` on that client. Configured 128K context and C1/C2
graphs require workload-specific validation before deployment.

Enable file-backed Engram with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and
`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file`. It requires the default safetensors
loader, mmap enabled, and checkpoint prefetch disabled. Prefill remains eager.
Decode graphs support exact one- and two-request batches with TP4, DP1/CP1/PP1,
no speculation and one compute stream. Use `--cuda-graph-backend-decode full`,
`--cuda-graph-backend-prefill disabled`, `--max-running-requests 2`, and
`--cuda-graph-config '{"decode":{"backend":"full","bs":[1,2],"max_bs":2}}'`.
Engram hashing
and bounded file reads happen before replay; captured code consumes refreshed
native row buffers. A failure after the history commit terminates the worker
so the serving controller can stop the TP group; that decode step cannot be
retried in place. Unsupported file-backed graph configurations are rejected.

Graph selection also distinguishes the indexer's sequence-length branches;
mixed-length batches select a branch that handles every request. Prefill and
larger batches retain their ordinary paths. Native C1/C2 projections use the
existing activation quantizer and FP8/E8M0 weights, with FP32 reduction and
BF16 output. This applies only to marked V4.1 Engram, TP4 attention, and TP4/EP1
separate shared-expert projections. The FP32 mHC projection has its own exact
C1/C2 shape guard; RMS, mixing and Sinkhorn operations are unchanged. These
reductions can round differently from the generic matrix kernels.

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

The other checks cover MXFP4 experts, mHC, FP4 indexing and FP8 quantization.
They use independent numerical or bit-level references and changing-input
graph replay where applicable. `ENGRAM_CPU_ONLY=1` runs the row-store checks
without a GPU. Fixtures are synthetic and small; no checkpoint or run logs are
included here.

`tests/fp8.py` checks all six guarded projection shapes with exact native-format
operands, the original quantizer, C1/C2 graph replay and generic fallbacks.
`tests/mhc.py` includes a FP64 projection reference with an FP32 reduction-error
bound. `tests/engram.py` checks prepared-row embedding and bounded graph input
generations; its local ownership checks do not replace a multi-rank collective
or full-model test.

`tests/weight-loading.py` checks bounded source ownership, serial loading and
copy-error propagation using the installed loader source. Run it with
`HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''` for its CPU-only checks.

The complete vision tower and image projection match the pinned official
reference bitwise on three image inputs. Run `tests/vision.py` with the runtime
path and checkpoint directory to repeat that comparison; it reads only the
vision shard. Full serving produced correct text/image smoke outputs and useful
Pi/OpenCode benchmark fixes that were independently reviewed and tested. These
bounded checks are not a general coding-quality guarantee.

With the launcher's 512-token prefill chunk, two related 130,563-token prompts
passed all ten retrieval-answer checks across cold loads, repeats, branching,
switching back and paired submissions. Repeats retained 130,560 cached tokens;
the branch reused 65,024 shared-prefix tokens. Both paired rounds retained the
full cached prompts, with actual two-request graph decode observed in one round.
This establishes capacity and cache behavior for that workload, not unrelated
concurrent cold 128K prompts or batched prefill.

Cross-batch numerical differences remain unresolved, including with the eager
baseline. The 128K run failed strict logprob invariance between serial and paired
requests and between paired rounds, despite correct answers. This reproducibility
requirement is not an independently derived accuracy bound. The package remains
a serving candidate pending a full-model reference comparison; measured speed
does not establish proximity to the hardware roofline.

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
