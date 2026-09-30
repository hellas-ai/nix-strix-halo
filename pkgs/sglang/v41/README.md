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
128 GB machines. Runtime memory and loader peaks still need measurement.

After staging weights on Trex, publish a frozen, read-only SPDK model snapshot
and point each rank at that snapshot mounted over NVMe/RDMA. Trex's writable
`/models` and a client's pinned `/models` can contain different model generations.
Verify the namespace UUID and RDMA transport before measuring loading or Engram
lookup performance; a separate NFS export exercises a different storage path.

Enable file-backed Engram with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and
`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file`. It requires the default safetensors
loader, mmap enabled, and checkpoint prefetch disabled. Both prefill and decode
must run eagerly: graph capture rejects this synchronous lookup path rather
than replaying stale embeddings.

`--model-loader-extra-config '{"enable_multithread_load":false}'` disables both
iterator threads and model-side weight-copy threads. With threading enabled,
the existing `num_threads` setting (default 8) also bounds outstanding copies,
so CPU dequantization cannot run arbitrarily far ahead of device copies.

Use `SGLANG_USE_AITER=0`, `SGLANG_HACK_FLASHMLA_BACKEND=triton`,
`SGLANG_DSV4_KV_LAYOUT=v4`, `SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp8`,
`--moe-runner-backend triton`, `--fp8-gemm-backend triton`,
`--disable-custom-all-reduce`. The V4.1 vision tower uses a model-specific
rank-3 SDPA path and computes RoPE tables on the GPU to preserve the official
BF16 rounding behavior.
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

`tests/weight-loading.py` checks bounded source ownership, serial loading and
copy-error propagation using the installed loader source. Run it with
`HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''` for its CPU-only checks.

Component qualification is in progress. Full-model text/image correctness,
TP4 performance, 128K cache reuse and useful Pi/OpenCode acceptance remain
unqualified. The complete vision tower and image projection match the pinned
official reference bitwise on three image inputs. Run `tests/vision.py` with
the runtime path and checkpoint directory to repeat that comparison; it reads
only the vision shard. This package is a candidate for full serving qualification.

The native top512 selector preserves strict FP32 score ordering and breaks exact
cutoff ties by smaller logical token ID, including compact candidate rows. When
a coarse score bin exceeds the native scratch capacity, it rescans the original
row rather than selecting from truncated scratch. `tests/topk.py` checks strict
winners, synthetic cutoff ties under candidate/page permutations, masked tails,
128K overflow and changed-input graph replay. Level-one candidate-block cutoff
ties retain their upstream unspecified policy.
