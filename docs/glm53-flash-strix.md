# GLM-5.3-Flash staging and four-Strix serving review

Updated on trex, 2026-09-29 (Europe/Zurich). The full checkpoint serves on
four TP ranks. Fixing duplicate MoE scaling restored coherent answers: the
arithmetic, streamed tool call and tool-result continuation checks now pass.
The FP32-router fix also passes component and API checks. Numerical
consistency still exceeds its probability limits. Stable routing made all 64
sampled token choices agree on TCP; the RoCE launch has one disagreement. Pi completed four
coding repairs: one after review feedback and three on the first draft,
including a task starting above 100K prompt tokens. Three-marker retrieval passed at
129,017 prompt tokens both uncached and with prefix reuse. The cached repeat
reused 128,960 tokens and took 4.45 seconds, versus 358 seconds cold.
Stable routing makes synthetic fresh-repeat, identical-prefix and appended-prefix
probabilities exact. All five cache-branch checks pass on the latest RoCE
launch, including truncation (maximum delta 0.0201); this is one tested workload.
General cache correctness, broader coding quality and performance relative
to the hardware roofline remain unqualified.

RoCE bring-up exposed two independent faults. The SGLang wrapper omitted
`libibverbs`, allowing RCCL to fall back silently to sockets. The package now
includes `rdma-core` in its library search path. The CRS812 also flooded
unicast fabric traffic despite showing learned host entries. During a single
transfer to strix-1, unrelated ports each transmitted 6.35 GB of RoCE traffic.
Persistent static forwarding entries for the four Strix MACs removed the flood;
the same raw RDMA transfer improved from 9.74 to 27.48 Gb/s. Trex's fabric MAC
is pinned as well. No switch firmware, PFC, port or VLAN settings were changed.
The standalone reconciliation script is
`machines/routeros/crs812/glm53-fabric-fdb.rsc` in the separate `nixos-config`
task worktree. Revalidate these mappings after recabling; current host pairs
share external ports despite the older per-host cable comments.

With explicit HCA selection, `NCCL_IB_TC=106` (DSCP 26, ECT(0)) and
`NCCL_NET=IB`, four-rank nonzero reductions pass from 4 bytes through 64 MiB.
The 64 MiB median improved from 277.8 ms before the forwarding change to
46.4 ms afterward; 8 KiB takes 0.074 ms in the standalone FP32 probe.
Repeated BF16 reductions at 8 KiB through 8 MiB agree bitwise across PyTorch,
SGLang PyNccl and graph execution. GPU Direct RDMA remains unavailable;
transfers use host buffers. The launcher defaults to RoCE and fails if IB
cannot initialize; `GLM_TRANSPORT=tcp` selects a deliberate TCP control.
Detailed results are in
[`glm53-roce-2026-09-29.json`](../lib/bench/results/glm53-roce-2026-09-29.json).
The full server reached readiness in 230 seconds. Arithmetic, streamed tool
calls and tool-result continuation pass. The same three-run short completion
benchmark measures 9.63 tokens/s and 0.355-second time to first token, versus
8.81 tokens/s and 0.374 seconds on TCP. Decode/prefill probability maxima
are 0.1821/0.2020 for the short/chunked cases, with one argmax disagreement;
the strict numerical gate remains failed. All five cache-branch cases pass.

At 22:05 UTC on September 28, all four Strix clients lost their NVMe/RDMA
storage connections while copying/checking local checkpoint caches, with no
model server running. Trex logged keep-alive timeouts and RDMA retry errors;
strix-1 logged `no usable path - requeuing I/O`. At 22:34 UTC, clearing PFC
for two seconds on the CRS812's interconnect to the CRS804 drained both stuck
lossless queues. The original PFC setting was restored. RDMA discovery then
passed and nodes 3/4 reconnected their storage. This establishes a PFC stall
as a cause of the storage outage; the precise trigger remains unproven.
TP4 remains the requested configuration. The canonical checkpoint and caches
on nodes 1, 3 and 4 have now passed full verification. Node 2 has recovered and its restaged private-volume cache has also passed
full verification.

The separate `nixos-config` recovery work adds an initrd hardware watchdog,
reboot actions for failed boot/storage startup, and a runtime direct-read
watchdog. Its signed boot images are published and cached on the router;
the runtime service is active on all four nodes. Fault-injection tests passed
for transient errors, persistent errors and a stopped probe process. A real
SIGSTOP test on node 2 triggered its 120-second watchdog and automatic reboot.
The new initrd activated its hardware watchdog at 7.9 seconds, and the node
returned to SSH on the published image. The old runtime hardware watchdog
alone could keep being fed while storage was unavailable.

TP4 relaunched at 23:07 UTC and initialized all four ranks. Weight loading
finished in 109 seconds on node 1 and 423–427 seconds on nodes 3/4. Node 2
was still loading directly from NFS when the 480-second post-load barrier
expired at 23:18 UTC. The remaining rank was stopped; no node rebooted.
TP4 restarted at 23:26 UTC with verified private-volume caches on all four
nodes. All four completed loading at 23:43 UTC; loading took 216–981 seconds.
Sequential prefetch helped the slow ranks complete. Each rank held roughly
42 GB of weights and allocated 0.39 GB for its initial 32768-token KV pool.
Initialization then failed with `NameError: flashinfer` in the DSA backend: its
`device_sm_major >= 10` test selected NVIDIA workspace allocation on gfx1151.
The patch requires `is_cuda()` for that allocation and the corresponding
TRT-LLM ragged-attention branch. The next launch uses serial eager safetensors
reads to avoid scattered mmap faults on network storage. The largest shard
is 20 GB; serial loading keeps its temporary memory bounded. A separate
patch makes the post-load barrier deadline configurable, preserving the
480-second upstream default; this launcher selects 1800 seconds.

The subsequent full launch reached HTTP startup at 23:56 UTC but warmup
failed: pooled DSA requires 64-token pages while ROCm forced single-token
pages. The reduced fixture reproduced the issue and a second incompatible
assertion in the cache allocator. The pooled indexer also called CUDA-only
DeepGEMM and rejected HIP at dispatch. The ROCm patch preserves the packed
64-token cache layout, pooled history/tail selection, and causal masks; it
uses Triton activation quantization and matrix products with FP32 accumulation.
FP8 operands are converted exactly to BF16, allowing the matrix products to
use RDNA instructions. `lib/bench/glm53-kpool-check.py` compares scores against
an independent FP64 CPU calculation: ten ragged/paged cases passed on every
node, with maximum absolute error 2.61e-7. This validates these component
scores, not full-model quality or long-context behavior.

The reduced fixture passed KDA and sparse prefill, then exposed further ROCm
gaps in dense clamping, query rotation and pooled selection. Dense clamping
now respects the AITER switch and reuses the existing Triton activation; its
CPU checks match exactly. Triton sparse attention now handles zero RoPE
dimensions and all 2051 history/tail columns. Eight CPU reference checks pass
for prefill/decode, zero/64 RoPE dimensions, short/2051-column index tables and
fully masked rows (relative L2 error 0.00176–0.00229 in BF16).

The query rotation reuses the existing normalized Triton Hadamard transform
used by pooled keys. Its BF16/FP32 checks pass for empty, contiguous and strided
inputs against a CPU Hadamard matrix. HIP pooled selection uses a deterministic
PyTorch baseline in place of the CUDA JIT kernel, preserving short histories,
ragged row starts, logical/page/offset mapping and compact live tails. Exact
CPU index checks cover boundary lengths, tied scores, remapped page rows,
padded output rows and empty history with a live tail. Sorting cost is not yet
optimized. The TP4 fixture completed warmup at 00:29 UTC and passed 7-token and
3073-token prompts, each producing eight tokens with finite log probabilities.
Component results are recorded in
`lib/bench/results/glm53-rocm-components-2026-09-29.json`.

The full checkpoint became ready at 00:42:49 UTC. Serial eager loading took
177 seconds on strix-1 and 499–526 seconds on the other ranks. The first real
request (`137 * 29`, greedy, low reasoning effort) emitted 1024 repetitive
reasoning tokens without a final answer. Decode throughput was approximately
4.4 tokens/s, including no graph or prefix-cache acceleration; this is not a
quality-qualified performance result. The consistency probe then found maximum
log-probability differences of 18.03 and 9.08 on 12-token and 2322-token prompts
(respective means 3.57 and 1.85), versus investigation limits 0.05/0.01.
The same generated token IDs were rescored by prefill with caches flushed.
A checkpoint-backed, four-layer diagnostic identified double routed-expert
scaling in the HIP Triton path: the runner applies 2.5 and the outer MoE
module applies it again. Its captured first-MoE output matches a 6.25 scale
(relative L2 0.00288); against the required 2.5 scale, error is 1.0045.
Patch 0012 skips the duplicate outer scaling when HIP uses the Triton runner.
With the patch, the same real-weight/input calculation agrees with an
independent CPU reference at relative L2 0.000839. This is a layer check;
subsequent full-model and coding results are recorded below.

`lib/bench/glm53-moe-scaling-check.py` exercises the complete MoE module,
including top-eight routing, INT4 dequantization, clamped activation, the
shared expert and scaling. The unpatched module fails at relative L2 1.449.
The corrected single-stream path passes 1/7/33-token cases on all four GPUs
(relative L2 0.000716–0.001831); multi-stream
scaling also passes when expert output is out-of-place. The upstream
multi-stream method reads its shared-expert input after the routed experts
can overwrite it in-place. The launcher explicitly disables ROCm multi-stream
while that independent aliasing issue remains unresolved. Captured first-layer
weights match the canonical checkpoint exactly; its KDA attention also agrees
with an independent CPU recurrence (relative L2 0.00493). Results are recorded
in `lib/bench/results/glm53-moe-scaling-2026-09-29.json`.


After correcting scaling, the full model returns `3973` for `137 * 29`, emits
a valid streamed `read_file` call and returns the exact supplied marker after
the tool response. The API probe now handles nullable `tool_calls` in streaming
deltas. The model generates sensible Fibonacci and parity code. Short/chunked
prefill/decode maximum log-probability deltas fall to 0.1154/0.1398, with means
0.0114/0.0130; these still exceed the initial 0.05/0.01 investigation limits.

Transformers' `Glm5NextTextTopkRouter` computes the router projection in FP32.
The checkpoint also declares `moe_router_dtype: float32`, but SGLang's HIP gate
ignored it and returned BF16 logits. Rounding the projection changed top-eight
expert sets in 21 of 840 captured token/layer pairs. Patch 0013 honors that
configuration through SGLang's existing FP32 gate path. Projection and complete
MoE CPU-reference tests pass on all four GPUs (relative L2 0.000679–0.001720);
the prior package fails the requested dtype assertion. Full-model results follow below. Component evidence is in
`lib/bench/results/glm53-router-fp32-2026-09-29.json`.

Repeated real TP4 captures subsequently isolated another router issue: identical
FP32 scores entered the grouped `torch.topk(sorted=False)` fallback, but its
expert order varied. This changes floating-point normalization and MoE summation;
fully tied scores also changed the selected expert set. The launcher now enables
`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`. Its fused Triton router agrees with
stable CPU selection (lowest expert ID wins ties), repeats exactly in eager and
graph execution, and preserves the single 2.5 routed-expert scale. The new
`glm53-router-repeat.py` regression and complete MoE checks pass on all four nodes.
With the first four real checkpoint layers at TP4, all 28 captured stage/chunk
outputs and fresh-request probabilities become bitwise identical. A truncated
cache branch still exceeds the probability limit. Full-model API checks pass;
fresh, identical and appended requests now reproduce exact probabilities, and
all tested branch token IDs agree. Decode/prefill maxima are 0.0618/0.1086 with
all 64 argmax choices agreeing; truncation still differs by 0.0978. The strict
probability gates remain failed. Evidence is in
[`glm53-fused-router-2026-09-29.json`](../lib/bench/results/glm53-fused-router-2026-09-29.json).

An independent `--enable-fp32-lm-head` experiment improves the output projection's
agreement with CPU double precision. It does not make the four-layer fixture
pass all cache or decode/prefill comparisons, so it remains disabled in the full
server. Results, including failures, are in
[`glm53-fp32-head-2026-09-29.json`](../lib/bench/results/glm53-fp32-head-2026-09-29.json).


On the FP32-router build, arithmetic and streamed tool use pass again. The
short/chunked consistency maxima are 0.0848/0.1598 and means 0.00647/0.01220;
all 64 sampled decode tokens are also the teacher-forced prefill argmax.
The strict numerical thresholds remain failed and visible. These results
justify an isolated coding evaluation, not a claim of full qualification.
The launcher enables cache-usage reporting so prefix reuse can be measured.

The package now supplies `sglang-python SCRIPT [ARGS...]`, sharing the serving
CLI's Python dependencies and ROCm/JIT environment. For example, on a Strix node:

```sh
SGLANG_USE_AITER=0 sglang-python lib/bench/glm53-moe-scaling-check.py \
  --model-path /mnt/glm53-fabric/GLM-5.3-Flash-AWQ-W4A16
```

The first isolated Pi task repaired the TheRock tarball updater's `10.1` versus
`10.10` series-boundary bug, added three tests and passed all 19 repository tests
plus five separate acceptance cases in 323 seconds. Review then found that the
candidate rejected `10.1.0rc2` for series `10.1.0`. The follow-up fixed that regression in 520 seconds. Independent review then
confirmed all 21 repository tests and nine separate acceptance cases pass;
the two-file repair is included in this branch. The two rounds used 59,716
input tokens and 2,453 output tokens, with no prefix reuse. This is one useful
review-and-repair result, not near-perfect first-pass correctness. Evidence is
in `lib/bench/results/glm53-pi-acceptance-2026-09-29.json`.

At 02:07 UTC, the PyTorch/ROCm profiler segfaulted while finalizing the
trace on strix-2, stopping TP4; the nodes and storage stayed up. Three ranks
saved CPU traces, but no device-kernel timing was captured, so those traces
cannot establish GPU utilization or roofline efficiency. The CPU trace shows
substantial overhead in repeated mHC reductions. Patch 0015 reuses the existing
Triton Sinkhorn kernel for the supported ROCm fallback parameters, retaining
FP32 projection and the general fallback for other parameterizations. Six
independent CPU-double cases and two fallback cases pass on every GPU.

Patch 0014 packs head-major ROCm query tensors before sparse attention. The
absorbed BMM's head stride previously included the prompt length, causing
Triton to compile another kernel for each unseen final prefill chunk. In an
isolated test, the old path took 2.75 seconds at its next unseen length; the
packed path took 0.012 seconds, with BF16 relative L2 around 0.00194 against
CPU double precision. Twelve sparse prefill/decode cases now pass on every
GPU, including head-major zero/64-RoPE layouts. These tests do not yet establish
an end-to-end speedup. Evidence is in
`lib/bench/results/glm53-layout-mhc-2026-09-29.json`. TP4 relaunched at 02:17 UTC
with a 131072-token limit, radix caching and graphs disabled, for regression
and long-context qualification.

With patches 0014/0015, API checks pass and all 64 sampled token choices still
agree between prefill and decode. The numerical maxima are 0.0588/0.2147 and
means 0.00691/0.01421, so the existing strict thresholds remain failed. At
131072 context capacity, a 129017-token prompt returned all three exact markers
at 10%, 50% and 90% depth. Two uncached runs returned identical answers in
478.46 and 400.06 seconds; the first includes compilation overhead. Three
128-output-token streaming runs measured median steady decode 6.60 tokens/s
and median first-token latency 0.498 seconds on a 12-token coding prompt.
Graphs and radix caching were disabled. This establishes a baseline, not the
requested roofline efficiency or coding quality at 128K. Evidence is in
`lib/bench/results/glm53-context-performance-2026-09-29.json`.

Patch 0016 fuses the four-stream mHC post mix, avoiding the intermediate
`[tokens, 4, 4, hidden]` tensor. Hidden columns are adjacent in workgroup order
to preserve memory locality. BF16/FP32 tests include empty inputs, hidden-size
tails and strided tensors; twelve independent CPU-double cases and three
fallback cases pass on every GPU. The initial 1024-token prototype improved
4.54 ms to 0.33 ms. Packaged per-node measurements are recorded in
`lib/bench/results/glm53-mhc-post-2026-09-29.json`. Full-model API and retrieval
checks pass with this patch; strict numerical comparisons still fail, as
reported above. A separate per-rank checkpoint reload experiment preserved generated
IDs but differed in probabilities (maximum 0.0429 on the four-layer slice),
so serving continues to use the standard loader pending investigation.

Hydra also exposed a Tensile wheel metadata mismatch after the nixpkgs update.
The distribution now uses the packaged ROCm release version, retaining the
independent upstream algorithm/CMake version. A full local Tensile build passes
including the metadata check; final-head Hydra checks remain required.

## Snapshot

- Repository: https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16
- Revision: `abd7b07719111f137e1de8a0c1b7e01c11b74d1a`
- Destination: `/models/GLM-5.3-Flash-AWQ-W4A16`
- Total: **190,843,146,350 bytes** (190.84 GB / 177.74 GiB), 24 files,
  including nine main weight shards and four NextN/MTP shards.
- Model: Z.ai GLM-5.3-Flash, `Glm5NextForConditionalGeneration` / `glm5_next`.
  45 layers, 288 routed experts, eight selected per token, 64 attention heads;
  hybrid KDA linear attention / DSA sparse attention and mHC connections.
- AWQ symmetric INT4 group size 128, compressed-tensors `pack-quantized`, no
  activation ordering. Routed experts are INT4; attention, shared experts,
  routers, vision, dense layers and MTP remain BF16. The quantizer derived it
  from the official FP8 checkpoint. Its card does not establish quality or
  speed on Strix Halo.

On trex, `_staging/manifest.json` records the upstream file sizes and digests.
`_staging/status.json` reports downloading, verifying, complete, or a failure.
Verification completed at **2026-09-28 21:14:36 UTC**: all 24 files and
111,346 indexed tensors passed. `_staging/verified.json` certifies that every file passed upstream SHA-256 / Git
blob digest checks, exact size checks, safetensors payload validation and
resolution of every tensor-to-shard index reference. Auxiliary tensors outside
the main index are counted separately. File presence or `du` alone
does not establish completion; downloads can contain partial/preallocated files.

The downloader is `stage-glm53-flash.service` in grw's systemd user manager:

```bash
systemctl --user status stage-glm53-flash.service
journalctl --user -u stage-glm53-flash.service -f
cat /models/GLM-5.3-Flash-AWQ-W4A16/_staging/status.json
```

It downloads only this revision, retries failed transfers, and runs
`verify.py` after the download. It contains no inference launch command.

## Cluster observations

Before the storage outage, all four nodes reported one gfx1151 Radeon 8060S GPU, about
124 GiB system RAM and 117–119 GiB available before testing. All four nodes are
now reachable. Torch sees approximately 124 GiB of GPU-addressable shared memory.

After its first recovery, strix-1 reported 85 W STAPM, 120 W fast PPT, 85 W
slow PPT and 70 W APU limits; its `ryzenadj` service was masked. The optional
minus-10 curve-optimizer service was inactive on every node. Nodes 2–4 logged
successful application of neutral CO at boot. No surviving previous-boot
journal or pstore record established the cause of strix-1's earlier reset.
These settings were left unchanged. Record power settings again before timing
inference; prior component timings must not be treated as current-power results.

The fabric addresses are `192.168.25.101` through `.104` on `cx5fabric0`.
Four-rank RCCL over TCP and the corrected RoCE deployment pass nonzero-data
correctness checks from 4 bytes through 64 MiB. Earlier unmarked RoCE tests
stalled at large sizes while the switch flooded unicast traffic; see the
forwarding and runtime fixes above. RCCL reports that GPU Direct RDMA is
unavailable (`GDR 0`). The launcher derives the HCA from the fabric netdev.

The nodes' `/models` mounts are older read-only SPDK snapshots. Publishing a
replacement requires a coordinated storage rollout and also affects other
clients. For bring-up, trex exports its current model filesystem read-only to
the four fabric IPs via `/etc/exports.d/glm53.exports`; each node mounts
`192.168.25.8:/strix-models` at `/mnt/glm53-fabric`, explicitly using
`ro,vers=4.1,proto=tcp,nconnect=4,nosharecache`. This gives each node a fabric
session (`addr=192.168.25.8`, client addresses `.101`–`.104`); a 1 GiB direct
read on strix-1 measured 1.3 GB/s. The initial NFS 4.2 mount at
`/mnt/glm53-models` reused the existing home-directory session on the slower
LAN and was abandoned for model loading. The temporary exports and mounts
need declarative integration after the runtime is qualified.

The attempted per-node cache is
`/tmp/glm53-models/GLM-5.3-Flash-AWQ-W4A16`. Here `/tmp` is on each host's
private NVMe/RDMA volume, not a local physical SSD, and is reformatted on every
boot. After recovery, nodes 3 and 4 completed full verification. Node 2's previous copy
was discarded during the watchdog reboot test; its replacement passed full
verification at 23:26 UTC. Strix-1 now
uses NFS for the system store, with its former private volume attached over
NVMe/TCP at `/mnt/glm53-cache`; its verified checkpoint is under
`/mnt/glm53-cache/tmp/glm53-models/GLM-5.3-Flash-AWQ-W4A16`. Its `/tmp` is now
RAM-backed, so do not copy another checkpoint there. A copied verification
marker alone is insufficient: run `_staging/verify.py` on each new copy.

On nodes 2–4, the fabric NIC negotiated PCIe 3.0 x4. Its encoding ceiling is
about 3.94 GB/s per direction before transaction overhead, even though the
Ethernet link reports 100 Gb/s. Use the negotiated PCIe link and measured
collectives when constructing the communication roofline.

Nominal snapshot size divided by four is 44.4 GiB per rank, or about 41 GiB
for the main shards without MTP. Actual resident memory also includes replicated
weights, packing/transposition buffers, vision, KV/linear-attention state and
workspace. This gives much more headroom than the official 328.37 GB FP8
snapshot. Confirm actual allocation during a later launch; do not use the
division as a measured memory figure.

## Serving recommendation

Start with **SGLang 0.5.20**, TP=4, one GPU per node, over the 100 GbE fabric.
This branch updates `sglang-rocm` from 0.5.14, which has no `glm5_next`
implementation, to 0.5.20.

The 0.5.20 source includes GLM-5.3-Flash, a ROCm
`CompressedTensorsWNA16TritonMoE` dispatch matching the staged checkpoint,
Triton DSA prefill/decode, and Triton KDA. This is source-level compatibility
evidence, not a successful gfx1151 model test.

The package uses the flake's ROCm 10 / torch 2.13 stack and current nixpkgs,
including compressed-tensors 0.18.0. ModelScope downloads are omitted because
nixpkgs marks that optional dependency insecure; local paths and Hugging Face
remain available. Its local ROCm patches are rebased onto
0.5.20's new kernel and argument-resolution module paths. Import checks cover
the model configuration and server arguments without loading model weights.

The gfx1151 package now builds native `sglang-kernel` 0.4.7 from the same
SGLang v0.5.20 source. It applies upstream's
`docker/patches/sgl-kernel-gfx1151.sh`, including the host/device wave32 fix,
and links against TheRock's libtorch. A local patch sends HIP clamped SwiGLU
through the existing Triton activation path; the upstream unfiltered path
asserts because it selects a CUDA/XPU-only kernel. TheRock invokes Clang directly, so the
build translates `--amdgpu-target` to `--offload-arch`. Other unsupported GPU
architectures retain the existing optional-kernel fallbacks.

The full GLM implementation imports on strix-1. Native softmax and sigmoid
routing (288 experts, top-8, batches 1/17/1024) agree with CPU references on all
four GPUs. BF16 matrix multiplication also passes its reference check. This
is component qualification. Full-model and coding results, with remaining
quality limitations, are recorded at the beginning of this document.

The official HIP recipe pins transformers 5.12.1; nixpkgs supplies 5.17.0.
The staged model config and compressed-tensors metadata parse with this package;
all four ranks complete weight loading and initialization. The corrected
runtime passes the API checks above; strict numerical consistency is still
under investigation.

The package build, GLM config/server-argument/native cache imports, and
`sglang serve --help` passed on trex. CLI help and quantization-config parsing
also emit a PyTorch `_clear_torch_ops_cache` cleanup traceback after AITER is
imported (`ValueError: too many values to unpack`), while exiting successfully.
This remains an upstream compatibility issue to investigate during bring-up.

Upstream also provides `docker/rocm-gfx1151.Dockerfile` as an alternative
bring-up environment. Its nightly workflow publishes dated
`rocm/sgl-dev:v<VERSION>-rocm724-gfx1151-<YYYYMMDD>` images. Resolve an existing
tag to a digest before use. This recipe uses ROCm 7.2.4 / torch 2.9.1 and older
compressed-tensors; the newer staged quantization metadata still needs a
loader check. The generic MI300/MI350 image is not a gfx1151 qualification.

Keep AITER acceleration disabled initially. Avoid CDNA-only MXFP4 kernels,
NVIDIA Marlin and FP8 attention defaults. mHC defaults to TileLang in 0.5.20;
explicitly select its torch fallback for the initial run. Disable speculative
decoding, graph capture and radix caching initially, then enable and measure
one optimization at a time after coherent generation is established.

The initial server uses the following conservative configuration. The
[`node launcher`](../lib/bench/glm53-node.sh) checks the verified snapshot and
starts one rank per Strix host; rank values are 0, 1, 2, 3. GPU imports and
weight loading and startup have passed; full qualification remains incomplete.
`GLM_NNODES=2` also selects TP2 (ranks 0/1); set `GLM_DIST_ADDR` to that pair's
rank-zero host. TP2 remains unqualified and needs measured memory headroom.

```bash
export HIP_VISIBLE_DEVICES=0
export SGLANG_USE_AITER=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export NCCL_SOCKET_IFNAME=cx5fabric0
export GLOO_SOCKET_IFNAME=cx5fabric0
export NCCL_IB_DISABLE=1  # TCP control; the node launcher defaults to qualified RoCE
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=1800

# NODE_RANK must be set separately on each node.
nix run .#sglang-rocm -- serve \
  --model-path /mnt/glm53-fabric/GLM-5.3-Flash-AWQ-W4A16 \
  --served-model-name glm-5.3-flash \
  --tp-size 4 --nnodes 4 --node-rank "$NODE_RANK" \
  --dist-init-addr 192.168.25.101:50000 --dist-timeout 1800 \
  --weight-loader-disable-mmap \
  --model-loader-extra-config '{"enable_multithread_load":false}' \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-backend dsa \
  --dsa-prefill-backend triton --dsa-decode-backend triton \
  --linear-attn-backend triton --moe-runner-backend triton \
  --disable-shared-experts-fusion --disable-custom-all-reduce \
  --disable-cuda-graph --disable-radix-cache \
  --context-length 32768 --chunked-prefill-size 1024 \
  --max-running-requests 1 --mem-fraction-static 0.75 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --default-chat-template-kwargs '{"clear_thinking":true,"reasoning_effort":"low"}' \
  --host 127.0.0.1 --port 30000
```

Let the model config select `compressed-tensors`; do not override it with
`--quantization awq`, which names a different checkpoint layout. Start with
text requests. Qualify the vision path separately. For interactive chat use
the official template's `clear_thinking=true` option; begin with
`reasoning_effort=low` when latency is the priority.

After correctness checks, benchmark RCCL/RoCE against TCP on the same fabric,
then compare TP4 with TP2 (or two TP2 replicas for aggregate throughput).
Fewer nodes may reduce communication overhead; TP2 leaves less memory for
loading and context. The corrected uncached, eager baseline has a measured median of 6.60 tokens/s
on short coding requests; it is not roofline-qualified. Each node uses the transient
user unit `glm53-sglang.service`. On trex, `glm53-tunnel.service` forwards
`127.0.0.1:30053` to rank zero's loopback API at port 30000. A listening tunnel
alone does not mean the model is ready; check `/health` after loading.

## Pi client

With the cluster and trex tunnel healthy, run from your coding checkout:

```bash
nix run /mnt/Home/src/nix-strix-halo-glm53#glm53-pi
```

The client selects `glm-5.3-flash` at `http://127.0.0.1:30053/v1`, low reasoning
effort, a 131072-token context limit and an 8192-token output limit. Its sessions
and editable `models.json` live in `$XDG_STATE_HOME/glm53-pi` (default
`~/.local/state/glm53-pi`); `GLM_PI_DIR` selects another directory. The command
uses Pi's offline startup mode and retains its normal coding tools. The
isolated acceptance runner additionally restricts filesystem access and disables
extensions/context files. Client startup, model discovery and a coding repair
through this packaged command pass. Pi fixed wheel URLs containing query
strings or hash fragments in 150 seconds, with 23 repository tests and eight
held-out cases passing on its first draft. Across seven assistant turns it
used 4,981 uncached input tokens, 22,080 cached tokens and 1,155 output tokens.
The reviewed patch is included in this branch. Evidence is in
[`glm53-pi-wheel-links-2026-09-29.json`](../lib/bench/results/glm53-pi-wheel-links-2026-09-29.json).

A third task supplied 99,998 tokens of repository source and instructions
before asking Pi to repair ROCm series matching in the wheel updater. The
first API prompt contained 101,440 tokens including Pi's instructions/tools;
later prompts reached 108,454. Pi completed twelve turns in 593 seconds,
using 106,167 uncached input tokens, 1,156,224 cached tokens and 2,480 output
tokens. The first draft passes all 28 repository tests and eight independent
held-out cases; the reviewed patch is included. This establishes one useful
coding run with a large context and repeated tool use, rather than general
coding quality or full-window acceptance. See
[`glm53-pi-long-context-2026-09-29.json`](../lib/bench/results/glm53-pi-long-context-2026-09-29.json).

A fourth task fixed the benchmark client treating streamed server errors and
EOF without `[DONE]` as successful responses. Pi completed twelve turns in
583 seconds; its first submitted patch passes 13 local tests and 14 independent
tests using a real HTTP fixture (the old client failed eight). The reviewed
implementation and tests are included. Failed streams no longer contribute
to reported throughput. Evidence is in
[`glm53-pi-stream-failure-2026-09-29.json`](../lib/bench/results/glm53-pi-stream-failure-2026-09-29.json).

## Qualification target and initial measurements

The user-selected acceptance workload is useful coding in isolated worktrees
of this flake, using Pi or another compatible client. Judge tasks by actual
patches, builds and tests, including tool-call parsing and streaming. Qualify
**128K tokens** after basic generation correctness, then compare cold and warm
prefixes, cache-on/cache-off output, and multi-turn tool use. Do not equate a
successful import or a coherent one-line answer with this acceptance target.

The prepared harnesses are `glm53-api-check.py` (math, streaming tool calls
and tool continuation), `glm53-consistency.py` (decode versus teacher-forced
prefill logprobs), and `glm53-context-cache.py` (measured context length,
three-depth retrieval and cold/warm output with cache-hit assertions). The
context harness defaults to 131072 tokens including a 2048-token output
reserve. Run it with `--expect-cache off` before enabling radix caching, and
with `--expect-cache on` afterward. These are investigation gates, not a
claim that all coding tasks or quantization quality have passed.

`glm53-cache-branches.py` adds a fresh/fresh control and repeated, appended,
truncated and divergent synthetic prefixes. Before stable routing, on the
graph-enabled runtime,
the fresh control's maximum log-probability difference was 0.212; appended
and truncated branches differed by 0.313 and 0.847. The latter also changed
generated token IDs. Identical and divergent cases passed the 0.05 limit.
Fresh/fresh variation means these results cannot be attributed solely to
cache state restoration. All comparisons are retained, including failures.
An isolated BF16 TCP all-reduce probe found identical repeated results and
agreement between PyTorch, PyNccl and graph execution at 8 KiB through 8 MiB.
It therefore did not reproduce the full-model variation; it does not rule
out other shapes or batch-size-dependent reduction order.

In the first-four-layer cache fixture, a 3073-token prefix truncated to 2119
tokens still differs by 0.0627 in output log probability. Captures of the
matched final 71-token chunk show bitwise-identical KDA initial/final states,
attention outputs, DSA selected KV values and full DSA layer output. Physical
cache page addresses differ as expected. This rules out a state-restoration
difference at those captured stages for this case, while the full output
probability gate remains failed. See
[`glm53-cache-state-2026-09-29.json`](../lib/bench/results/glm53-cache-state-2026-09-29.json).

The latest runtime uses context 131072, `--max-mamba-cache-size 128`, radix
caching and decode graphs restricted to batch size one. Pinning 512 Mamba
states exhausted the remaining static memory budget during initialization;
128 states fits. Three 128-token short-request measurements give a median
8.63 tokens/s and 0.386-second time to first token. This includes the fused
mHC post operation and graphs, so it does not isolate their individual gains
from the earlier 6.60 tokens/s baseline. Evidence is in
[`glm53-prefix-graph-2026-09-29.json`](../lib/bench/results/glm53-prefix-graph-2026-09-29.json).
After enabling the stable fused router alone, the same three-run short-request
measurement gives 8.81 tokens/s and 0.374-second time to first token. These
measurements do not establish the hardware-roofline target; see
[`glm53-fused-router-2026-09-29.json`](../lib/bench/results/glm53-fused-router-2026-09-29.json).

`lib/bench/glm53-hardware.py` measures components without model weights. The
initial results are in
[`glm53-hardware-2026-09-28.json`](../lib/bench/results/glm53-hardware-2026-09-28.json).
All four GPUs delivered roughly 228–231 GB/s for a three-array streaming add
(3 GiB transferred per iteration), and roughly 33–37 BF16 TFLOP/s for a
4096-square GEMM. TCP four-rank all-reduce on strix-1 measured about 0.29 ms
for 64 KiB, 3.76 ms for 4 MiB and 75.74 ms for 64 MiB. These are initial
component baselines, not model throughput or proof of a roofline target.

For the model, derive separate decode and prefill limits from bytes read,
actual selected experts, math, attention/state traffic and measured collective
cost. The target is approximately 80% of the applicable hardware roofline;
report the gap and its causes if that target is not attainable. Numerical
agreement and executable coding acceptance take priority over optimization.

## References

- [Official model](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Pinned quantized checkpoint](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16/tree/abd7b07719111f137e1de8a0c1b7e01c11b74d1a)
- [SGLang 0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)
- [gfx1151 recipe](https://github.com/sgl-project/sglang/blob/v0.5.20/docker/rocm-gfx1151.Dockerfile)
- [ROCm WNA16 dispatch](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py)
