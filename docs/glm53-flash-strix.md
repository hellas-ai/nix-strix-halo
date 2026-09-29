# GLM-5.3-Flash staging and four-Strix serving review

Updated on trex, 2026-09-29 (Europe/Zurich). The full checkpoint loads on all
four TP ranks and serves requests, but **real-model correctness failed**. The
first arithmetic request degenerated into repetition, and prefill/decode
log-probability consistency failed on both short and chunked prompts. Pi
acceptance, 128K context and performance tuning are gated on resolving this.
The reduced TP4 dummy fixture and isolated kernels pass within their stated
scope; those checks do not establish full-model correctness.

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
full-model quality and consistency must be retested before any coding task.

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
Four-rank RCCL over TCP passed nonzero-data correctness checks from 4 bytes
through 64 MiB. RoCE restricted to the HCA carrying `cx5fabric0` passed through
4 MiB but stalled at 64 MiB and was stopped after more than three minutes.
RCCL reported that GPU Direct RDMA was unavailable (`GDR 0`). Do not select
RoCE for serving until this larger-transfer failure is resolved. HCA numbers
must be derived from each host's netdev, not assumed to match across nodes.

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
is component qualification; the complete model, quantization quality and
agentic coding behavior have not yet passed acceptance.

The official HIP recipe pins transformers 5.12.1; nixpkgs supplies 5.17.0.
The staged model config and compressed-tensors metadata parse with this package;
all four ranks complete weight loading and initialization. The real-model
generation checks above fail and must be repaired before acceptance.

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
weight loading and startup have passed; real-model correctness has not.
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
export NCCL_IB_DISABLE=1  # establish a TCP baseline before qualifying RoCE
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
loading and context. There is no measured model tokens/s result yet. Each node uses the transient
user unit `glm53-sglang.service`. On trex, `glm53-tunnel.service` forwards
`127.0.0.1:30053` to rank zero's loopback API at port 30000. A listening tunnel
alone does not mean the model is ready; check `/health` after loading.

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
