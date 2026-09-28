# GLM-5.3-Flash staging and four-Strix serving review

Updated on trex, 2026-09-28. GPU kernel and collective probes have run on all
four nodes. TP4 reached weight loading, but strix-1 reset at approximately
21:44 UTC before startup completed. The remaining ranks were stopped. No
successful generation or coding-agent acceptance is claimed. Local checkpoint
caches and host recovery are in progress.

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

All four nodes are reachable and report one gfx1151 Radeon 8060S GPU, about
124 GiB system RAM and 117–119 GiB available before testing. strix-2 is now
online. Torch sees approximately 124 GiB of GPU-addressable shared memory.

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
weight loading still needs an end-to-end check.

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
weight-loader selection have passed; complete server startup is still pending.
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

# NODE_RANK must be set separately on each node.
nix run .#sglang-rocm -- serve \
  --model-path /mnt/glm53-fabric/GLM-5.3-Flash-AWQ-W4A16 \
  --served-model-name glm-5.3-flash \
  --tp-size 4 --nnodes 4 --node-rank "$NODE_RANK" \
  --dist-init-addr 192.168.25.101:50000 \
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
