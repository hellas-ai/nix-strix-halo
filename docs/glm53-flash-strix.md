# GLM-5.3-Flash staging and four-Strix serving review

Reviewed on trex, 2026-09-28. Inference has not been launched. Runtime behavior
and performance below remain unmeasured on this cluster.

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
Only `_staging/verified.json` certifies that every file passed upstream SHA-256 / Git
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

Read-only SSH checks reached strix-1, strix-3 and strix-4. Each reports about
125 GiB usable system RAM and about 119 GiB available at inspection.
strix-2 refused TCP/22 on both `192.168.23.192` and `192.168.25.102`.

The fabric addresses are `192.168.25.101` through `.104` on `cx5fabric0`.
strix-1 reports a 100 Gb/s link and an active mlx5 Ethernet/RDMA port.
End-to-end RCCL/RoCE operation was not tested. Do not infer a working RDMA
collective from link state alone.

The nodes mount `/models` as a read-only XFS filesystem over trex's SPDK NVMe
export (`nqn.2026-07.link.satanic.trex:models`). strix-1 did not see the newly
created model directory. Before serving, use the existing storage publication
procedure to flush trex's writer and refresh the readers; verify identical
snapshot contents on every node. These reader mounts were not changed.

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

This is preparation for bring-up, not qualification of the full GLM execution
path. The Nix package still omits `sglang-kernel`: some operations have local
Torch/Triton/JIT fallbacks, while host KV transfers and other AOT-only paths
remain unavailable. Qualify the selected GLM kernels on gfx1151 before relying
on this launch template. If native AOT kernels are needed, package the matching
0.5.20 source with upstream's `docker/patches/sgl-kernel-gfx1151.sh` wave32 and
architecture fixes. The official HIP recipe pins transformers 5.12.1; nixpkgs
supplies 5.17.0, so model loading also needs an end-to-end check.

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

The following is an **untested bring-up template**, for the updated and
validated SGLang environment. It has not been executed. Run one rank on each
Strix host; rank values are 0, 1, 2, 3. It is not a trex server command.

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
  --model-path /models/GLM-5.3-Flash-AWQ-W4A16 \
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
loading and context. There is no measured tokens/s estimate yet.

## References

- [Official model](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Pinned quantized checkpoint](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16/tree/abd7b07719111f137e1de8a0c1b7e01c11b74d1a)
- [SGLang 0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)
- [gfx1151 recipe](https://github.com/sgl-project/sglang/blob/v0.5.20/docker/rocm-gfx1151.Dockerfile)
- [ROCm WNA16 dispatch](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py)
