# Staged SGLang remote expert-parallel wrapper

This directory is source-shaped staging material, not an applied patch. Copy
`python/sglang/srt/layers/moe/remote_ep_wrapper.py` into the matching upstream
SGLang tree when the integration task is ready.

The class follows the upstream `KTEPWrapperMethod` pattern but replaces the
CPU KTransformers executor with the transport client from
`pkgs/heterogeneous-moe/expert_server`. Its placement map assigns every global
expert either to the local dense GPU (`None`) or to an integer remote node ID.
Thus hot or latency-sensitive experts may remain local without relying on the
KTransformers convention that all local experts form a low-ID prefix.
M1 servers still store contiguous shards: if a retained hot expert falls inside
a server's range, its weight may be loaded redundantly on that server but no
route for it is sent there.

## Intended upstream attach point

In upstream SGLang, update:

`python/sglang/srt/layers/moe/fused_moe_triton/layer.py:~318`

That is the `FusedMoE.__init__` selection site where `KTEPWrapperMethod` is
currently installed around the ordinary GPU quantization method. Add remote-EP
configuration parsing beside `create_kt_config_from_server_args`, then wrap the
selected local method with `RemoteEPWrapperMethod` when remote EP is enabled.

With SGLang `--enable-dp-attention`, attention, routing, shared-expert work, and
KV cache ownership stay data-parallel on the fast GPUs. Each dense replica's
appropriate submission rank invokes the foreign remote executor. The expert
servers are endpoints attached to those ranks; they are not additional SGLang
EP ranks. The exact TP/DP submission and reduction ownership must be validated
against the chosen SGLang topology before enabling this scaffold.

## Required integration work

- Add server arguments for the placement map, endpoint list, timeouts, dtype,
  and enable flag, plus validation against the model's expert count.
- Add an upstream weight-loader hook that streams each remote checkpoint shard
  into `RemoteWeightShard` without first retaining it on every dense GPU.
- Remap arbitrary global local-expert IDs to the physical local slots created
  by the wrapped SGLang method, including scale tensors and EPLB metadata.
- Share transport clients across layers and implement connection lifecycle,
  cancellation, failure handling, backpressure, and CUDA graph exclusions.
- Replace Python route packing and the TCP fallback with an asynchronous data
  plane that overlaps remote work with local expert GEMMs.
- Define how expert placement interacts with EPLB, elastic EP, quantization,
  tensor parallelism, speculative decoding, and batch overlap.
- Add the scaffold as a version-pinned patch in `pkgs/sglang/patches/` and wire
  the expert-server package in the separate flake/overlay integration task.
