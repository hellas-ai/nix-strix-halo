# Remote expert-parallel SGLang graft (M2)

This is an out-of-tree graft for the repository's packaged SGLang 0.5.14. It
does not modify an upstream checkout in place. The design follows SGLang's
`KTEPWrapperMethod` seam, but the foreign executor is an M1 expert-server
client and placement is explicit per expert rather than the KTransformers
`expert_id >= num_gpu_experts` threshold.

The lifecycle is:

1. `submit_forward(x, topk_ids, topk_weights, stream)` asynchronously partitions
   the untouched global routes by expert-server node and starts scatter/remote
   execution.
2. The wrapped SGLang MoE method computes any hot experts placed on the local
   V620. Its IDs are masked and remapped to compact local weight slots.
3. `sync_forward(x, stream)` gathers the four already route-weighted remote
   partials, sums them, and the wrapper adds that result to the hot-GPU result.

Each `RemoteEPWrapperMethod` is layer-specific. Its client includes the
configuration's `layer_idx` in the transport envelope used by M1, while the
payload remains `(hidden_states, topk_ids, topk_weights)`.

## Files and exact attachment point

- `remote_ep_wrapper.py` becomes
  `python/sglang/srt/layers/moe/remote_ep_wrapper.py` in an SGLang source tree,
  or `sglang/srt/layers/moe/remote_ep_wrapper.py` in installed site-packages.
- `0001-wire-remote-ep-wrapper.patch` targets the installed site-packages layout
  used by `pkgs/sglang/default.nix`.
- `placement.example.json` maps experts 0--47 to node 0, 48--95 to node 1,
  96--143 to node 2, and 144--191 to node 3.

The patch attaches in
`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, in `FusedMoE.__init__`
near the existing KTransformers wiring (about line 318 in SGLang 0.5.14). The
decisive wiring line is:

```python
self.quant_method = RemoteEPWrapperMethod(gpu_method, remote_ep_config)
```

Remote EP takes precedence if both remote and KT arguments are configured.
Do not configure both; they represent competing foreign executors for the same
expert routes.

## Apply to an upstream source checkout

From the SGLang repository root at tag/version 0.5.14:

```bash
install -Dm644 \
  /path/to/nix-strix-halo/pkgs/sglang/graft/remote-ep/remote_ep_wrapper.py \
  python/sglang/srt/layers/moe/remote_ep_wrapper.py

patch -p1 -d python < \
  /path/to/nix-strix-halo/pkgs/sglang/graft/remote-ep/0001-wire-remote-ep-wrapper.patch
```

The patch also adds the five annotated `ServerArgs` fields. SGLang's annotated
argument generator exposes their underscore names as these CLI flags:

| Flag | Purpose |
| --- | --- |
| `--remote-ep-placement PATH` | Required JSON placement; enables the graft |
| `--remote-ep-endpoints CSV` | M1 node endpoints in owner-ID order |
| `--remote-ep-transport rccl-rdma` | Packed RCCL/OFI transport selector |
| `--remote-ep-client-class MODULE:OBJECT` | Optional replacement client factory |
| `--remote-ep-timeout-ms N` | Foreign-executor deadline, default 30000 |

## Promote into this repository's Nix package

`pkgs/sglang/default.nix` installs a wheel and applies every patch in
`pkgs/sglang/patches/` to its site-packages root. When the transport client is
ready for hardware integration, add the graft explicitly to `postInstall`
before the existing patch loop:

```nix
install -Dm644 \
  ${./graft/remote-ep/remote_ep_wrapper.py} \
  "$out/${pythonSitePackages}/sglang/srt/layers/moe/remote_ep_wrapper.py"
patch -p1 -d "$out/${pythonSitePackages}" < \
  ${./graft/remote-ep/0001-wire-remote-ep-wrapper.patch}
```

Do not also copy the graft patch into `pkgs/sglang/patches/`, or the loop will
apply it twice. The scaffold leaves this promotion as an explicit human step
The default `RemoteEPClient` imports M1's shared
`RCCLRDMAAllToAllTransport`; install that module or put
`pkgs/moe-expert-server` on `PYTHONPATH` as described in M1's
`RDMA_TRANSPORT.md`.

After applying, compile the installed files and confirm that CLI generation
sees the new fields before attempting model startup.

## Placement map

The placement file must cover every routed expert exactly once. It may use a
list, where the list index is the global expert ID, or an object whose keys are
contiguous decimal expert IDs. An integer value is the M1 node ID. `null` or
`"gpu"` keeps that expert hot on the dense GPU:

```json
{
  "experts": {
    "0": "gpu",
    "1": 0,
    "2": 3
  }
}
```

That shortened example is valid only for a three-expert test model. Use
`placement.example.json` for the full all-remote 192-expert design. The wrapper
validates the map length against SGLang's `num_experts` and validates every
node ID against `--remote-ep-endpoints`.

Hot experts need not form a prefix. During checkpoint loading the wrapper
skips remote weights and remaps arbitrary global hot IDs into compact local GPU
slots. During routing it applies the same lookup, writing `-1` only in the copy
passed to the GPU MoE method. The original global IDs always go to the remote
client. Fused shared experts are appended after routed expert IDs by SGLang;
the wrapper always retains and remaps those on the dense GPU, outside the
192-entry routed placement map.

## `RemoteEPClient` / M1 seam

The default `RemoteEPClient`, or the class selected by
`--remote-ep-client-class`, is constructed once per MoE layer as
`factory(RemoteEPConfig)`. It implements:

```python
load_weights(*, layer_idx, placement, endpoints) -> None
submit_forward(x, topk_ids, topk_weights, stream) -> None
sync_forward(x, stream) -> Tensor
```

`load_weights` does not copy model weights from SGLang: the M1 processes
already own them. It initializes or registers the layer's endpoint/placement
state. `submit_forward` must be non-blocking with respect to the supplied HIP
stream. `sync_forward` waits for dispatch, M1 compute, and inverse gather, then
returns one `[T, H]` tensor with `x.dtype` on `x.device`. Its result must
already include router weights and the sum across remote nodes, matching M1's
`ExpertTransport.send_response()` semantics.

The default client now packs assignments by node, exchanges split sizes,
issues asynchronous RCCL `all_to_all_single` operations over RoCEv2, and
inverts the exchange for results by importing M1's implementation. See M1's
`RDMA_TRANSPORT.md` for the rank layout and exact environment. HCA/GID
selection, host-bounce behavior, timeout handling, and all network details stay
behind this client/M1 transport boundary. There is no raw ibverbs in
`RemoteEPWrapperMethod`.

## DP-attention composition

The intended launch shape is one dense V620 rank per node and one M1 iGPU
server per node. A starting SGLang configuration is conceptually:

```bash
python -m sglang.launch_server \
  --model-path /srv/model \
  --nnodes 4 \
  --tp-size 4 \
  --dp-size 4 \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --remote-ep-placement /etc/sglang/remote-ep-placement.json \
  --remote-ep-endpoints strix-1:29700,strix-2:29700,strix-3:29700,strix-4:29700 \
  --remote-ep-transport rccl-rdma \
```

The normal four-node SGLang rendezvous arguments are omitted from this sketch.
`--enable-dp-attention` gives each dense rank a distinct batch slice; every
rank that enters the wrapper therefore submits its own local `x`. The wrapper
does not use KTransformers' `tp_rank == 0` guard, which would incorrectly drop
three DP slices in this topology. `--moe-dense-tp-size 1` keeps dense/shared
MoE-side work local to each batch slice. `--ep-size 1` and
`--moe-a2a-backend none` avoid a second SGLang expert sharding/all-to-all layer;
the placement map and M1 client own sparse EP.

This composition is not yet proven by the scaffold. SGLang's parallel-group
derivation can change across versions and model families, and
`intermediate_size_per_partition`, shared-expert handling, output reductions,
and DP padding must be inspected on the pinned build. Log the effective
attention DP rank, MoE TP/EP ranks, token count, and layer ID on all four ranks
before enabling performance paths.

## Known integration risks

- DP-attention plus a custom foreign executor is an unvalidated SGLang path.
  In particular, CUDA/HIP graph capture, overlap scheduling, padded DP batches,
  failure propagation, and stream ownership need hardware tests. Begin with
  CUDA graphs and overlap disabled, then enable them separately.
- The wrapper assumes the standard token-dispatcher named tuples and the
  `FusedMoEMethodBase` contract present in SGLang 0.5.14. Rebase the patch and
  rerun syntax/import tests when the packaged version changes.
- Arbitrary hot-expert loading is handled by a weight-loader adapter, but model
  loaders with a different callback signature or fused/shared experts need a
  targeted compatibility check. The 192-entry map is for routed experts only;
  the wrapper explicitly appends shared experts to the dense GPU slots.
- One wrapper allows one in-flight forward. A client that supports SGLang batch
  overlap must add request handles and make the wrapper's pending state
  multi-flight rather than silently reusing buffers.
- A collective client must make all participating ranks issue compatible RCCL
  operations even when a rank has zero assignments for a node.

## Deliberately not built

- The default `RemoteEPClient` reuses M1's packed RCCL/RDMA transport.
- The gfx1151 fp8 grouped-GEMM is a `GroupedExpertKernel` hook in M1.
- No raw RDMA/ibverbs code is included.
- The patch is not applied to the packaged SGLang by this scaffold.
