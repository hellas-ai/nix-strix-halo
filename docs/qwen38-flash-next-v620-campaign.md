# Qwen3.8-Flash-Next on 4x V620

## Decision and current status

The production target is **expert-only INT4 W4A16, symmetric group 32, FP16
activations, FP32 GDN state/accumulation where required, TP4, and PLE host
offload**. BF16 is the immutable quality oracle. Official FP8 is a quality
control, not a four-card serving format. Native MXFP4/NVFP4 is not available on
gfx1030 and must not be emulated in the production path.

As of 2026-08-28:

- the exact 131-shard BF16 revision is complete and inventory-clean on trex at
  `/mnt/optane/models/Qwen3.8-Flash-Next-BF16`;
- the G32 conversion is complete: 131 shards, 173.558 GiB logical size, 73,728
  routed expert projections, aggregate reconstruction RMSE `0.09651483`, and a
  closed-world compressed-tensors manifest that leaves every non-routed tensor
  unquantized;
- the source-built SGLang Qwen4-Exp support closure is pinned to PR head
  `73a255206f916366c8d26d4022f82ddfb0ab558d`;
- a two-pass HIP Triton kernel replaces QSA decode's NVIDIA-only FA2/FA4 path;
- gfx1030 import gates prevent unused AITER MXFP4/Quark paths from being
  selected merely because PyTorch exposes HIP through `torch.cuda`;
- the real Qwen4-Exp model import, including the in-tree Triton MoE runner and
  generic runner's EAGLE import fan-out, is `sgl_kernel`-free on HIP;
- the expert-only compressed-tensors producer and cautious TP4 launcher exist;
- all four V620s and the PEX switch are recovered on strix-2, while the Strix
  iGPU remains excluded from the heterogeneous process;
- TP4 construction selects `CompressedTensorsWNA16TritonMoE`, and a real
  checkpoint load stays within both the four 32 GiB cards and 128 GiB host RAM;
- PLE remains BF16 in the immutable checkpoint but is explicitly downcast into
  FP8 host embedding storage at load time, reducing its payload from
  102,466,171,160 bytes to about half that size. Gathered PLE values are
  converted back to the model dtype before computation;
- live serving qualification is in progress. Constructor or load completion is
  not a serving result; the gate remains an externally observed generated-token
  response.

The hardware recovery gate remains explicit: firmware **PCI Hot-Plug -> PCI
Buses Padding = 5**, remove independent PEX-board power until its LEDs are dark,
restore board power, then boot the host. Do not use a live PCI rescan; that has
caused HIP faults on this fleet. Before serving, `lspci` must show all four
V620s and the switch topology, and the selected model path must be a read-only
mount containing the verified checkpoint.

Upstream references:

- [Qwen3.8-Flash-Next BF16](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- [official Qwen repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [SGLang support PR #36497](https://github.com/sgl-project/sglang/pull/36497)
- [SGLang model cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)

## Cluster campaign triage

The 2026-08-28 campaign treats the hardware as two useful but distinct pools:
four gfx1030 V620s on strix-2, and one gfx1151 Strix Halo iGPU on each of
strix-1 and strix-2. Their memory does not become one transparent 256 GiB
address space. A heterogeneous or cross-host design must earn its keep through
an explicit SGLang transport; it cannot be justified by adding the capacity
numbers together.

| track | observed control | campaign decision |
|---|---|---|
| Qwen3.8-27B dense | TP4 V620 decode: 20.674 tok/s B1, 129.00 B8, 146.31 B16, 262.83 B32 | regression and thermal control |
| DeepSeek V4 Flash | prior TP2 decode: 10.99-11.51 tok/s prose and 14.62-16.50 tok/s code | strongest existing SGLang control; requalify without dead strix-3 |
| GLM-5.2 | one-node heterogeneous correctness exists; gfx1030 attention and production multi-host transport remain limiting | transport/correctness work, not the first serving target |
| Qwen3.8-Flash-Next | 125B total / 6B active hybrid model; expert-only G32 is resident only with PLE host compression and low-peak layout conversion | highest-value four-V620 experiment |

The sequence is therefore dense Qwen control, bounded DeepSeek TP2 control,
then Flash-Next TP4 on the four V620s. The two Halos remain available for
controls and future disaggregated work, but are not mixed into this TP group.
TP4 across four hosts is not tested while strix-3 is dead.

## Why this precision ladder

The V620 is RDNA2/gfx1030. It has fast native packed FP16 and INT4/INT8 dot
operations. It does not have native BF16, FP8, MXFP4, MFMA, or WMMA. A format
name in a checkpoint is not a hardware capability.

| candidate | checkpoint size | TP4 after BF16/FP8 PLE offload | decision |
|---|---:|---:|---|
| BF16 | 335.29 GiB | about 59.98 GiB/card | immutable oracle; cannot reside |
| official FP8 | 172.76 GiB | about 31.27 GiB/card | exceeds the cards' usable ~29.98 GiB before KV/workspace |
| official NVFP4 | smaller | Blackwell-only kernels | reject on gfx1030 |
| SGLang MXFP4 | smaller | gfx95 AITER path only | reject on gfx1030 |
| routed experts W4A16-G32 | 173.558 GiB full checkpoint; about 68 GB packed routed-expert payload | about 22 GiB/card observed during load, PLE excluded | primary candidate |
| routed experts W4A16-G64 | estimated 59.77 GiB expert payload | about 18.67 GiB/card total, PLE excluded | quality/latency control |
| routed experts W4A16-G128 | estimated 58.01 GiB expert payload | about 18.23 GiB/card total, PLE excluded | residency control only |

The W4 estimates follow directly from 48 layers, 512 routed experts per layer,
and three `640 x 2560` projections per expert: 120,795,955,200 routed-expert
parameters. G32 stores 4-bit values plus one FP16 scale per 32 values. The
estimate is conservative because the coding service uses
`--language-model-only` and never loads the vision tower.

The full G32 checkpoint is much larger than the packed expert payload because
PLE alone is 102,466,171,160 bytes in the immutable BF16 source. Four ranks
therefore cannot each retain BF16 PLE in the host. `ple_embedding_dtype` is an
explicit runtime storage decision, not a claim that the source checkpoint was
rewritten or that the V620 has native FP8 arithmetic.

G32 is the quality-first starting point. The campaign may promote an
activation-aware G32 checkpoint over round-to-nearest G32, but it may not
silently widen quantization to routers, the shared expert, PLE, GDN, QSA,
hyperconnections, norms, embeddings, `lm_head`, or MTP. Those tensors remain in
their source precision on disk and load as native FP16 after a complete BF16 to
FP16 range audit.

## Immutable inputs and reproduction

The campaign worktree is `/mnt/Home/src/nix-strix-halo-qwen38-flash-next` on
`feat/qwen38-flash-next-v620`, forked from the final corrected Qwen3.8 V620
campaign at `4a6ae1a4`. The main `npu-exporter` worktree and the dirty
`nixos-config` worktree are deliberately untouched.

Pinned inputs:

| input | revision |
|---|---|
| Qwen BF16 | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| SGLang support | `73a255206f916366c8d26d4022f82ddfb0ab558d` |
| SGLang source hash | `sha256-7idPvXJCurLcQXcpppOuZjvoy/mEYIWB85ItBCQK/pI=` |
| Transformers | `5.12.1` wheel, used by the SGLang-local Qwen4-Exp config |

Build and resolve the exact closure:

```console
nix build .#legacyPackages.x86_64-linux.gfx1030.sglang-qwen38-flash-next-rocm \
  --out-link .bench-artifacts/closures/sglang-qwen4-exp-gfx1030-pr-73a2552
closure=$(readlink -f .bench-artifacts/closures/sglang-qwen4-exp-gfx1030-pr-73a2552)
```

`sglang version` must report both
`0.5.17.dev0+qwen4exp.73a2552` and git revision `73a2552`.

Before device work, run the converter's fused-layout self-test and the offline
gfx1030 QSA compiler gate. The admitted score/value kernels use 166/193 VGPRs
and zero bytes of private scratch; the rejected one-pass design spilled as much
as 4,272 bytes.

```console
smoke=$(mktemp -d -p .bench-artifacts qwen-w4-smoke.XXXXXX)
SGLANG_USE_AITER=0 \
  "$closure/bin/qwen38-flash-next-runtime-smoke"
"$closure/bin/qwen38-flash-next-quantize-experts" \
  --self-test-dir "$smoke" --group-size 32
"$closure/bin/qwen38-flash-next-qsa-aot" \
  --output "$smoke/qsa-gfx1030-aot.json"
```

The runtime smoke gate imports the actual Qwen4-Exp module through its
activation, MoE, model-runner, and EAGLE dependencies and fails if
`sgl_kernel` enters the process. `sglang serve --help` is not a substitute: it
does not import the model implementation.

Resume and verify the BF16 staging download with
[`qwen38-flash-next-stage-bf16.sh`](../scripts/qwen38-flash-next-stage-bf16.sh).
The verifier reads every safetensors header, checks the index-to-shard mapping,
and refuses a partial 131-shard checkpoint.

For the multi-hour transfer, run it as a restartable user service rather than
leaving it attached to an agent shell:

```console
systemd-run --user \
  --unit=qwen38-flash-next-bf16-download.service \
  --working-directory=/mnt/Home/src/nix-strix-halo-qwen38-flash-next \
  --property=RuntimeMaxSec=infinity \
  --property=Restart=on-failure \
  --property=RestartSec=15s \
  --collect \
  bash scripts/qwen38-flash-next-stage-bf16.sh
```

The SPDK namespace root is intentionally root-owned. The stage script requires
the pre-provisioned model directory and cache directory themselves to be
writable; it does not require permission to create arbitrary top-level model
names. A successful unit proceeds directly from the pinned download to the
complete inventory and writes `.campaign-revision` only after verification.

After the BF16 gate passes, create the first resident checkpoint with the
quantizer installed in the exact SGLang closure:

```console
closure=$(readlink -f .bench-artifacts/closures/sglang-qwen4-exp-gfx1030-pr-73a2552)
"$closure/bin/qwen38-flash-next-quantize-experts" \
  --source /mnt/optane/models/Qwen3.8-Flash-Next-BF16 \
  --output /mnt/optane/models/Qwen3.8-Flash-Next-W4A16-G32 \
  --group-size 32
```

The official source stores each language layer as two fused 3-D tensors,
`experts.gate_up_proj` and `experts.down_proj`. The conversion is resumable per
shard and immutable per plan. It requires exactly `48 * 2 = 96` fused language
expert tensors, slices all 512 experts, and emits exactly
`48 * 512 * 3 = 73,728` per-expert packed projections in the naming convention
SGLang maps to W13/W2. MTP's two fused expert tensors remain bit-identical and
unquantized. The converter records aggregate reconstruction error and produces
a new internally consistent safetensors index.

Do not publish `/mnt/optane/models` read-write. The first campaign publication
uses trex's read-only NFS export. Linux NFSv4 recognizes trex's management and
fabric addresses as one server, so merely mounting `192.168.25.8` can silently
reuse the existing `192.168.23.8` transport. Verify both the RPC transport's
source/destination addresses and interface byte counters; the pathname and
mount source string are not proof. The 2026-08-28 campaign migrated the shared
RPC transports to `192.168.25.102 -> 192.168.25.8` and measured a cold 3.47 GB
shard read entirely on `cx5fabric0` (3.51 GB fabric RX versus 164 KB management
RX). Persist that export and routing in the fleet configuration before treating
it as production.

## Admission gates

No optimization advances without its gate artifact under `.bench-artifacts`.

### G0: hardware and topology

- Four V620s appear as the only devices in `HIP_VISIBLE_DEVICES=0,1,2,3`.
- The Strix iGPU is excluded.
- PEX P2P/IPC and RCCL all-reduce pass before SGLang starts.
- The selected model path is a read-only publication containing the verified
  checkpoint, and interface counters prove that loading uses the intended
  fabric rather than the management LAN.
- Board clocks, junction temperature, fan state, and throttling counters are
  recorded before and after each run.

### G1: checkpoint fidelity and consumption

- All 131 BF16 shards and all index entries pass header/offset/size validation.
- Every BF16 value in every tensor is scanned before runtime FP16 conversion;
  overflow is a hard failure and flush-to-zero counts are reported by tensor.
- The quantized index contains exactly three packed tensors for each routed
  projection and unchanged names for every non-routed tensor.
- SGLang startup emits a parameter-consumption manifest. Missing, silently
  dropped, multiply consumed, or unexpected tensors are hard failures. This is
  mandatory because the DS4 0.5.14 loader silently dropped MTP tensors.

### G2: numerical correctness

- W4 pack/unpack round-trips its INT4 values exactly.
- Dequantized tensor error is measured with the FP16 scale actually stored in
  the checkpoint, not an FP32 pre-rounding scale.
- QSA HIP eager decode is compared against its FP32 reference over random and
  adversarial index layouts, including paged logical-to-physical mapping.
- Both QSA passes must lower for `hip/gfx1030` with zero private scratch before
  device testing; offline lowering is not a numerical or performance claim.
- Eager and graph modes produce identical greedy tokens for at least 1,600
  tokens across code, prose, long-prefix, tool-call, and multi-request cases.
- Radix-cache cold and warm outputs match. Speculative mode must match greedy
  target tokens before any speed result is admitted.
- Tool parsing must produce populated `tool_calls`; token-0/`!` loops are a
  hard failure, not a parser cosmetic.

### G3: quality

Use the BF16 checkpoint or an externally provisioned BF16 oracle for the
quality baseline; the 188 GiB trex host cannot hold the 335 GiB oracle. Freeze
prompts and reference outputs before comparing G32/G64/G128.

The quality suite must include:

- held-out code completion and repository-edit tasks;
- function/tool calling with parallel and sequential calls;
- long-context retrieval at 8K, 32K, 64K, and 128K;
- preserved-thinking and non-preserved-thinking conversations;
- perplexity or teacher-forced NLL on code and prose corpora;
- the two-subagent coding acceptance fixture described below.

Choose by quality first among variants that meet residency. G64/G128 exist to
measure the quality-memory curve, not to justify avoidable degradation. If RTN
G32 misses the frozen threshold, collect expert-input activation statistics on
the resident model and produce an activation-aware G32 checkpoint; do not
quantize additional tensor classes to make an easier kernel.

### G4: performance

Start from eager correctness, then admit one change at a time:

1. CUDA graphs after graph/eager parity.
2. Radix/prefix caching with `cached_tokens` evidence.
3. QSA Triton decode against the HIP reference.
4. GDN prefill and decode tiles tuned separately.
5. W4 MoE tiles for actual Flash-Next shapes and batch regimes.
6. NGRAM `k={2,4,8}` and native MTP/EAGLE, separately by workload.
7. Scheduler/concurrency and memory fraction.

Measure:

| axis | required points |
|---|---|
| decode latency | batch 1; contexts 1K, 8K, 32K, 64K |
| aggregate decode | concurrency 1, 2, 4, 8, 16, 32 |
| TTFT | 1K, 8K, 31K, 64K, 128K, cold and exact-prefix warm |
| cache | exact hit, one-token divergence, unique nonce miss, explicit flush |
| speculation | code edit, code generation, tool JSON, prose, long-context |
| stability | cold and sustained 30-minute runs with thermal telemetry |

Trust server-reported generation throughput for aggregate results. Blocking
Python client threads understated the old server by up to 4x. Every client
prompt gets a nonce unless a cache hit is the independent variable. Warm timing
starts no earlier than the third run. Variants run ABAB with fresh prompts and
min-of-N reporting. Buffers for bandwidth/kernel work rotate over more than 128
MiB so the result is DRAM, not Infinity Cache.

Use the async benchmark client from the exact closure; it emits append-only,
fsynced JSONL with cold/warm cache evidence and ABAB block labels:

```console
"$closure/bin/qwen38-flash-next-bench" \
  --out .bench-artifacts/perf/prefix.jsonl \
  --variant A --block-index 0 --trials 3 --samples 4 \
  --concurrency 4 --prefix-tokens 16384 --max-new-tokens 128
```

## Coding-agent acceptance

The target endpoint is OpenAI-compatible at
`http://127.0.0.1:30800/v1`, served model `qwen3.8-flash-next`, with
`qwen3_coder` tool parsing and `qwen3` reasoning parsing. Preserve thinking by
default because it improves conversational continuity and makes stable prefixes
reusable; test the opt-out explicitly.

Port the production fixture from
`/mnt/Home/src/nix-strix-halo-ds4-agent-prod/docs/production/ds4-agent.md` to
both OpenCode/OMP and Pi. A passing run must:

1. launch two distinct subagents with non-identical investigations;
2. read the required repository files before editing;
3. reproduce the intentionally failing boundary test;
4. make the single minimal correction;
5. run the focused and full tests;
6. inspect the final diff and status;
7. leave structured JSONL proving tool-call identity, order, inputs, outputs,
   and completion.

Prose claiming that work happened is not evidence. Duplicate subagents, empty
tool results, edits before investigation, or a correct final file without the
required process all fail. Run the fixture at concurrency 1 and with multiple
simultaneous parent agents; fast B1 decode alone is not a good subagent service.

Pi uses the same local Chat Completions endpoint through the repository's
durable-runtime wrapper. Match its advertised limits to the qualified server;
the Pi/OMP subagent extension remains an independently tested client component.

```console
export OPENAI_BASE_URL=http://127.0.0.1:30800/v1
export OPENAI_API_KEY=qwen-local-no-auth
export OPENAI_MODEL=qwen3.8-flash-next
export PI_CONTEXT_WINDOW=32768
export PI_MAX_TOKENS=8192
export PI_REASONING=true
export PI_WRAP_RUNTIME_DIR=/mnt/Home/src/.runtime/qwen38-flash-next-pi
nix run .#pi-wrap
```

## Lessons carried forward

- The old Qwen frontier was 20.674 tok/s B1, 33.832s cold 31K TTFT, and
  0.464s warm 30K TTFT. Prefix reuse was the largest user-visible win: 74x.
- NGRAM averaged 1.35x but ranged from 2.02x for a code edit to 0.92x for prose.
  It remains workload-controlled.
- CUDA graphs can erase standalone dispatch/fusion gains; measure end to end.
- M<=8 GEMV and larger-M GEMM need different kernels. Do not inherit the old
  dense model's hard-coded shapes into this MoE.
- A 1.74x DS4 MoE kernel gain became only 3.1% end to end. Component wins need
  server evidence.
- Autotune keys include sequence length and every shape dimension. Record the
  runtime-selected config, not merely the intended config.
- Exact resolved Nix store paths and `/proc/<pid>/environ` are the truth.
- Hugging Face Xet can buffer tens of gigabytes and finalize several shards in
  one burst. Diagnose a stalled download from shard finalization together with
  Xet controller progress, cgroup memory, PID I/O/CPU, sockets, and link errors;
  a flat `du` sample alone is not evidence of a dead transfer.
- The old V620s spent 90-95% of sustained work throttled near 106-107 C;
  sustained TTFT regressed 11.9% and decode 7.9%. Thermal telemetry and ABAB
  ordering are correctness requirements for performance claims.

The prior corrected evidence remains in
`/mnt/Home/src/nix-strix-halo-qwen38/.bench-artifacts/CAMPAIGN-INDEX.md` and
`/mnt/Home/src/nix-strix-halo-ds4-focus/.bench-artifacts/DS4-CAMPAIGN-INDEX.md`.
Where an older per-experiment note disagrees with those indexes, the index wins.

## First qualified launch

Resolve the closure, select the published W4 snapshot, and start eagerly:

```console
export SGLANG_QWEN38_FLASH_NEXT_CLOSURE="$(readlink -f \
  .bench-artifacts/closures/sglang-qwen4-exp-gfx1030-pr-73a2552)"
export MODEL=/mnt/trex-models-fabric/Qwen3.8-Flash-Next-W4A16-G32
export CUDA_GRAPH=0
bash scripts/qwen38-flash-next-serve.sh
```

The launcher forces TP4, FP16, FP32 SSM state, FP16 GDN convolution state,
language-model-only loading, PLE host offload, Triton attention/linear
attention/MoE, Torch dense GEMM, durable caches, and V620-only visibility. It
also disables the unavailable custom all-reduce extension and drops checkpoint
page cache after loading. It does not enable MXFP4, FP8 compute, AITER,
FlashInfer, TileLang, CuTe, or speculation by accident. The model pathname above
records the campaign mount; a persistent deployment may use another read-only
mount only after passing the same inventory and fabric-counter gates.
