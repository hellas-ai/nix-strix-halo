# GLM-5.3-Flash performance audit

2026-09-29. Four gfx1151 Strix Halo nodes, TP=4, one decoding request,
AWQ W4A16 routed experts, BF16 dense weights, stable FP32 routing, RoCE,
128K configured capacity. The timed completion has a **12-token prompt**;
these throughput numbers do not qualify decode at 128K occupied context.

## Established baseline

| Change | Decode tok/s | ms/output token |
|---|---:|---:|
| Stable router, TCP | 8.81 | 113.50 |
| Repaired RoCE | 9.63 | 103.82 |
| Split mHC FP32 projection | 11.65 | 85.84 |
| W4 decode reduction tile 64 → 128 | **13.68** | **73.08** |

Each throughput result is the median of three 128-output-token streaming
requests after warmup. The latest runtime is
`j95k13kapgc1a26v8q9xh2w57xwc4cmr-sglang-rocm-gfx1151-0.5.20`.
The W4 tile change preserves all checked token IDs and log probabilities
from the mHC-only run, as well as all synthetic component outputs on all
four GPUs. API and five cache-branch checks pass. Decode/prefill probability
maxima remain 0.0634/0.1079 against a 0.05 limit, with no token-choice
disagreement across 64 sampled positions. This numerical gate remains failed.

Evidence: [RoCE](../lib/bench/results/glm53-roce-2026-09-29.json),
[mHC projection](../lib/bench/results/glm53-mhc-projection-2026-09-29.json),
[W4 tuning](../lib/bench/results/glm53-w4-tuning-2026-09-29.json).

## Independent memory bound and its limits

Counting checkpoint tensors and the runtime's TP/replicated placement gives
approximately **5.85 GB of weight reads per rank per token**, plus a minimum
0.071 GB of recurrent-state reads/writes. Of the weight bytes, **1.09 GB are
selected W4 experts; 4.76 GB are other weights**. Calling the checkpoint
"4-bit" does not make its dense projections four-bit.

At this campaign's measured 228 GB/s streaming-add rate, this is **25.97 ms**,
or **38.51 tok/s**, before attention traffic, communication and execution
overhead. The latest 13.68 tok/s is about **36% of this optimistic bandwidth
bound**, or a 2.8× gap. The original 8.81 tok/s was about a 4.4× gap.

This is not a complete serving roofline. It assumes each selected weight is
read once, models TP partitioning from source, and does not measure actual
memory transactions. Small weights can hit cache; generic kernels can also
reread weights or fetch inefficiently. The count needs reconciliation against
live tensor shapes and hardware counters. Historical Halo **pure-read** probes
reached 241.5 GB/s; using that rate gives an even more optimistic 24.52 ms /
40.79 tok/s bound. Streaming-add and pure-read rates measure different traffic.
Neither an achieved GEMV rate nor measured full-model latency defines the
hardware ceiling.

## What the full GPU trace establishes

The device trace was collected **before** the two latest kernel changes,
at the 9.63 tok/s baseline, with decode graphs enabled. Sixteen decode steps
on every rank had 3,952 dispatches/token. Strix-1's nonoverlapping kernels sum
to 99.27 ms/token:

| Work | Device ms/token | Evidence |
|---|---:|---|
| W4 experts | 31.12 | 42 gate/up calls: 26.64 ms; 42 down calls: 4.48 ms |
| FP32 BLAS | 21.12 | 101 tiny-grid calls: 18.76 ms; 42 router calls: 2.36 ms |
| BF16 BLAS | 26.09 | Dense/shared/attention projections and output head |
| RCCL | 6.39 | 92 collectives; other ranks reach 8.33 ms |
| Other kernels | 14.55 | Includes 113 float×BF16 elementwise multiplies taking 4.98 ms |

The synchronized trace region is about 108 ms/token, versus 104 ms/token
unprofiled. Its gaps and profiling overhead must be separated before assigning
a launch-overhead budget. Collective durations include peer waiting. The
measured A/B speedups establish that mHC and W4 were real costs, but a new
trace is needed to assign the **current** 73 ms precisely. Do not subtract
microbenchmark gains from this table and present the result as a new trace.

See [full trace summary](../lib/bench/results/glm53-gpu-profile-2026-09-29.json).

## Runtime paths requiring attention

1. **Expert decode uses generic padded matrix tiles.** SGLang's
   `fused_moe_kernel_gptq_awq` pads a single routed token into a 16-row tile.
   The original expert path achieves only about 35 GB/s of useful weight
   bytes when its 1.09 GB count is divided by 31.12 ms. That is not measured
   DRAM bandwidth. Transaction amplification, register pressure, instruction
   overhead and insufficient memory concurrency need separating. The trace
   reports 168 VGPRs/thread and no scratch allocation for these kernels.
2. **RMSNorm falls back to separate Torch operations.** With AITER disabled
   and no vLLM custom op available, `RMSNorm.forward_hip` selects
   `forward_native`: casts, square/reduction, normalization and multiplication.
   The count of 113 slow mixed-dtype multiplies matches the model's ordinary
   norm calls, but stage annotations/counters should confirm this attribution.
3. **KDA projection fusion is disabled by the global quantization object.**
   `Glm5NextLinearAttention.do_fuse_qkvbfg` requires `quant_config is None`.
   This checkpoint has quantized experts and unquantized KDA projections, so
   the entire model's quantization setting prevents that projection fusion.
   Enabling it requires correct weight loading and BF16-rounding checks;
   changing the boolean alone is not a qualified fix.
4. **Small dense shapes are not automatically bandwidth efficient.** The
   generic FP32 mHC projection launched only two workgroups. The dedicated
   split projection removed about 18 ms/token end to end. Other skinny shapes
   should be measured individually, while preserving the larger projections
   that already stream weights efficiently.

## Historical evidence and reusable work

The July GLM-5.2 campaign retained a hand-written HIP W4 GEMV kernel in
`pkgs/strix-int4-hip/`, including vector weight loads, independent accumulator
chains, architecture-specific wave layouts and non-temporal loads. A real
checkpoint c1 test measured **170.4 effective packed-weight GB/s**, versus
182.2 at c32. The c1 measurement is useful evidence that the Halo can execute
this class of decode efficiently; it is not a compatible GLM-5.3 kernel result.

Its arithmetic and ABI differ: asymmetric zero arrays, intermediate width
divisible by 1024, FP32 group-scaled accumulation, no GLM-5.3 clamp, and FP32
atomic route reduction. Current GLM-5.3 uses width 512 per rank, symmetric
zeros, BF16-rounded dequantized weights and gate/up outputs, clamp 10, stable
route ordering and scale 2.5. Reuse the memory-access structure while retaining
these rounding and ordering rules. The old README's wave32 description is
stale relative to its final wave64 gfx1151 source.

The GLM-5.2 "92.65% of full-system roofline" report divides observed serving
speed by a denominator built from **measured full-model pass time**. It
accounts for zero speculative acceptance and omitted components, but does
not establish hardware efficiency. It also used four V620 dense GPUs plus
four Halo expert workers at batch 32, a different topology and workload.

The August DeepSeek-V4 campaign found large gains by auditing active paths:
an environment assignment kept the optimized indexer disabled (8.19 → 13.29
tok/s), and a missing FFN mHC fusion left 44 tiny FP32 GEMVs per token
(13.29 → 15.85 tok/s after fusion). It also corrected a supposed 196 GB/s
hardware ceiling to a measured 241.5 GB/s pure-read ceiling. Its mixed
MXFP4/BF16 kernels and the separate Q8 campaign are not directly comparable
to this checkpoint. Their relevant lessons are to verify actual dispatch,
count replicated traffic and retain graph-enabled profiling.

Historical sources retained on trex:

- `nix-strix-halo-inkling/pkgs/strix-int4-hip/int4_moe.hip` and `.h`;
  real-weight results `/mnt/Home/glm52-4node-serve/results/int4-kernel-roofline-strix1.json`.
- `nix-strix-halo-inkling/docs/glm52-serve-baseline.md`, 2026-07-27.
- `nix-strix-halo-ds4-focus/.bench-artifacts/DS4-CAMPAIGN-INDEX.md`,
  especially the 2026-08-18 bandwidth correction and TP4 path audit.
- `/mnt/Home/src/DS4_Q8.md`, separate GGUF/Q8 evidence.

## Next measurements and decisions

Freeze the 13.68 tok/s baseline while making a current per-stage latency and
traffic budget. Collect hardware memory-request counters for expert and
dense kernels, and annotate norms/KDA boundaries so fallback attribution is
measured. gfx1151 exposes `FETCH_SIZE`/`GL2C_EA_RDREQ_*`; these measure L2
external traffic, which must not be casually relabeled as DRAM bytes when
the system cache can service it.

Then choose work by full-token time saved: adapt the proven HIP expert
memory-access pattern with GLM-5.3 arithmetic intact; remove norm/mHC fallback
dispatches; qualify KDA projection fusion. Each candidate needs real weights,
CPU/reference checks, unchanged or explained numerical behavior, then an
isolated full-model A/B. Continue the existing agent, long-context and cache
acceptance after selecting a runtime. The approximately 80%-of-roofline
objective remains unmet; it must not be achieved by redefining the denominator
as the current implementation's measured pass time.
