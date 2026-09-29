# GLM-5.3-Flash performance audit

2026-09-29. Four gfx1151 Strix Halo nodes, TP=4, one decoding request,
stable FP32 routing, RoCE,
128K configured capacity. The timed completion has a **12-token prompt**;
these throughput numbers do not qualify decode at 128K occupied context.
The current quality-preserving optimization work uses the official FP8
checkpoint. Earlier AWQ measurements are retained separately below.

## Official FP8 decode

Exact FP8-to-FP16 widening and smaller batch-one tiles improve full-model
decode from **1.97 to 8.90 tok/s (4.52×)**. Median TTFT is 1.395 seconds.
Both consistency prompts reproduce the original FP8 baseline's token IDs
and all checked generated/teacher-forced probabilities bit for bit. The
cache-branch results also remain unchanged. This establishes preservation
of those measurements, not overall model correctness: decode/prefill maxima
remain 0.1918/0.1235 and truncation remains 0.0798, above the 0.05 limit.
The first official-FP8 Pi coding task passes all 14 held-out checks. Three-marker
retrieval passes at 129,017 prompt tokens: 943.40 seconds cold and 6.60 seconds
warm, with an exact repeated answer and 128,960 tokens reused. This does not
qualify general long-context coding reliability.

The generic dense kernel's implicit FP8 conversion generated thousands of
compare/select instructions and 384 bytes of private scratch per work item.
The selected decode kernel has no private scratch and uses BM16/BN64/BK128,
four warps and one stage. Other batch sizes retain their prior tile defaults.
All 256 FP8 bit patterns pass widening checks on all four nodes; paired
dense and MoE probes preserve outputs exactly. Component tuning times include
a 256 MiB cache flush and must not be interpreted as pure kernel times.

The tested package is `fy1c1ig0frvfb8jya52mcwgsrkxf4j2n-sglang-rocm-gfx1151-0.5.20`.
GPUs reported approximately 2.9 GHz and 38–43°C edge temperatures during
qualification. The 8.90 tok/s result is about **24%** of the optimistic
36.65 tok/s memory bound below; reaching the target still requires substantial
work. Evidence: [FP8 decode](../lib/bench/results/glm53-fp8-decode-2026-09-29.json).

## Established baseline

| Change | Decode tok/s | ms/output token |
|---|---:|---:|
| Stable router, TCP | 8.81 | 113.50 |
| Repaired RoCE | 9.63 | 103.82 |
| Split mHC FP32 projection | 11.65 | 85.84 |
| W4 decode reduction tile 64 → 128 | **13.68** | **73.08** |

Each throughput result is the median of three 128-output-token streaming
requests after warmup. The measured AWQ runtime is
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

The official FP8 checkpoint changes this byte estimate to **6.15 GB of weights
per rank per token**, plus the same minimum recurrent-state traffic: about 5%
more than the community AWQ checkpoint. More expert bytes are partly offset
by dense weights that the official checkpoint stores in FP8. The ideal bound
at 228 GB/s is 36.65 tok/s. Its initial generic-kernel result is only 1.97
tok/s, so this small byte-count difference does not explain the slowdown.
The compiled dense and expert kernels use FP16 WMMA after FP8 conversion.
The widening and batch-one tile changes above remove major instruction and
padding costs; the byte-count bound itself is unchanged.

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

## Trace after the mHC and W4 fixes

A new 16-step trace on all four nodes after the power-strip reset confirms
**3,502 dispatches/token**, down from 3,952. Strix-1's nonoverlapping kernel
intervals total **68.15 ms/token**, with a synchronized region median of
76.84 ms. The established unprofiled result is 73.08 ms/token.

| Work | Device ms/token, Strix-1 |
|---|---:|
| BF16 BLAS | 26.06 |
| W4 experts | 18.91 |
| FP32 BLAS | 3.33 |
| Dedicated mHC projection | 0.93 |
| RCCL | 5.25 |
| 113 mixed-dtype norm weight multiplies | 5.00 |
| Other kernels | 8.66 |

Other ranks have 18.54–18.67 ms of expert work and 6.65–7.34 ms in collectives.
The trace directly confirms the reductions in mHC and expert time; BF16 BLAS
and normalization remain significant. No hardware traffic counters were
collected in this trace. See
[updated profile](../lib/bench/results/glm53-gpu-profile-tuned-2026-09-29.json).

## Expert memory counters

A separate Strix-3 probe collected `FETCH_SIZE` from the matching ROCm wheel
profiler. A known 256 MiB read measured 268,448,896 bytes, within 0.0051% of
the expected 268,435,456 bytes, at 215.66 GB/s. These are **L2 external
reads**, which can hit the system cache; they are not an unqualified DRAM
bandwidth measurement.

With the actual per-rank dimensions and synthetic expert weights, increasing
the reduction tile from 64 to 128 cut median gate/up traffic from **38.74 MB
to 17.97 MB**. The selected packed weights and scales total 17.04 MB. The
earlier tile therefore caused substantial read amplification, and the
accepted change removes most of it. Down-projection reads remain 8.66 MB
against an 8.52 MB payload.

The larger tile uses 256 VGPRs/thread and reaches only about 43 GB/s in the
gate/up counter probe. Eliminating redundant reads did not make the kernel
stream efficiently. Register pressure, instruction cost and memory concurrency
still need separating. Counter-instrumented latency is not full-model latency.
See [counter samples and calibration](../lib/bench/results/glm53-w4-counters-2026-09-29.json).

## Normalization experiments remain unqualified

Fusing the full normalization reached 15.31 tok/s but increased the cache
truncation probability delta to 0.0942, above the unchanged 0.05 gate. A
narrower candidate preserving Torch statistics reached 14.83 tok/s and matched
the native implementation exactly in 298 component cases on each GPU, yet
changed full-model probabilities and one generated sequence. Neither change
is accepted on the strength of its speed or component checks.

Restarting the accepted runtime after the user's reboot reproduced the earlier
decode and prefill probabilities bitwise for both consistency prompts, at
13.83 tok/s. A same-process native/candidate/native comparison found **zero
differing norm elements across all 113 norms on all four ranks**. All three
phases had identical full-model probabilities, matching the candidate runtime
and differing from the uninstrumented baseline. The checked norm arithmetic
is therefore not the source of this difference; allocation, layout or another
execution dependency needs investigation. The candidate is retained under
`lib/bench/experiments`, outside the package's applied patches.
See [trial results](../lib/bench/results/glm53-rmsnorm-investigation-2026-09-29.json).

## Runtime paths requiring attention

1. **Expert decode uses generic padded matrix tiles.** SGLang's
   `fused_moe_kernel_gptq_awq` pads a single routed token into a 16-row tile.
   The original expert path achieves only about 35 GB/s of useful weight
   bytes when its 1.09 GB count is divided by 31.12 ms. That is not measured
   DRAM bandwidth. Transaction amplification, register pressure, instruction
   overhead and insufficient memory concurrency need separating. The trace
   originally reported 168 VGPRs/thread and no scratch allocation. The tuned
   gate/up counter probe reports 256 VGPRs/thread with no scratch allocation.
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

Another DS4 candidate, `nix-strix-halo-ds4-focus/pkgs/sglang/patches/0061-w8a8-m1-gemv-contiguous-k.patch`,
reads contiguous K columns for batch-one GEMV. Its reported cold-weight
108–144 GB/s results used FNUZ weights, while current GLM uses FN weights.
The old synthetic checks also reported a few one-ULP differences; no complete
model probability-parity result was found. Retest this memory-access design
with FN operands and independent references before drawing conclusions.

## Next measurements and decisions

Keep the verified FP8 widening/tile runtime while expanding coding and 128K
acceptance beyond the first bounded checks. Next compare the contiguous-K GEMV design with current FN kernels,
then collect a new full-model FP8 trace to locate the remaining token time.
The earlier expert counter probe establishes AWQ read amplification; its
numbers must not be reused as FP8 traffic measurements. gfx1151 exposes
`FETCH_SIZE`/`GL2C_EA_RDREQ_*`; these measure
L2 external traffic, which must not be casually relabeled as DRAM bytes when
the system cache can service it.

Then choose work by full-token time saved: adapt the proven HIP expert
memory-access pattern with GLM-5.3 arithmetic intact; remove norm/mHC fallback
dispatches; qualify KDA projection fusion. Each candidate needs real weights,
CPU/reference checks, unchanged or explained numerical behavior, then an
isolated full-model A/B. Continue the existing agent, long-context and cache
acceptance after selecting a runtime. The approximately 80%-of-roofline
objective remains unmet; it must not be achieved by redefining the denominator
as the current implementation's measured pass time.
