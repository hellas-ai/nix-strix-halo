# Qwen3.8-Flash-Next V620 qualification evidence

Date: 2026-08-28

Host: `strix-2`

Hardware: four AMD Radeon Pro V620 cards, TP4, HIP devices `0,1,2,3`; the
Strix Halo iGPU was excluded.

Model: `/mnt/trex-models-fabric/Qwen3.8-Flash-Next-W4A16-G32`, read-only
publication of the 131-shard, 173.558 GiB expert-only W4A16-G32 checkpoint.

Package:
`/nix/store/816iqv6ikd913gzbkqz1gy605rszfz7m-sglang-rocm-gfx1030-0.5.17.dev0+qwen4exp.73a2552`

Unit: `codex-qwen38-flash-next-tp4-v29.service`

Raw log on the campaign host:
`.bench-artifacts/serve/serve-20260828-222635.log`

## Load and readiness

The server loaded all 131 shards, skipped 333 language-model-excluded visual
weights on every rank, allocated a 4,096-token cache, and reported:

```text
[2026-08-28 22:29:43 TP0] max_total_num_tokens=4096, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=1, context_len=4096, available_gpu_mem=8.50 GB
[2026-08-28 22:29:44] The server is fired up and ready to roll!
```

The packaged launcher resolved its own immutable closure. Its logged arguments
showed `disable_overlap_schedule=True`; CUDA graphs were also disabled. The
launcher set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`, so `/health`
observed readiness without injecting a model request.

## External completions

The first request disabled thinking, used greedy decoding, and asked for an
exact response. The OpenAI-compatible endpoint returned HTTP 200 in 5.271655
seconds from the final packaged closure:

```json
{"content":"V620 FLASH WORKS","prompt_tokens":23,"completion_tokens":8,"reasoning_tokens":0}
```

The immediately following request had a 59-token prompt and asked for roughly
120 words explaining mixture-of-experts routing. It returned HTTP 200 in
11.075127 seconds with 140 completion tokens. The response began:

```text
A Mixture-of-Experts (MoE) model scales capacity by replacing dense feed-forward layers with multiple specialized expert networks.
```

A third arithmetic control returned exact content `14` for 420 tokens over 30
seconds: HTTP 200, 36 prompt tokens, three completion tokens, and 0.378691
seconds. Passive health returned HTTP 200 in 1.4--1.6 ms after every request.

The server recorded the padded 64-token QSA prefill tile and warm decode:

```text
[2026-08-28 22:30:08 TP0] Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 0, input throughput (token/s): 56.23
[2026-08-28 22:30:14 TP0] Decode batch, #running-req: 1, #full token: 192, gen throughput (token/s): 12.54
[2026-08-28 22:30:17 TP0] Decode batch, #running-req: 1, #full token: 192, gen throughput (token/s): 13.93
```

## Failure reproduced and closed

The preceding v24 run returned a valid short completion, then the longer
prompt reproduced the gfx1030 QSA prefill failure:

```text
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 67584, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
```

v26 used one Triton software-pipeline stage for ROCm QSA prefill while retaining
the upstream tile and warp count. Both requests returned valid output, but an
asynchronous `at::native::vectorized_gather_kernel` hardware exception then
faulted all four ranks. The traceback showed SGLang's overlap event loop and QSA
gathers from shared ring buffers; SGLang enables its default write-after-read
barrier from `is_cuda()`, which is false for HIP in this branch.

v27 disabled scheduler overlap. The campaign's readiness poll then exposed a
second edge: upstream `/health` defaults to generating one token from raw input
ID `[0]`, and that non-chat QSA probe faulted. The final launcher therefore
keeps overlap disabled and makes `/health` passive. v28 passed seven sequential
chat completions in total, including a post-idle exact `STILL STABLE`
completion, with passive health between the preceding requests. v29 repeated
the acceptance set from the exact packaged closure and returned exact `STILL
PACKAGED` after its own idle interval. No `sgl_kernel`, AITER, or NVIDIA-only
backend entered the serving path.

## Build gates

The exact reviewed revision passed:

```text
nix build .#legacyPackages.x86_64-linux.gfx1030.sglang-qwen38-flash-next-rocm
/nix/store/816iqv6ikd913gzbkqz1gy605rszfz7m-sglang-rocm-gfx1030-0.5.17.dev0+qwen4exp.73a2552

nix build .#sglang-qwen38-flash-next-rocm
/nix/store/n67vv71dw7axn9vvlypnncn1843w2c28-sglang-rocm-gfx1151-0.5.17.dev0+qwen4exp.73a2552

nix build .#legacyPackages.x86_64-linux.gfx1030.vllm-rocm
/nix/store/lshnzk3143w5fdlggwm1gkbjhafrkgb2-python3.13-vllm-0.25.1

nix build .#vllm-rocm
/nix/store/1mmff759x0yixyqjc9i84zikqk4b9nii-python3.13-vllm-0.25.1

qwen38-flash-next-runtime-smoke
{"aiter":"0","moe_topk":"torch_native","sgl_kernel_loaded":false,"status":"ok"}

nix build .#hydraJobs.x86_64-linux.ci.source
/nix/store/w5vp68qz5rvdqpmwjdzpyv0v5x3gl6mi-nix-strix-halo-ci-source

nix build .#hydraJobs.x86_64-linux.ci.checks
/nix/store/mlqrvn96dsj5x81zh0df85aih5dkw4bv-nix-strix-halo-ci-checks
```

`nix flake check --accept-flake-config --no-build` also passed. The final
stacked branch's ROCm source-provider aggregate includes vLLM, SGLang, MLX,
DS4, and both llama.cpp tracks.
