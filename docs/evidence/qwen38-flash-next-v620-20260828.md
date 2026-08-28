# Qwen3.8-Flash-Next V620 qualification evidence

Date: 2026-08-28

Host: `strix-2`

Hardware: four AMD Radeon Pro V620 cards, TP4, HIP devices `0,1,2,3`; the
Strix Halo iGPU was excluded.

Model: `/mnt/trex-models-fabric/Qwen3.8-Flash-Next-W4A16-G32`, read-only
publication of the 131-shard, 173.558 GiB expert-only W4A16-G32 checkpoint.

Package:
`/nix/store/m5k617a9fx8w5rv66nnnkznhyb334px2-sglang-rocm-gfx1030-0.5.17.dev0+qwen4exp.73a2552`

Unit: `codex-qwen38-flash-next-tp4-v26.service`

Raw log on the campaign host:
`.bench-artifacts/serve/serve-20260828-220237.log`

## Load and readiness

The server loaded all 131 shards, skipped 333 language-model-excluded visual
weights on every rank, allocated a 4,096-token cache, and reported:

```text
[2026-08-28 22:05:49 TP0] max_total_num_tokens=4096, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=1, context_len=4096, available_gpu_mem=8.50 GB
[2026-08-28 22:05:49] The server is fired up and ready to roll!
```

## External completions

The first request disabled thinking, used greedy decoding, and asked for an
exact response. The OpenAI-compatible endpoint returned HTTP 200 in 7.522666
seconds:

```json
{"content":"V620 FLASH WORKS","prompt_tokens":23,"completion_tokens":8,"reasoning_tokens":0}
```

The immediately following request had a 48-token prompt and asked for roughly
120 words explaining mixture-of-experts routing. It returned HTTP 200 in
10.789894 seconds with 146 completion tokens. The response began:

```text
Mixture-of-Experts (MoE) models decouple total parameter count from computational cost by employing a sparse gating mechanism.
```

The server recorded the padded 64-token QSA prefill tile and warm decode:

```text
[2026-08-28 22:06:26 TP0] Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 0, input throughput (token/s): 9.43
[2026-08-28 22:06:31 TP0] Decode batch, #running-req: 1, #full token: 128, gen throughput (token/s): 14.01
[2026-08-28 22:06:34 TP0] Decode batch, #running-req: 1, #full token: 192, gen throughput (token/s): 14.13
```

## Failure reproduced and closed

The preceding v24 run returned a valid short completion, then the longer
prompt reproduced the gfx1030 QSA prefill failure:

```text
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 67584, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
```

v26 used one Triton software-pipeline stage for ROCm QSA prefill while retaining
the upstream tile and warp count. Both the short completion and the formerly
failing longer prefill passed after that change. v24's successful short request
had already forced the HIP JIT build of the patched MoE alignment kernel, and
v26 executed the same patched path; no `sgl_kernel`, AITER, or NVIDIA-only
backend entered the serving path.

## Build gates

The exact reviewed revision passed:

```text
nix build .#legacyPackages.x86_64-linux.gfx1030.sglang-qwen38-flash-next-rocm
/nix/store/m5k617a9fx8w5rv66nnnkznhyb334px2-sglang-rocm-gfx1030-0.5.17.dev0+qwen4exp.73a2552

nix build .#sglang-qwen38-flash-next-rocm
/nix/store/2baakfxc4ri7qqqi3v290h01y5gdyq8q-sglang-rocm-gfx1151-0.5.17.dev0+qwen4exp.73a2552

nix build .#legacyPackages.x86_64-linux.gfx1030.vllm-rocm
/nix/store/lshnzk3143w5fdlggwm1gkbjhafrkgb2-python3.13-vllm-0.25.1

nix build .#vllm-rocm
/nix/store/1mmff759x0yixyqjc9i84zikqk4b9nii-python3.13-vllm-0.25.1

qwen38-flash-next-runtime-smoke
{"aiter":"0","moe_topk":"torch_native","sgl_kernel_loaded":false,"status":"ok"}

nix build .#hydraJobs.x86_64-linux.ci.source
/nix/store/w5vp68qz5rvdqpmwjdzpyv0v5x3gl6mi-nix-strix-halo-ci-source

nix build .#hydraJobs.x86_64-linux.ci.checks
/nix/store/h1vmxqng7gf16hjm011hl5fgrqp8ld7j-nix-strix-halo-ci-checks
```

`nix flake check --accept-flake-config --no-build` also passed. The final
stacked branch's ROCm source-provider aggregate includes vLLM, SGLang, MLX,
DS4, and both llama.cpp tracks.
