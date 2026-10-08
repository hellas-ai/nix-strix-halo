# DeepSeek V4 on gfx1151

The ROCm package supports the official Flash-0731 checkpoint's packed E2M1
expert weights and E8M0 scales without additional weight quantization. The
portable path uses Triton experts and mHC when `SGLANG_USE_AITER=0`; it selects
the portable paged indexer on devices without AITER paged-MQA support.

For this path, set `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` and disable
the TileLang mHC overrides with `SGLANG_OPT_USE_TILELANG_MHC_PRE=0` and
`SGLANG_OPT_USE_TILELANG_MHC_POST=0`. Use `--moe-runner-backend triton`,
`--fp8-gemm-backend triton`, `--disable-shared-experts-fusion`, and
`--kv-cache-dtype bfloat16`. Non-mmap loading uses safetensors' per-tensor
`pread` backend, which preserves native E8M0 storage.

Run the focused checks using the built package's `bin/sglang-python`:

- `lib/bench/ds4-loader-check.py`: native bytes through mmap and pread.
- `lib/bench/ds4-mhc-check.py`: independent FP64 reference and graph replay.
- `lib/bench/ds4-mxfp4-check.py`: independent unpacking, expert computation,
  routed scaling, and graph replay.

The component checks and a two-node text/tool-call smoke test passed. Full
coding-agent acceptance, long-context cache reuse, and full-model graph
execution remain unqualified. These fixes do not add support for the separate
Vision-Exp or V4.1 checkpoints.
