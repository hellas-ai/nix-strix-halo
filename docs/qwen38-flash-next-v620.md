# Qwen3.8-Flash-Next on four V620s

The `sglang-qwen38-flash-next-rocm` package builds the pinned SGLang Qwen4-Exp
revision `73a255206f916366c8d26d4022f82ddfb0ab558d`. It is separate from the
released SGLang package. Transformers 5.12.1 and tokenizers 0.22.2 preserve the
API expected by that experimental source.

The gfx1030 serving configuration uses four V620s, expert-only symmetric INT4
with group size 32, FP16 activations, FP32 recurrent state, and host PLE
embedding offload. The Strix Halo iGPU must be excluded from the TP group.

```sh
nix build .#legacyPackages.x86_64-linux.gfx1030.sglang-qwen38-flash-next-rocm
closure=$(readlink -f result)
"$closure/bin/qwen38-flash-next-runtime-smoke"
MODEL=/models/Qwen3.8-Flash-Next-W4A16-G32 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  "$closure/bin/qwen38-flash-next-serve.sh"
```

Check the actual HIP device order before selecting devices. `MODEL` must point
to a complete checkpoint on a read-only model mount. The launcher defaults to
port 30800 and context length 32768. `ART` and `XDG_CACHE_HOME` select persistent
log and compiler-cache directories.

Scheduler overlap and CUDA graphs remain disabled. The pinned HIP QSA path
has shared ring-buffer hazards under the overlap scheduler. `/health` is
passive because generating from raw token zero is not a valid chat-template
probe; readiness also requires a real chat completion. The ROCm QSA prefill
patch uses one pipeline stage to fit the V620's 64 KiB shared memory.

The packaged inventory, expert quantization, FP16 audit, QSA compiler check,
and benchmark tools support checkpoint preparation and validation. Run each
with `--help` for its arguments. The BF16 staging helper pins the model
revision `f5d08274bafd880402bd16f5e3e6c514136ec06c` and verifies all 131 shards.
Keep the source checkpoint immutable and write converted checkpoints to a
separate directory.

The original package served chat requests on four V620s on 2026-08-28 with
serial scheduling and passive health. Those measurements apply to the old
closure. The 2026-09-29 refresh has package builds, focused QSA checks, and the
real model-import/MoE smoke test; it still needs a fresh four-V620 full-model
qualification after the hardware is available. Build success does not establish
new throughput or full-model correctness.
