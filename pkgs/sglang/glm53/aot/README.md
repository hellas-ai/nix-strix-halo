# Offline WMMA compilation

`default.nix` uses the supplied pinned `sglang-glm53-rocm` runtime and compiles
seven TP4 dense projection shapes for gfx1151 without GPU access. It produces
13 unique Triton cache entries: seven WMMA partial kernels and six sum kernels.
Its seed covers the observed 16-byte-aligned, 32-bit-pointer-range input
specializations with nonconstant row count and no Triton invalidating environment
variables. Other pointer attributes, toolchains, shapes, kernels, environment
values or targets may miss and compile normally. Graph capture still happens at
runtime. This is not a full-model AOT build.

The November 2 CPU-only build reproduced every one of the 14 per-shape partial
and sum cache keys from the strix-3 GPU synthetic C8 test; all corresponding
HSACO files matched byte for byte (one sum entry is shared by two shapes). The
Nix output also passed `nix build --rebuild` for reproducibility. The separately
executed GPU test ran the original runtime-compiled code and does not prove that
this seed was loaded. Seeded GPU replay, cache-hit evidence, startup savings and
full-model qualification remain separate tests.

`stage.nix` packages a small staging command. It verifies the seed and exact
runtime identity and installs to a new directory atomically. On filesystems without
no-replace rename, an advisory lock protects cooperating publishers. If the directory
already exists it preserves it, including later JIT additions. The caller must
choose a namespace keyed by the seed's Nix store name; never point the command at
a shared unversioned cache. The namespace must be writable by the service. The
seed is verified on every invocation, and it is never merged into an existing
cache. Triton performs its own specialization key checks on use.

For a launcher with Nix variables `aotSeed`, `aotStage`, and
`sglang-glm53-rocm`:

```sh
cache_base="${XDG_CACHE_HOME:-$HOME/.cache}/glm53/$(hostname)/triton-aot"
dest="$cache_base/$(basename "${aotSeed}")"
"${aotStage}/bin/glm53-aot-stage" --seed "${aotSeed}/seed" \
  --identity "${aotSeed}/identity.json" --runtime "${sglang-glm53-rocm}" --dest "$dest"
export TRITON_CACHE_DIR="$dest"
```

The packaged launcher now stages this seed by default before starting the
node script, which respects the preselected `TRITON_CACHE_DIR`. A failed
stage fails startup. `GLM_AOT=0` selects the ordinary runtime-versioned writable
cache; an explicitly supplied `TRITON_CACHE_DIR` also bypasses automatic
staging. These controls do not disable JIT or runtime graph capture.
